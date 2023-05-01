import numpy
import whitematteranalysis as wma
import os
import argparse
import mnist
import torch
from torchvision import transforms
import fibers
import torch.nn as nn
import fiber_distance
import copy
import vtk
import nets

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def convert_fiber_to_array(inputFile, numberOfFibers, fiberLength, numberOfFiberPoints, preproces=True,data='HCP'):
    if not os.path.exists(inputFile):
        print("<wm_cluster_from_atlas.py> Error: Input file", inputFile, "does not exist.")
        exit()
    print("\n==========================")
    print("input file:", inputFile)

    if numberOfFibers is not None:
        print("fibers to analyze per subject: ", numberOfFibers)
    else:
        print("fibers to analyze per subject: ALL")
    number_of_fibers = numberOfFibers
    fiber_length = fiberLength
    print("minimum length of fibers to analyze (in mm): ", fiber_length)
    points_per_fiber = numberOfFiberPoints
    print("Number of points in each fiber to process: ", points_per_fiber)

    # read data
    print("<wm_cluster_with_DEC.py> Reading input file:", inputFile)
    pd = wma.io.read_polydata(inputFile)

    if preproces:
        # preprocessing step: minimum length
        print("<wm_cluster_from_atlas.py> Preprocessing by length:", fiber_length, "mm.")
        pd2 = wma.filter.preprocess(pd, fiber_length, return_indices=False, preserve_point_data=True,
                                    preserve_cell_data=True, verbose=False)
    else:
        pd2 = pd

    # downsampling fibers if needed
    if number_of_fibers is not None:
        print("<wm_cluster_from_atlas.py> Downsampling to ", number_of_fibers, "fibers.")
        input_data = wma.filter.downsample(pd2, number_of_fibers, return_indices=False, preserve_point_data=True,
                                           preserve_cell_data=True, verbose=False)
    else:
        input_data = pd2

    fiber_array = fibers.FiberArray()
    fiber_array.convert_from_polydata(input_data, points_per_fiber=args.numberOfFiberPoints,data=data)
    feat = numpy.dstack((abs(fiber_array.fiber_array_r), fiber_array.fiber_array_a, fiber_array.fiber_array_s))
    feat_ROI = fiber_array.roi_list
    feat_surf_ve = fiber_array.fiber_surface_ve
    feat_surf_dk = fiber_array.fiber_surface_dk
    feat_surf_des = fiber_array.fiber_surface_des
    return input_data, feat, feat_ROI,feat_surf_ve,feat_surf_dk,feat_surf_des
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        # intersection = input_flat == target_flat
        # loss = ((intersection.sum(1) + smooth)).float() / (input_flat.size(1)+ smooth)
        intersection = input_flat * target_flat
        if torch.sum(intersection)==0:
            print('0')
        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1.3 - loss#.sum() / N
        return loss
class DiceScore(torch.nn.Module):
    def __init__(self):
        super(DiceScore, self).__init__()

    def forward(self, input, target):
        N = target.shape[0]
        smooth = 1
        # intersection = input_flat == target_flat
        # loss = ((intersection.sum(1) + smooth)).float() / (input_flat.size(1)+ smooth)
        intersection = input * target
        # if torch.sum(intersection)==0:
        #     print('0')
        loss = (2 * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        loss = loss.sum() / N
        return loss
def DB_index3(x_array,predicted):
    #Sprint('mask')
    cluster_id=numpy.unique(predicted)
    fiber_array = numpy.reshape(x_array, (len(x_array), -1, 3))
    alpha = []
    c = []
    flag_detected = numpy.zeros(len(cluster_id))
    for id,i in enumerate(list(cluster_id)):
        d_cluster=fiber_array[predicted==i]
        assert not len(d_cluster)==0
        if len(d_cluster) > 20:
            flag_detected[id] = 1
        if len(d_cluster) > 100:
            numpy.random.seed(12345)
            index = numpy.random.randint(0, len(d_cluster), 100)
            #print(index)
            d_cluster = d_cluster[index]
        distance_array = numpy.zeros((len(d_cluster), len(d_cluster)))
        distance_sum = numpy.zeros((len(d_cluster)))
        assert not numpy.isnan(c).any()
        for j in range(len(d_cluster)):
            fiber=d_cluster[j]
            distance = fiber_distance.fiber_distance(fiber, d_cluster)
            distance_array[j,:]=distance
            distance_sum[j] = numpy.sum(distance)
        c.append(d_cluster[numpy.argmin(distance_sum)])
        if len(d_cluster)==1:
            distance_clu=0
        else:
            distance_clu=numpy.sum(distance_array)/(len(d_cluster)*(len(d_cluster)-1))
        alpha.append(distance_clu)
        #alpha.append(numpy.mean(distance_array))
        assert not numpy.isnan(alpha).any()
    DB_all=[]
    dis_inter=[]
    for i in range(len(cluster_id)):
        alpha1=copy.deepcopy(alpha)
        c1 = copy.deepcopy(c)
        del c1[i]
        del alpha1[i]
        c1=numpy.array(c1)
        alpha1 = numpy.array(alpha1)
        temp=(alpha[i]+alpha1)/ (fiber_distance.fiber_distance(c[i], c1))
        DB_clu=numpy.max(temp)
        DB_all.append(DB_clu)
        dis_inter_clu=numpy.min(fiber_distance.fiber_distance(c[i], c1))
        dis_inter.append(dis_inter_clu)
    DB_all=numpy.array(DB_all)
    DB_all1=DB_all[numpy.where(flag_detected == 1)]
    DB = numpy.mean(DB_all1)
    dis_intra=numpy.array(alpha)
    dis_inter=numpy.array(dis_inter)
    return DB,DB_all,dis_intra,dis_inter
def calculate_predictions_roi(model, dataloader, params,roi_cluster,surf_cluster,surf_flag=False):
    loss_fn=DiceLoss()
    model.eval()
    preds=None
    probs=None
    for data in dataloader:
        input1,input2,sim_score,roi_bat,surf_bat,_ = data
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])
        roi_bat = roi_bat.to(params['device'])
        surf_bat = surf_bat.to(params['device'])
        outputs, clusters, _,embedding,_,dis_point = model(input1,input2)

        roi_3D = roi_bat.unsqueeze(1).to(params['device'])
        roi_3D = roi_3D.repeat(1, model.num_clusters, 1)
        roi_3D = torch.reshape(roi_3D, (-1, roi_3D.shape[-1]))
        roi_cluster_3D = roi_cluster.repeat(outputs.shape[0], 1)
        dis_roi1 = loss_fn(roi_3D, roi_cluster_3D)
        dis_roi = torch.reshape(dis_roi1, (outputs.shape[0], model.num_clusters))
        _, pred_point = dis_point.min(1)
        _, pred_roi = dis_roi.min(1)

        if surf_flag:
            if surf_bat.shape[1]>surf_cluster.shape[1]:
                surf_cluster=torch.cat([surf_cluster,torch.zeros(surf_cluster.shape[0],surf_bat.shape[1]-surf_cluster.shape[1])],dim=1)
            # print(surf_bat.size())
            # print(surf_cluster.size())
            dis_surf = 1 - torch.mm(surf_bat.float(), surf_cluster.t().float())/1.5

            # dis_surf = torch.ones((dis_roi.shape[0], dis_roi.shape[1]))
            # for i_s in range(surf_bat.shape[0]):
            #     surf_fiber = surf_bat[i_s]
            #     inds = surf_fiber.nonzero().squeeze()
            #     if inds.shape == torch.Size([0]):  # no
            #         pass
            #     elif inds.shape == torch.Size([]):  # one
            #         dis_surf[i_s] = 1 - surf_cluster[:, inds]
            #     elif inds.shape == torch.Size([2]):  # two
            #         dis_surf[i_s] = 1 - torch.sum(surf_cluster[:, inds], 1)
            #     else:
            #         print('Error: more than two labels for a fiber', inds)
            #         exit()
            dis_surf = dis_surf.to(params['device'])
            x = 1.0 + dis_point * dis_roi * dis_surf
        else:
            x = 1.0 + dis_point * dis_roi
        x = 1.0 / x
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        clusters = x
        probs_single,preds_single=torch.max(clusters,1)
        if preds is not None:
            preds=numpy.concatenate((preds, preds_single.cpu().detach().numpy()), 0)
            probs = numpy.concatenate((probs, probs_single.cpu().detach().numpy()), 0)
        else:
            preds = preds_single.cpu().detach().numpy()
            probs = probs_single.cpu().detach().numpy()
    return  preds,probs
def calculate_predictions_test(model, dataloader, params):
    preds=None
    probs=None
    model.eval()
    for data in dataloader:
        input1,input2,sim_score,_,_,_ = data
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])
        x_rec, clusters, _,embedding,_,_ = model(input1,input2)
        probs_single,preds_single=torch.max(clusters,1)
        if preds is not None:
            preds=numpy.concatenate((preds, preds_single.cpu().detach().numpy()), 0)
            probs = numpy.concatenate((probs, probs_single.cpu().detach().numpy()), 0)
        else:
            preds = preds_single.cpu().detach().numpy()
            probs = probs_single.cpu().detach().numpy()

    return  preds,probs #x_recs, embeddings
def cluster_save(pd_c_list, outdir, input_pd, cluster_numbers_s, number_of_clusters, cluster_colors):
                # if args.fs:
                #     outdir=outdir+'_fs'
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                print('<wm_cluster_atlas.py> Saving output cluster files in directory:', outdir)
                cluster_sizes = list()
                cluster_fnames = list()
                fnames = list()
                # cluster_colors = list()
                for c in range(number_of_clusters):
                    mask = cluster_numbers_s == c
                    cluster_size = numpy.sum(mask)
                    cluster_sizes.append(cluster_size)
                    # pd_c = wma.filter.mask(output_polydata_s, mask, preserve_point_data=True, preserve_cell_data=True,verbose=False)
                    pd_c = pd_c_list[c]
                    # The clusters are stored starting with 1, not 0, for user friendliness.
                    fname_c = 'cluster_{0:05d}.vtp'.format(c + 1)
                    # save the filename for writing into the MRML file
                    fnames.append(fname_c)
                    # prepend the output directory
                    fname_c = os.path.join(outdir, fname_c)
                    cluster_fnames.append(fname_c)
                    wma.io.write_polydata(pd_c, fname_c)

                # Notify user if some clusters empty
                print(
                    "<wm_cluster_atlas.py> Checking for empty clusters (can be due to anatomical variability or too few fibers analyzed).")
                for sz, fname in zip(cluster_sizes, cluster_fnames):
                    if sz == 0:
                        print(sz, ":", fname)

                cluster_sizes = numpy.array(cluster_sizes)
                print("<wm_cluster_from_atlas.py> Mean number of fibers per cluster:", numpy.mean(cluster_sizes),
                      "Range:",
                      numpy.min(cluster_sizes), "..", numpy.max(cluster_sizes))
                # Also write one with 100%% of fibers displayed
                fname = os.path.join(outdir, 'clustered_tracts_display_100_percent.mrml')
                wma.mrml.write(fnames, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=0.1)
                render = True
                # View the whole thing in png format for quality control
                if render:

                    try:
                        print('<wm_cluster_from_atlas.py> Rendering and saving images of clustered subject.')
                        ren = wma.render.render(input_pd, 1000, data_mode='Cell', data_name='EmbeddingColor',
                                                verbose=False)
                        ren.save_views(outdir)
                        del ren
                    except:
                        print('<wm_cluster_from_atlas.py> No X server available.')

                print("\n==========================")
                print('<wm_cluster_from_atlas.py> Done clustering subject.  See output in directory:\n ', outdir, '\n')
def add_prob(inpd, prob):
    vtk_array = vtk.vtkDoubleArray()
    vtk_array.SetName('Prob')
    inpd.GetLines().InitTraversal()
    for lidx in range(0, inpd.GetNumberOfLines()):
        ptids = vtk.vtkIdList()
        inpd.GetLines().GetNextCell(ptids)
        prob_line = prob[lidx]
        for pidx in range(0, ptids.GetNumberOfIds()):
            vtk_array.InsertNextTuple1(prob_line)
    inpd.GetPointData().AddArray(vtk_array)
    return inpd
def metrics_calculation(predicted, x_arrays, x_fs, x_surf):
    loss_fn = DiceScore()
    def tapc_calculation1(num_clusters, preds, roi_fs):
        roi_cluster = numpy.zeros([num_clusters, roi_fs.shape[1]])
        tapc_all = []
        for i in range(num_clusters + 1):
            t = roi_fs[preds == i]
            if t.size == 0:
                continue
            else:
                t1 = numpy.sum(t, 0)
                roi_all = numpy.where(t1 > t.shape[0] * 0.4)[0]
                if 0 in roi_all:
                    roi_all = roi_all[1:]
                roi_cluster[i, roi_all] = 1
                roi_preds = numpy.repeat(roi_cluster[i].reshape((1, len(roi_cluster[i]))), t.shape[0], axis=0)
                tapc = loss_fn(t, roi_preds)
                tapc_all.append(tapc)
        # roi_preds = roi_cluster[preds]
        # tapc = loss_fn(roi_fs, roi_preds)
        tapc = numpy.mean(tapc_all)
        return tapc, numpy.array(tapc_all)

    def tspc_calculation1(num_clusters, preds, ds_surf_onehot):
        tspc_sub = []
        N_surf_all = []
        for i in range(num_clusters + 1):
            t = ds_surf_onehot[preds == i]
            if t.size == 0:
                continue
            else:
                t1 = numpy.sum(t, 0)
                if t1.sum() == 0:
                    continue
                surf_cluster = t1 / t1.sum()
                tspc_all = surf_cluster * t
                tspc1 = numpy.sum(tspc_all, 1)
                tspc_clu = numpy.mean(tspc1)

                surf_all = numpy.where(t1 > 0)[0]
                N_surf_all.append(len(surf_all))
                tspc_sub.append(tspc_clu)
        tspc = numpy.array(tspc_sub).mean()
        N_surf_all = numpy.array(N_surf_all).mean()
        return tspc, N_surf_all, numpy.array(tspc_sub)

    tapc_train, tapc_all = tapc_calculation1(num_clusters, predicted, x_fs)
    tspc_train, _, tspc_all = tspc_calculation1(num_clusters, predicted, x_surf)
    # print(x_arrays.shape)
    # print(predicted.shape)
    DB_score, DB_all, dis_intra, dis_inter = DB_index3(x_arrays, predicted)
    n_detected = 0
    flag_detected = numpy.zeros(num_clusters)
    for n in range(num_clusters):
        n_fiber = numpy.sum(predicted == n)
        if n_fiber >= 20:
            n_detected += 1
            flag_detected[n] = 1
    wmpg_train = n_detected / num_clusters
    print('DB: {0:.4f}\tWMPG: {1:.4f}\tTAPC: {2:.4f}\tTSPC: {3:.4f}'.format(DB_score, wmpg_train, tapc_train,tspc_train))
    return DB_score, wmpg_train, tapc_train, tspc_train
def one_hot_encoding(ds_fs):
    roi_map = numpy.load('relabel_map.npy')
    ds_fs_onehot = numpy.zeros((len(ds_fs), len(numpy.unique(roi_map[1])))).astype(numpy.float32)
    if not isinstance(ds_fs, list):
        roi_unique = numpy.unique(ds_fs)
        print(roi_unique)
        print(roi_map[0])
        assert set(roi_unique).issubset(set(roi_map[0]))
        for roi in roi_unique:
            roi_new = roi_map[1][roi_map[0] == roi]
            ds_fs[ds_fs == roi] = roi_new
        for f in range(ds_fs.shape[0]):
            roi_single = numpy.unique(ds_fs[f])
            if roi_single[0] == 0:
                roi_single = roi_single[1:]
            ds_fs_onehot[f, roi_single.astype(int)] = 1
    else:
        for f, roi_fiber in enumerate(ds_fs):
            roi_unique = numpy.unique(roi_fiber)
            assert set(roi_unique).issubset(set(roi_map[0]))
            for roi in roi_unique:
                roi_new = roi_map[1][roi_map[0] == roi]
                roi_fiber[roi_fiber == roi] = roi_new
            roi_single = numpy.unique(roi_fiber)
            if roi_single[0] == 0:
                roi_single = roi_single[1:]
            ds_fs_onehot[f, roi_single.astype(int)] = 1
    return ds_fs_onehot
def surf_encoding(fiber_surf_dk):
    fiber_surfs = fiber_surf_dk.astype(int)
    surf_labels = numpy.unique(fiber_surfs)
    surf_map = numpy.load('dk_map.npy')
    if surf_labels[0] == 0:
        surf_labels = surf_labels[1:]
    for surf_label in surf_labels:
        fiber_surfs[numpy.where(fiber_surfs == surf_label)] = numpy.where(surf_map == surf_label)
    ds_surf_onehot_dk = numpy.zeros((len(fiber_surfs), len(surf_map)))
    for s in range(len(fiber_surfs)):
        if fiber_surfs[s][1] == 0:
            fiber_surfs[s][1] = fiber_surfs[s][0]
        if fiber_surfs[s][0] == 0:
            fiber_surfs[s][0] = fiber_surfs[s][1]
        if fiber_surfs[s][1] == 0 and fiber_surfs[s][0] == 0:
            continue
        ds_surf_onehot_dk[s, fiber_surfs[s]] = 1
    return ds_surf_onehot_dk

parser = argparse.ArgumentParser(description='Use DFC for clustering')
parser.add_argument(
    '-indirt', action="store", dest="inputDirectoryt", default="data/test",
    help='path of tractography as vtkPolyData (.vtk or .vtp) from testing subjects.')
parser.add_argument(
    '-outdir', action="store", dest="outputDirectory", default="results",
    help='Output folder of clustering results.')
parser.add_argument(
    '-modeldir', action="store", dest="modelDirectory", default="nets/model.pt",
    help='Output folder of clustering results.')
parser.add_argument(
    '-l', action="store", dest="fiberLength", type=int, default=40,
    help='Minimum length (in mm) of fibers to analyze. 40mm is default.')
parser.add_argument(
    '-p', action="store", dest="numberOfFiberPoints", type=int, default=14,
    help='Number of points in each fiber to process. 14 is default.')
parser.add_argument('--fs', default=True, type=str2bool, help='inporparating freesurfer information')
parser.add_argument('--surf', default=True, type=str2bool, help='inporparating cortical information')
parser.add_argument('--ro', default=True, type=str2bool, help='outlier removal')
parser.add_argument('--save', default=False, type=str2bool, help='whether save clustering results')
parser.add_argument('--num_clusters', default=800, type=int, help='number of clusters')
parser.add_argument('--embedding_dimension', default=10, type=int, help='number of embeddings')
parser.add_argument('--idx', default=True, type=str2bool, help='idx for dgcnn')
parser.add_argument('--k', default=5, type=int, help='k for dgcnn')
parser.add_argument('--net_architecture', default='DGCNN',
                    choices=['CAE_3', 'CAE_pair', 'CAE_bn3', 'CAE_4', 'CAE_bn4', 'CAE_5', 'DGCNN', 'PointNet',
                             'CAE_DG_pair', 'GCN'], help='network architecture used')
parser.add_argument('--dataset', default='Fiber',
                    choices=['MNIST-train', 'custom', 'MNIST-test', 'MNIST-full', 'Fiber', 'FiberMap', 'FiberCom'],
                    help='custom or prepared dataset')
parser.add_argument('--data', default='HCP',
                    choices=['HCP', 'PPMI', 'open_fMRI'],
                    help='custom or prepared dataset')
parser.add_argument('--batch_size', default=1024, type=int, help='batch size')


args = parser.parse_args()
data_dirt=args.inputDirectoryt
batch = args.batch_size
workers = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_clusters = args.num_clusters
params = {'device': device}
num_points = args.numberOfFiberPoints
model_name = args.net_architecture
model_path = args.modelDirectory

DB_all = []
wmpg_all = []
tapc_all = []
tspc_all = []
DB_all_pre = []
wmpg_all_pre = []
tapc_all_pre = []
tspc_all_pre = []
DB_all_ro = []
wmpg_all_ro = []
tapc_all_ro = []
tspc_all_ro = []
rate_left_all = []
input_pd_fnames = wma.io.list_vtk_files(data_dirt)
for i in range(len(input_pd_fnames)):
    subject_id = os.path.split(input_pd_fnames[i])[1].split('.')[0]
    input_pd, x_array, d_roi, fiber_surf_ve, fiber_surf_dk, fiber_surf_des = \
        convert_fiber_to_array(input_pd_fnames[i], numberOfFibers=None, fiberLength=args.fiberLength,
                               numberOfFiberPoints=args.numberOfFiberPoints, preproces=True, data=args.data)
    ds_fs_onehot = one_hot_encoding(d_roi)
    surf_dk_onehot = surf_encoding(fiber_surf_dk)

    if args.dataset == 'FiberMap':
        dataset = mnist.FiberMap_pair(x_array, ds_fs_onehot, surf_dk_onehot,
                                      transform=transforms.Compose([transforms.ToTensor()]))
    elif args.dataset == 'Fiber':
        dataset = mnist.Fiber_pair(x_array, ds_fs_onehot, surf_dk_onehot,
                                   transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)
    dataset_size = len(dataset)
    print("Testing set size:\t" + str(dataset_size))

    if args.idx:
        idxf=torch.zeros((num_points,args.k),dtype=torch.int64,device=device)
        if args.k==5:
            idxf[:,0]=torch.tensor(range(num_points))
            idxf[:, 1]=torch.tensor(range(num_points))-2
            idxf[:, 2] = torch.tensor(range(num_points)) - 1
            idxf[:, 3] = torch.tensor(range(num_points)) + 1
            idxf[:, 4] = torch.tensor(range(num_points)) + 2
        elif args.k==3:
            idxf[:,0]=torch.tensor(range(num_points))
            idxf[:, 1] = torch.tensor(range(num_points)) - 1
            idxf[:, 2] = torch.tensor(range(num_points)) + 1
        idxf[idxf<0]=0
        idxf[idxf>num_points-1]=num_points-1
        idx=idxf.repeat(batch,1,1)
    else:
        idx=None
    to_eval = "nets." + model_name + "(k=args.k,input_channel=3,num_clusters=num_clusters,embedding_dimension=args.embedding_dimension,idx=idx)"
    model = eval(to_eval)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    roi_cluster=numpy.load('profiles/roi_cluster_fs.npy')
    surf_cluster = numpy.load('profiles/surf_cluster_fs.npy')
    roi_cluster = torch.tensor(roi_cluster).to(device)
    surf_cluster = torch.tensor(surf_cluster).to(device)

    if args.fs:
        preds, probs = calculate_predictions_roi(model, dataloader, params, roi_cluster,
                                                 surf_cluster, surf_flag=args.surf)
    else:
        preds, probs = calculate_predictions_test(model, dataloader, params)

    DB_score, wmpg, tapc, tspc = metrics_calculation(preds, x_array, ds_fs_onehot, surf_dk_onehot)
    DB_all.append(DB_score)
    wmpg_all.append(wmpg)
    tapc_all.append(tapc)
    tspc_all.append(tspc)

    if args.ro:
        num_stds = [0.7]  # HCP  f_IRM; 0.83; open_fMRI: 0.85
        DB_score_ro_thr = numpy.zeros(len(num_stds))
        wmpg_ro_thr = numpy.zeros(len(num_stds))
        tapc_ro_thr = numpy.zeros(len(num_stds))
        tspc_ro_thr = numpy.zeros(len(num_stds))
        rate_left_thr = numpy.zeros(len(num_stds))
        preds_fs_surf = preds
        probs_fs_surf = probs
        for i, num_std in enumerate(num_stds):
            rate_removed_clu = numpy.zeros(num_clusters)
            if num_stds[i] < 0.1:
                id_reject = numpy.where(probs_fs_surf < num_stds[i])[0]
            else:
                id_reject = []
                probs_rejected = []
                threshold = numpy.zeros(num_clusters)
                for ic in range(num_clusters):
                    index = numpy.where(preds_fs_surf == ic)[0]
                    probc = probs_fs_surf[index]
                    # numpy.save('probabilities/testing/pobc_{:03d}.npy'.format(ic), probc)
                    if len(probc) > 0:
                        # mean = mean_atlas[ic]
                        # std = std_atlas[ic]
                        # index1 = numpy.where((mean - probc) > num_std * std)[0]
                        index1 = numpy.where((probc.mean() - probc) > num_std * probc.std())[0]
                        if len(index1) > 0:
                            id_rejectc = index[index1]
                            id_reject.extend(id_rejectc)
                            prob_rejected = probc[index1]
                            probs_rejected.append(prob_rejected)
                            # rate_removed_clu[ic] = len(id_rejectc) / len(probc)
                        # else:
                        #     print(ic)
                # numpy.save('threshold_test.npy',threshold)
                id_reject = numpy.array(id_reject)
            # numpy.save('debug_ro/rate_removed_clu_{}.npy'.format(i), rate_removed_clu)
            probs_reject = probs_fs_surf[id_reject]
            if id_reject is not None:
                temp = numpy.ones(len(preds_fs_surf))
                temp[id_reject] = 0
                mask = temp > 0
                x_array_ro = x_array[mask]
                preds_fs_surf_ro = preds_fs_surf[mask]
                probs_fs_surf_ro = probs_fs_surf[mask]
                ds_fs_onehot_ro = ds_fs_onehot[mask]
                ds_surf_onehot_ro = surf_dk_onehot[mask]
                print('outlier_removed:')
                DB_score_ro, wmpg_ro, tapc_ro, tspc_ro = metrics_calculation(preds_fs_surf_ro, x_array_ro,
                                                                             ds_fs_onehot_ro, ds_surf_onehot_ro)
                rate_left = 1 - len(preds_fs_surf_ro) / len(preds_fs_surf)
                print('fiber_removed:', rate_left)
                DB_score_ro_thr[i] = DB_score_ro
                wmpg_ro_thr[i] = wmpg_ro
                tapc_ro_thr[i] = tapc_ro
                tspc_ro_thr[i] = tspc_ro
                rate_left_thr[i] = rate_left
                DB_all_ro.append(DB_score_ro_thr)
                wmpg_all_ro.append(wmpg_ro_thr)
                tapc_all_ro.append(tapc_ro_thr)
                tspc_all_ro.append(tspc_ro_thr)
                rate_left_all.append(rate_left_thr)

            if args.save:
                outdir = args.outputDirectory + '/' + subject_id
                print(outdir)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                # numpy.savez(os.path.join(outdir, subject_id + '_measure.npz'), DB_score_ro, wmpg_ro, tapc_ro, tspc_ro, rate_left)

                maskc = numpy.ones(len(preds))
                maskc[id_reject] = 0
                pd_c = wma.filter.mask(input_pd, maskc, preserve_point_data=True, preserve_cell_data=True,
                                       verbose=False)
                input_pd1 = add_prob(pd_c, probs_fs_surf_ro)
                cluster_colors = numpy.random.randint(0, 255, (num_clusters, 3))
                pd_c_list = wma.cluster.mask_all_clusters(input_pd1, preds_fs_surf_ro,
                                                          num_clusters,
                                                          preserve_point_data=True,
                                                          preserve_cell_data=True,
                                                          verbose=False)
                cluster_save(pd_c_list, outdir, input_pd1, preds_fs_surf_ro, num_clusters, cluster_colors)

print('final results:')
DB_score = numpy.array(DB_all).mean()
wmpg = numpy.array(wmpg_all).mean()
tapc = numpy.array(tapc_all).mean()
tspc = numpy.array(tspc_all).mean()
print(numpy.array(DB_all), numpy.array(wmpg_all), numpy.array(tapc_all), numpy.array(tspc_all))
print(DB_score, wmpg, tapc, tspc)

if args.ro:
    print('outlier removal results:')
    DB_score_ro = numpy.mean(numpy.array(DB_all_ro), 0)
    wmpg_ro = numpy.mean(numpy.array(wmpg_all_ro), 0)
    tapc_ro = numpy.mean(numpy.array(tapc_all_ro), 0)
    tspc_ro = numpy.mean(numpy.array(tspc_all_ro), 0)
    rate_left_ro = numpy.mean(numpy.array(rate_left_all), 0)
    print(DB_score_ro,wmpg_ro,tapc_ro,tspc_ro,rate_left_ro)



