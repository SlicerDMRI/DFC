from __future__ import print_function, division
import numpy
import whitematteranalysis as wma
#import tract_feat
import vtk
from sklearn import metrics
import argparse
import torch
from torchvision import datasets, models, transforms
import os
import nets
import mnist
from torch.utils.tensorboard import SummaryWriter
from training_functions_fiber_pair import  calculate_predictions_test
import  fiber_distance
import copy
import time
import fibers

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
class DiceLoss(torch.nn.Module):
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
def convert_fiber_to_array(inputFile, numberOfFibers, fiberLength, numberOfFiberPoints, preproces=False,data='HCP'):
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
def cluster_save(pd_c_list, outdir, input_pd, cluster_numbers_s, number_of_clusters, cluster_colors):
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
            wma.mrml.write(fnames, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=1.0)
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

def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
# def DB_index(x_array,predicted):
#     #Sprint('mask')
#     cluster_id=numpy.unique(predicted)
#     fiber_array = numpy.reshape(x_array, (len(x_array), -1, 3))
#     alpha = []
#     c = []
#     for i in list(cluster_id):
#         d_cluster=fiber_array[predicted==i]
#         assert not len(d_cluster)==0
#         if len(d_cluster) > 100:
#             numpy.random.seed(12345)
#             index = numpy.random.randint(0, len(d_cluster), 100)
#             #print(index)
#             d_cluster = d_cluster[index]
#         distance_array = numpy.zeros((len(d_cluster), len(d_cluster)))
#         distance_sum = numpy.zeros((len(d_cluster)))
#         assert not numpy.isnan(c).any()
#         for j in range(len(d_cluster)):
#             fiber=d_cluster[j]
#             distance = fiber_distance.fiber_distance(fiber, d_cluster)
#             distance_array[j,:]=distance
#             distance_sum[j] = numpy.sum(distance)
#         c.append(d_cluster[numpy.argmin(distance_sum)])
#         if len(d_cluster)==1:
#             distance_clu=0
#         else:
#             distance_clu=numpy.sum(distance_array)/(len(d_cluster)*(len(d_cluster)-1))
#         alpha.append(distance_clu)
#         assert not numpy.isnan(alpha).any()
#     DB_all=[]
#     for i in range(len(cluster_id)):
#         alpha1=copy.deepcopy(alpha)
#         c1 = copy.deepcopy(c)
#         del c1[i]
#         del alpha1[i]
#         c1=numpy.array(c1)
#         alpha1 = numpy.array(alpha1)
#         temp=(alpha[i]+alpha1)/ (fiber_distance.fiber_distance(c[i], c1))
#         DB_clu=numpy.max(temp)
#         DB_all.append(DB_clu)
#     DB=numpy.mean(DB_all)
#     return  DB,numpy.array(DB_all)
def DB_index(x_array,predicted):
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
        assert not numpy.isnan(alpha).any()
    DB_all=[]
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
    DB_all=numpy.array(DB_all)
    DB_all1=DB_all[numpy.where(flag_detected == 1)]
    DB = numpy.mean(DB_all1)
    return  DB,DB_all
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
def calculate_predictions_roi(model, dataloader, params,surf_flag=False):
    loss_fn=DiceLoss()

    print('calculate_predictions_roi')
    roi_cluster=numpy.load('roi_cluster_fs.npy')
    roi_cluster=torch.tensor(roi_cluster).to(params['device'])
    if surf_flag:
        surf_cluster = numpy.load('surf_cluster_fs.npy')
        surf_cluster = torch.tensor(surf_cluster).to(params['device'])

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
        #n_label=64
        # roi_batch = torch.zeros([outputs.shape[0], n_label])
        # for j in range(outputs.shape[0]):
        #     roi_onehot = torch.zeros([n_label]).to(params['device'])
        #     roi_label = torch.unique(roi_bat[j])
        #     if 0 in roi_label:
        #         index_0 = roi_label != 0
        #         roi_label = roi_label[index_0]
        #     roi_onehot[roi_label.long()] = 1
        #     roi_batch[j] = roi_onehot

        roi_3D = roi_bat.unsqueeze(1).to(params['device'])
        roi_3D = roi_3D.repeat(1, model.num_clusters, 1)
        roi_3D = torch.reshape(roi_3D, (-1, roi_3D.shape[-1]))
        roi_cluster_3D = roi_cluster.repeat(outputs.shape[0], 1)
        dis_roi1 = loss_fn(roi_3D, roi_cluster_3D)
        dis_roi = torch.reshape(dis_roi1, (outputs.shape[0], model.num_clusters))
        # roi_3D = roi_his.unsqueeze(1).to(params['device'])
        # roi_3D = roi_3D.repeat(1, model.num_clusters, 1)
        # roi_3D = torch.reshape(roi_3D, (-1, roi_3D.shape[-1]))
        # roi_his_cluster_3D = roi_cluster.repeat(outputs.shape[0], 1)
        # roi_sim = torch.sum(roi_3D * roi_his_cluster_3D, 1) * torch.sum((roi_3D * roi_his_cluster_3D > 0), 1)
        # # dis_his_roi1 = loss_fn(roi_3D, roi_his_cluster_3D)
        # sim_his_roi = torch.reshape(roi_sim, (outputs.shape[0], model.num_clusters))
        # dis_roi = 1 - sim_his_roi / sim_his_roi.max() + 0.5
        # calculate new p
        # alpha = dis_point.max()
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
        #_, preds = torch.max(clusters, 1)

        probs_single,preds_single=torch.max(clusters,1)
        if preds is not None:
            preds=numpy.concatenate((preds, preds_single.cpu().detach().numpy()), 0)
            probs = numpy.concatenate((probs, probs_single.cpu().detach().numpy()), 0)
        else:
            preds = preds_single.cpu().detach().numpy()
            probs = probs_single.cpu().detach().numpy()

        # if output_array is not None:
        #     output_array = np.concatenate((output_array, clusters.cpu().detach().numpy()), 0)
        # else:
        #     output_array = clusters.cpu().detach().numpy()

    # preds = np.argmax(output_array.data, axis=1)
    # probs = np.max(output_array.data, axis=1)
    # print(output_array.shape)
    return  preds,probs
if __name__ == "__main__":
    #start = time.time()
    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument(
        '-indir', action="store", dest="inputDirectory",
        default="/home/yuqian/Dropbox (Partners HealthCare)/YuqianFanSharedFolder/DFC_code/data/test",
        help='A file of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument(
        '-outdir', action="store", dest="outputDirectory", default="./results",
        help='Output folder of clustering results.')
    parser.add_argument(
        '-modeldir', action="store", dest="modelDirectory", default="nets/DGCNN.pt",
        help='Output folder of clustering results.')
    parser.add_argument(
        '-tef', action="store", dest="numberOfFibers_test", type=int, default=None,
        help='Number of fibers of each testing data to analyze from each subject.')
    parser.add_argument(
        '-l', action="store", dest="fiberLength", type=int, default=40,
        help='Minimum length (in mm) of fibers to analyze. 60mm is default.')
    parser.add_argument(
        '-p', action="store", dest="numberOfFiberPoints", type=int, default=14,
        help='Number of points in each fiber to process. 10 is default.')
    parser.add_argument('--fs', default=True, type=str2bool, help='inporparating freesurfer information')
    parser.add_argument('--ro', default=True, type=str2bool, help='remove outliers')
    parser.add_argument('--save', default=False, type=str2bool, help='remove outliers')
    parser.add_argument('--net_architecture', default='DGCNN',
                        choices=['CAE_3', 'CAE_bn3', 'CAE_4', 'CAE_bn4', 'CAE_5', 'CAE_bn5'],
                        help='network architecture used')
    parser.add_argument('--dataset', default='Fiber',
                        choices=['MNIST-train', 'custom', 'MNIST-test', 'MNIST-full','Fiber','FiberMap'],
                        help='custom or prepared dataset')
    parser.add_argument('--batch_size', default=2048, type=int, help='batch size')
    parser.add_argument('--num_clusters', default=800, type=int, help='number of clusters')
    parser.add_argument('--embedding_dimension', default=10, type=int, help='number of embeddings')
    parser.add_argument('--leaky', default=True, type=str2bool)
    parser.add_argument('--neg_slope', default=0.01, type=float)
    parser.add_argument('--data', default='HCP',
                        choices=['HCP', 'PPMI', 'open_fMRI'])
    args = parser.parse_args()
    batch = args.batch_size
    model_name = args.net_architecture
    num_clusters=args.num_clusters
    workers = 4
    data_dir = args.inputDirectory
    modelDirectory = args.modelDirectory
    print("\nData preparation\nReading data from:\t./" + data_dir)
    #device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Evaluate the proper model
    if args.net_architecture=='CAE_3_pair':
        to_eval = "nets." + model_name + "(img_size, num_clusters=num_clusters,embedding_dimension=args.embedding_dimension, leaky = args.leaky, neg_slope = args.neg_slope)"
    else:
        to_eval = "nets." + model_name + "(input_channel=3,num_clusters=num_clusters,embedding_dimension=args.embedding_dimension)"
    model = eval(to_eval)
    model.load_state_dict(torch.load(modelDirectory))
    model.eval()
    model = model.to(device)
    params = {'device': device}
    loss_fn=DiceScore()
    input_pd_fnames = wma.io.list_vtk_files(data_dir)
    num_pd = len(input_pd_fnames)
    DB_all = []
    wmpg_all = []
    tapc_all=[]
    tapc1_all = []
    DB_all_fs = []
    wmpg_all_fs = []
    tapc_all_fs=[]
    tapc1_all_fs = []
    DB_all_ro = []
    wmpg_all_ro = []
    tapc_all_ro=[]
    tspc_all_ro = []
    tapc1_all_ro = []
    fiber_number_cluster_all=[]
    rate_left_all=[]
    cluster_colors = numpy.random.randint(0, 255, (num_clusters, 3))
    for i in range(len(input_pd_fnames)):
        start = time.time()
        subject_id = os.path.split(input_pd_fnames[i])[1].split('.')[0]
        input_pd, x_array, d_roi, fiber_surf_ve, fiber_surf_dk, fiber_surf_des = \
            convert_fiber_to_array(input_pd_fnames[i], numberOfFibers=None, fiberLength=args.fiberLength,
                                   numberOfFiberPoints=args.numberOfFiberPoints, preproces=False, data=args.data)


        def one_hot_encoding(ds_fs):
            roi_map = numpy.load('relabel_map.npy')
            ds_fs_onehot = numpy.zeros((len(ds_fs), len(numpy.unique(roi_map[1])))).astype(numpy.float32)
            if not isinstance(ds_fs, list):
                roi_unique = numpy.unique(ds_fs)
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
        ds_fs_onehot = one_hot_encoding(d_roi)
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
        surf_dk_onehot = surf_encoding(fiber_surf_dk)
        dataset = mnist.Fiber_pair(x_array,ds_fs_onehot,surf_dk_onehot, transform=transforms.Compose([transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)
        dataset_size = len(dataset)
        print("Training set size:\t" + str(dataset_size))
        time_subject = time.time() - start
        print('time_data', time_subject)
        since = time.time()
        #preds, _ = calculate_predictions_test(model, dataloader, params)
        preds, probs = calculate_predictions_roi(model, dataloader, params,surf_flag=True)

        elapsed=time.time()-since
        print('prediction time:',elapsed)

        def metrics_calculation(predicted,x_arrays,x_fs,x_surf):
            loss_fn = DiceScore()
            def tapc_calculation1(num_clusters, preds, roi_fs):
                roi_cluster = numpy.zeros([num_clusters, roi_fs.shape[1]])
                tapc_all=[]
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
                        roi_preds=numpy.repeat(roi_cluster[i].reshape((1,len(roi_cluster[i]))),t.shape[0],axis=0)
                        tapc=loss_fn(t, roi_preds)
                        tapc_all.append(tapc)
                # roi_preds = roi_cluster[preds]
                # tapc = loss_fn(roi_fs, roi_preds)
                tapc=numpy.mean(tapc_all)
                return tapc,numpy.array(tapc_all)
            def tspc_calculation1(num_clusters,preds,ds_surf_onehot):
                tspc_sub=[]
                N_surf_all=[]
                for i in range(num_clusters + 1):
                    t = ds_surf_onehot[preds == i]
                    if t.size == 0:
                        continue
                    else:
                        t1 = numpy.sum(t, 0)
                        if t1.sum()==0:
                            continue
                        surf_cluster = t1 / t1.sum()
                        tspc_all = surf_cluster * t
                        tspc1 = numpy.sum(tspc_all, 1)
                        tspc_clu=numpy.mean(tspc1)

                        surf_all = numpy.where(t1 > 0)[0]
                        N_surf_all.append(len(surf_all))
                        tspc_sub.append(tspc_clu)
                tspc = numpy.array(tspc_sub).mean()
                N_surf_all=numpy.array(N_surf_all).mean()
                return tspc,N_surf_all,numpy.array(tspc_sub)
            tapc_train,tapc_all = tapc_calculation1(num_clusters, predicted, x_fs)
            tspc_train,_,tspc_all = tspc_calculation1(num_clusters, predicted, x_surf)
            # print(x_arrays.shape)
            # print(predicted.shape)
            DB_score,DB_all,dis_intra,dis_inter = DB_index3(x_arrays, predicted)
            n_detected = 0
            flag_detected=numpy.zeros(num_clusters)
            for n in range(num_clusters):
                n_fiber = numpy.sum(predicted == n)
                if n_fiber >= 20:
                    n_detected += 1
                    flag_detected[n]=1
            wmpg_train = n_detected / num_clusters
            return DB_score,wmpg_train,tapc_train,tspc_train
        # DB_score, wmpg, tapc,tapc1=metrics_calculation(x_array,preds,ds_fs_onehot)
        # DB_score_fs, wmpg_fs, tapc_fs, tapc1_fs= metrics_calculation(x_array, preds_fs, ds_fs_onehot,fs=True)
        # DB_all.append(DB_score)
        # wmpg_all.append(wmpg)
        # tapc_all.append(tapc)
        # tapc1_all.append(tapc1)

        if args.ro:
            num_stds = [0.75]
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
                            index1 = numpy.where((probc.mean() - probc) > num_std * probc.std())[0]
                            if len(index1) > 0:
                                id_rejectc = index[index1]
                                id_reject.extend(id_rejectc)
                                prob_rejected = probc[index1]
                                probs_rejected.append(prob_rejected)
                    id_reject = numpy.array(id_reject)
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
                    DB_score_ro, wmpg_ro, tapc_ro, tspc_ro = metrics_calculation(preds_fs_surf_ro, x_array_ro,
                                                                                  ds_fs_onehot_ro, ds_surf_onehot_ro)
                    rate_left = 1 - len(preds_fs_surf_ro) / len(preds_fs_surf)
                    print('fiber_removed:', rate_left)
                    # DB_score_ro_thr[i] = DB_score_ro
                    # wmpg_ro_thr[i] = wmpg_ro
                    # tapc_ro_thr[i] = tapc_ro
                    # tspc_ro_thr[i] = tspc_ro
                    # rate_left_thr[i] = rate_left
                    # DB_all_ro.append(DB_score_ro_thr)
                    # wmpg_all_ro.append(wmpg_ro_thr)
                    # tapc_all_ro.append(tapc_ro_thr)
                    # tspc_all_ro.append(tspc_ro_thr)
                    # rate_left_all.append(rate_left_thr)
                    print(DB_score_ro,wmpg_ro,tapc_ro,tspc_ro)
        #time_subject = time.time() - start
        #print('time_subject', time_subject)
        if args.save:
            outpath = os.path.join(args.outputDirectory, subject_id)
            pd_c_list = wma.cluster.mask_all_clusters(input_pd, preds_fs_surf_ro, num_clusters, preserve_point_data=True,
                                                                         preserve_cell_data=True,verbose=False)
            cluster_save(pd_c_list, outpath, input_pd, preds,num_clusters, cluster_colors)
            print('outdir', outpath)
    #
    # # print('db_test:', DB_all)
    # # print('wmpg_test:', wmpg_all)
    # # print('tapc_test:', tapc_all)
    # # print('fiber_left_rate:',rate_left_all)
    # if args.ro:
    #     numpy.save(args.inputDirectory+'/DB_ro.npy',numpy.array(DB_all_ro))
    #     numpy.save(args.inputDirectory+'/wmpg_ro.npy', numpy.array(wmpg_all_ro))
    #     numpy.save(args.inputDirectory+'/tapc_ro.npy', numpy.array(tapc_all_ro))
    #     numpy.save(args.inputDirectory + '/tapc1_ro.npy', numpy.array(tapc1_all_ro))
    # print(numpy.array(DB_all).mean(),numpy.array(wmpg_all).mean(),numpy.array(tapc_all).mean(),numpy.array(tapc1_all).mean())
    # numpy.save(args.inputDirectory+'/dgcnn/DB.npy',numpy.array(DB_all))
    # numpy.save(args.inputDirectory+'/dgcnn/wmpg.npy', numpy.array(wmpg_all))
    # numpy.save(args.inputDirectory+'/dgcnn/tapc.npy', numpy.array(tapc_all))
    # numpy.save(args.inputDirectory + '/dgcnn/tapc1.npy', numpy.array(tapc1_all))
    # numpy.save(args.inputDirectory+'/dgcnn/DB_fs.npy',numpy.array(DB_all_fs))
    # numpy.save(args.inputDirectory+'/dgcnn/wmpg_fs.npy', numpy.array(wmpg_all_fs))
    # numpy.save(args.inputDirectory+'/dgcnn/tapc_fs.npy', numpy.array(tapc_all_fs))
    # numpy.save(args.inputDirectory + '/dgcnn/tapc1_fs.npy', numpy.array(tapc1_all_fs))




