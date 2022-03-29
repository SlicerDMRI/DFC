from __future__ import print_function, division
import numpy
import whitematteranalysis as wma
import tract_feat
import vtk
from sklearn import metrics
import argparse
import torch
from torchvision import datasets, models, transforms
import os
import nets
import mnist
from torch.utils.tensorboard import SummaryWriter
from training_functions_fiber_pair import  calculate_predictions_test,calculate_predictions_roi
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
def convert_fiber_to_array(inputFile, numberOfFibers, fiberLength, numberOfFiberPoints, preproces=True):
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
    fiber_array.convert_from_polydata(input_data, points_per_fiber=args.numberOfFiberPoints)
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
def DB_index(x_array,predicted):
    #Sprint('mask')
    cluster_id=numpy.unique(predicted)
    fiber_array = numpy.reshape(x_array, (len(x_array), -1, 3))
    alpha = []
    c = []
    for i in list(cluster_id):
        d_cluster=fiber_array[predicted==i]
        assert not len(d_cluster)==0
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
        if len(d_cluster) == 1:
            distance_clu = 0
        else:
            distance_clu = numpy.sum(distance_array) / (len(d_cluster) * (len(d_cluster) - 1))
        alpha.append(distance_clu)
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
        dis_inter_clu = numpy.min(fiber_distance.fiber_distance(c[i], c1))
        DB_all.append(DB_clu)
        dis_inter.append(dis_inter_clu)
    DB=numpy.mean(DB_all)
    dis_intra = numpy.mean(alpha)
    dis_inter = numpy.mean(dis_inter)
    return  DB,dis_intra,dis_inter
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


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument(
        '-indir', action="store", dest="inputDirectory",
        default="/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/CNP1",
        help='A file of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument(
        '-outdir', action="store", dest="outputDirectory", default="./tractography_yc/PPMI/metrics",
        help='Output folder of clustering results.')
    parser.add_argument(
        '-modeldir', action="store", dest="modelDirectory", default="./nets/CAE_3_pair_001.pt",
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
    parser.add_argument('--save', default=True, type=str2bool, help='remove outliers')
    parser.add_argument('--net_architecture', default='CAE_3_pair',
                        choices=['CAE_3', 'CAE_bn3', 'CAE_4', 'CAE_bn4', 'CAE_5', 'CAE_bn5'],
                        help='network architecture used')
    parser.add_argument('--batch_size', default=2048, type=int, help='batch size')
    parser.add_argument('--num_clusters', default=800, type=int, help='number of clusters')
    parser.add_argument('--embedding_dimension', default=10, type=int, help='number of embeddings')
    parser.add_argument('--leaky', default=True, type=str2bool)
    parser.add_argument('--neg_slope', default=0.01, type=float)
    args = parser.parse_args()
    batch = args.batch_size
    model_name = args.net_architecture
    num_clusters=args.num_clusters
    workers = 4
    data_dir = args.inputDirectory
    modelDirectory_fs=args.modelDirectory[:-3]+'_fs_1.pt'
    modelDirectory = args.modelDirectory[:-3]+'_1.pt'
    print("\nData preparation\nReading data from:\t./" + data_dir)
    img_size = [28,28,3]
    #device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    to_eval = "nets." + model_name + "(img_size, num_clusters=num_clusters,embedding_dimension=args.embedding_dimension, leaky = args.leaky, neg_slope = args.neg_slope)"
    model = eval(to_eval)
    model.load_state_dict(torch.load(modelDirectory))
    model.eval()
    model = model.to(device)
    model_fs = eval(to_eval)
    model_fs.load_state_dict(torch.load(modelDirectory_fs))
    model_fs.eval()
    model_fs = model_fs.to(device)
    params = {'device': device}
    loss_fn=DiceScore()
    #data_dir='../dataFolder/deepClustering/cluster_dcec/test/102614_biClustered'
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
    tapc1_all_ro = []
    fiber_number_cluster_all=[]
    rate_left_all=[]
    cluster_colors = numpy.random.randint(0, 255, (num_clusters, 3))
    input_pd_fnames = wma.io.list_vtk_files(data_dir)
    num_pd = len(input_pd_fnames)
    for i in range(num_pd):
        subject_id = os.path.split(input_pd_fnames[i])[1][:6]
        input_pd,x_array,ds_fs,fiber_surf_ve,fiber_surf_dk,fiber_surf_des = convert_fiber_to_array(input_pd_fnames[i], numberOfFibers=args.numberOfFibers_test,
                                                        fiberLength=40,
                                                        numberOfFiberPoints=10000, preproces=True)
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
            return  ds_fs_onehot
        ds_fs_onehot=one_hot_encoding(ds_fs)

        #numpy.savez(os.path.join('/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/test_save', subject_id + '_data.npz'), x_array, ds_fs_onehot,fiber_surf_ve,fiber_surf_dk,fiber_surf_des)

    # test_data = os.listdir(args.inputDirectory)
    # for id,test_subject in enumerate(test_data):
    #     subject_id=test_subject.split('_')[0]
    #     data = numpy.load(os.path.join(args.inputDirectory, test_subject))
    #     #data=numpy.load(args.inputDirectory+'/{}_data.npz'.format(subject_id), allow_pickle=True)
    #     x_array=data['arr_0']
    #     ds_fs_onehot=data['arr_1']
    #     # fiber_surf_ve = data['arr_2']
    #     # fiber_surf_dk = data['arr_3']
    #     # fiber_surf_des= data['arr_4']
    #     ds_surf_onehot_dk = data['arr_2']

        # def surf_encoding(fiber_surf_ve,fiber_surf_dk,fiber_surf_des):
        #     fiber_surfs=fiber_surf_ve.astype(int)
        #     surf_labels = numpy.unique(fiber_surfs)
        #     surf_map = numpy.load('ve_map.npy')
        #     for surf_label in surf_labels:
        #         fiber_surfs[numpy.where(fiber_surfs == surf_label)] = numpy.where(surf_map == surf_label)
        #     ds_surf_onehot_ve = numpy.zeros((len(fiber_surfs), len(surf_map)))
        #     for s in range(len(fiber_surfs)):
        #         ds_surf_onehot_ve[s, fiber_surfs[s]] = 1
        #
        #     fiber_surfs = fiber_surf_dk.astype(int)
        #     surf_labels = numpy.unique(fiber_surfs)
        #     surf_map = numpy.load('dk_map.npy')
        #     for surf_label in surf_labels:
        #         fiber_surfs[numpy.where(fiber_surfs == surf_label)] = numpy.where(surf_map == surf_label)
        #     ds_surf_onehot_dk = numpy.zeros((len(fiber_surfs), len(surf_map)))
        #     for s in range(len(fiber_surfs)):
        #         ds_surf_onehot_dk[s, fiber_surfs[s]] = 1
        #
        #     fiber_surfs = fiber_surf_des.astype(int)
        #     surf_labels = numpy.unique(fiber_surfs)
        #     surf_map = numpy.load('des_map.npy')
        #     for surf_label in surf_labels:
        #         fiber_surfs[numpy.where(fiber_surfs == surf_label)] = numpy.where(surf_map == surf_label)
        #     ds_surf_onehot_des = numpy.zeros((len(fiber_surfs), len(surf_map)))
        #     for s in range(len(fiber_surfs)):
        #         ds_surf_onehot_des[s, fiber_surfs[s]] = 1
        #     return ds_surf_onehot_ve,ds_surf_onehot_dk,ds_surf_onehot_des
        #
        # ds_surf_onehot_ve,ds_surf_onehot_dk,ds_surf_onehot_des = surf_encoding(fiber_surf_ve,fiber_surf_dk,fiber_surf_des)

        #folder = 'test/'+os.path.splitext(os.path.basename(input_pd_fnames[i]))[0][:6] + '_biClustered'
        dataset = mnist.Fiber_pair(x_array,ds_fs_onehot, transform=transforms.Compose([transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)
        dataset_size = len(dataset)
        print("Training set size:\t" + str(dataset_size))
        since = time.time()
        preds, _ = calculate_predictions_test(model, dataloader, params)
        preds_fs, probs_fs = calculate_predictions_roi(model_fs, dataloader, params)

        elapsed=time.time()-since
        print('prediction time:',elapsed)

        # def metrics_calculation(x_array,preds,ds_fs_onehot,ds_surf_onehot_ve,ds_surf_onehot_dk,ds_surf_onehot_des,ds_fs):
        #     DB_score,dis_intra,dis_inter = DB_index(x_array, preds)
        #     n_detected = 0
        #     for n in range(num_clusters):
        #         n_fiber = numpy.sum(preds == n)
        #         if n_fiber >= 10:
        #             n_detected += 1
        #     wmpg10 = n_detected / num_clusters
        #     n_detected = 0
        #     for n in range(num_clusters):
        #         n_fiber = numpy.sum(preds == n)
        #         if n_fiber >= 20:
        #             n_detected += 1
        #     wmpg20 = n_detected / num_clusters
        #
        #     def tapc_calculation(preds, roi_fs,fs=False):
        #         if fs:
        #             roi_cluster = numpy.load('roi_cluster_fs_1.npy')
        #         else:
        #             roi_cluster = numpy.load('roi_cluster_1.npy')
        #         roi_preds = roi_cluster[preds]
        #         tapc = loss_fn(roi_fs, roi_preds)
        #         return tapc
        #     def tapc_calculation1(num_clusters, preds, roi_fs):
        #         roi_cluster = numpy.zeros([num_clusters, roi_fs.shape[1]])
        #         N_roi_all=[]
        #         N_roi_thr1=[]
        #         for i in range(num_clusters + 1):
        #             t = roi_fs[preds == i]
        #             if t.size == 0:
        #                 continue
        #             else:
        #                 t1 = numpy.sum(t, 0)
        #                 roi_all = numpy.where(t1 > t.shape[0] * 0.4)[0]
        #                 if 0 in roi_all:
        #                     roi_all = roi_all[1:]
        #                 roi_cluster[i, roi_all] = 1
        #
        #                 roi_all = numpy.where(t1 > 0)[0]
        #                 N_roi_all.append(len(roi_all))
        #                 roi_all1 = numpy.where(t1 > 0.1 * t.shape[0])[0]
        #                 N_roi_thr1.append(len(roi_all1))
        #         roi_preds = roi_cluster[preds]
        #         tapc = loss_fn(roi_fs, roi_preds)
        #         N_roi=numpy.mean(numpy.array(N_roi_all))
        #         N_roi_thr1 = numpy.mean(numpy.array(N_roi_thr1))
        #         return tapc, N_roi, N_roi_thr1
        #     def tspc_calculation(num_clusters,preds,ds_surf_onehot):
        #         tspc_sub=[]
        #         N_surf_all=[]
        #         for i in range(num_clusters + 1):
        #             t = ds_surf_onehot[preds == i]
        #             if t.size == 0:
        #                 continue
        #             else:
        #                 t1 = numpy.sum(t, 0)
        #                 surf_cluster = t1 / t1.sum()
        #                 tspc_all = surf_cluster * t
        #                 tspc1 = numpy.sum(tspc_all, 1)
        #
        #                 surf_all = numpy.where(t1 > 0)[0]
        #                 N_surf_all.append(len(surf_all))
        #                 tspc_sub.extend(list(tspc1))
        #         tspc = numpy.array(tspc_sub).mean()
        #         N_surf_all=numpy.array(N_surf_all).mean()
        #         return tspc,N_surf_all
        #
        #     def anatomical_metrics(num_clusters,ds_fs):
        #         us=[]
        #         entropys=[]
        #         for i in range(num_clusters + 1):
        #             indexes=list(numpy.where(preds == i)[0])
        #             roi_list = []
        #             for index in indexes:
        #                 roi = ds_fs[index]
        #                 roi_list.extend(list(roi))
        #             roi_array = numpy.array(roi_list, dtype=int)
        #             nums = numpy.bincount(roi_array)
        #             # the coefficient of unalikeability
        #             u = 1 - numpy.sum(numpy.square(nums / len(roi_array)))
        #             us.append(u)
        #             # information entropy
        #             p = nums / len(roi_array) + 1e-20
        #             entropy = numpy.sum(numpy.log2(p) * p) * (-1)
        #             entropys.append(entropy)
        #         entropy=numpy.array(entropys).mean()
        #         us=numpy.array(us).mean()
        #         return entropy,us
        #
        #     tapc1,N_roi, N_roi_thr1 = tapc_calculation1(num_clusters, preds, ds_fs_onehot)
        #     entropy, us = anatomical_metrics(num_clusters, ds_fs)
        #     tspc_ve,N_surf_ve = tspc_calculation(num_clusters, preds, ds_surf_onehot_ve)
        #     tspc_dk,N_surf_dk = tspc_calculation(num_clusters, preds, ds_surf_onehot_dk)
        #     tspc_des,N_surf_des = tspc_calculation(num_clusters, preds, ds_surf_onehot_des)
        #     return DB_score,dis_intra,dis_inter, wmpg10,wmpg20,tapc, tspc_ve,tspc_dk,tspc_des,entropy, us,N_roi, N_roi_thr1,N_surf_ve,N_surf_dk,N_surf_des
        def metrics_calculation(predicted,x_arrays,x_fs,x_surf):
            loss_fn = DiceScore()
            # def tapc_calculation1(num_clusters, preds, roi_fs):
            #     roi_cluster = numpy.zeros([num_clusters, roi_fs.shape[1]])
            #     for i in range(num_clusters + 1):
            #         t = roi_fs[preds == i]
            #         if t.size == 0:
            #             continue
            #         else:
            #             t1 = numpy.sum(t, 0)
            #             roi_all = numpy.where(t1 > t.shape[0] * 0.4)[0]
            #             if 0 in roi_all:
            #                 roi_all = roi_all[1:]
            #             roi_cluster[i, roi_all] = 1
            #     roi_preds = roi_cluster[preds]
            #     tapc = loss_fn(roi_fs, roi_preds)
            #     return tapc
            # def tspc_calculation1(num_clusters,preds,ds_surf_onehot):
            #     tspc_sub=[]
            #     N_surf_all=[]
            #     for i in range(num_clusters + 1):
            #         t = ds_surf_onehot[preds == i]
            #         if t.size == 0:
            #             continue
            #         else:
            #             t1 = numpy.sum(t, 0)
            #             surf_cluster = t1 / t1.sum()
            #             tspc_all = surf_cluster * t
            #             tspc1 = numpy.sum(tspc_all, 1)
            #
            #             surf_all = numpy.where(t1 > 0)[0]
            #             N_surf_all.append(len(surf_all))
            #             tspc_sub.extend(list(tspc1))
            #     tspc = numpy.array(tspc_sub).mean()
            #     N_surf_all=numpy.array(N_surf_all).mean()
            #     return tspc,N_surf_all
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
            # print('dis_intra:',dis_intra.mean())
            # print('dis_inter:', dis_inter.mean())
            #numpy.savez('debug_ro/results.nzp',DB_all,tapc_all,tspc_all,dis_intra,dis_inter)
            # DB1=DB_score
            # print(len(DB_all),len(flag_detected))
            # DB_score=numpy.mean(DB_all[numpy.where(flag_detected == 1)])
            # print(DB1,DB_score)
            #print(DB_score, wmpg_train, tapc_train,tspc_train)
            #print(DB_score, wmpg_train, tapc_train,tspc_train)
            return DB_score,wmpg_train,tapc_train,tspc_train
        # DB_score, wmpg, tapc, tspc_dk = metrics_calculation(preds, x_array, ds_fs_onehot, ds_surf_onehot_dk)
        # DB_score_fs, wmpg_fs, tapc_fs, tspc_dk_fs = metrics_calculation(preds_fs, x_array, ds_fs_onehot, ds_surf_onehot_dk)
        # print(DB_score,wmpg,tapc, tspc_dk)
        # print(DB_score_fs,wmpg_fs,tapc_fs, tspc_dk_fs)

        # outdir = args.inputDirectory + '/metrics'
        # if not os.path.exists(outdir):
        #     os.makedirs(outdir)
        # print(outdir)
        # numpy.savez(os.path.join(outdir, subject_id + '_measure.npz'), DB_score, wmpg,tapc,tspc_dk)
        # outdir = args.inputDirectory + '/metrics_fs'
        # if not os.path.exists(outdir):
        #     os.makedirs(outdir)
        # print(outdir)
        # numpy.savez(os.path.join(outdir, subject_id + '_measure.npz'), DB_score_fs,wmpg_fs,tapc_fs,tspc_dk_fs)

        if args.ro:
            id_reject = numpy.where(probs_fs < 0.0180)[0]
            clusters_outlier = preds_fs[id_reject]
            prob_outlier = probs_fs[id_reject]
            #numpy.save('clusters_outlier.npy', clusters_outlier)
            #print(id_reject)
            if id_reject is not None:
                temp = numpy.ones(len(preds_fs))
                temp[id_reject] = 0
                mask = temp > 0
                preds_fs_ro = preds_fs[mask]
                probs_fs_ro = probs_fs[mask]
                ds_fs_onehot = ds_fs_onehot[mask]
                #ds_surf_onehot_ve=ds_surf_onehot_ve[mask]
                #ds_surf_onehot_dk=ds_surf_onehot_dk[mask]
                #ds_surf_onehot_des=ds_surf_onehot_des[mask]
                x_array = x_array[mask]
                index_left=list(numpy.where(mask==1)[0])
                # ds_fs_left=[ds_fs[k] for k in index_left ]
                # ds_fs=ds_fs_left
                input_pd = wma.filter.mask(input_pd, mask, preserve_point_data=True, preserve_cell_data=True,verbose=False)
                #rate_left1 = (len(preds_fs_ro) - len(id_reject)) / len(preds_fs_ro)
                rate_left=len(id_reject) / len(preds_fs)
                rate_left_all.append(rate_left)
                rate_left_sub=numpy.mean(rate_left_all)
                print('fiber_remove_rate:', rate_left)
                # DB_score_fs_ro, wmpg_fs_ro, tapc_fs_ro, tspc_dk_fs_ro = metrics_calculation(preds_fs_ro, x_array, ds_fs_onehot,ds_surf_onehot_dk)
                # print(DB_score_fs_ro, wmpg_fs_ro,tapc_fs_ro, tspc_dk_fs_ro, rate_left_sub)
                outdir = args.inputDirectory + '/metrics_fs_ro'
                print(outdir)
                # if not os.path.exists(outdir):
                #     os.makedirs(outdir)
                # numpy.savez(os.path.join(outdir, subject_id + '_measure.npz'), DB_score_fs_ro, wmpg_fs_ro, tapc_fs_ro,tspc_dk_fs_ro, rate_left_sub)

        # since =time.time()
        if args.save:
            #outpath = os.path.join(args.outputDirectory, folder)
            outpath = args.outputDirectory
            pd_c_list = wma.cluster.mask_all_clusters(input_pd, preds_fs_ro, num_clusters, preserve_point_data=True,
                                                                         preserve_cell_data=True,verbose=False)
            cluster_save(pd_c_list, outpath, input_pd, preds,num_clusters, cluster_colors)
            print('outdir', outpath)

    time_subject = time.time() - start
    print('time_subject', time_subject)



