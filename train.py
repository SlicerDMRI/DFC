from __future__ import print_function, division
import numpy
import whitematteranalysis as wma
import training_functions_fiber_pair
#import tract_feat
import vtk
from  test import DB_index,DB_index3
from test import DiceScore
from sklearn import metrics
import fibers
import glob
import time
from training_functions_fiber_pair import  calculate_predictions_test

def list_files(input_dir,str):
    # Find input files
    input_mask = ("{0}/"+str+"*").format(input_dir)
    input_pd_fnames = glob.glob(input_mask)
    input_pd_fnames = sorted(input_pd_fnames)
    return(input_pd_fnames)
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
def read_data(data_dir):
    inputDir_train = data_dir
    input_pd_fnames = wma.io.list_vtk_files(inputDir_train)
    num_pd = len(input_pd_fnames)
    input_pds = []
    x_arrays=[]
    d_rois=[]
    fiber_surfs_ve=[]
    fiber_surfs_dk = []
    fiber_surfs_des = []
    for i in range(num_pd):
        input_pd,x_array,d_roi,fiber_surf_ve,fiber_surf_dk,fiber_surf_des = \
            convert_fiber_to_array(input_pd_fnames[i], numberOfFibers=args.numberOfFibers_train,fiberLength=args.fiberLength,
                                 numberOfFiberPoints=args.numberOfFiberPoints, preproces=False)
        # def one_hot_encoding(ds_fs):
        #     roi_map = numpy.load('relabel_map.npy')
        #     ds_fs_onehot = numpy.zeros((len(ds_fs), len(numpy.unique(roi_map[1])))).astype(numpy.float32)
        #     if not isinstance(ds_fs, list):
        #         roi_unique = numpy.unique(ds_fs)
        #         assert set(roi_unique).issubset(set(roi_map[0]))
        #         for roi in roi_unique:
        #             roi_new = roi_map[1][roi_map[0] == roi]
        #             ds_fs[ds_fs == roi] = roi_new
        #         for f in range(ds_fs.shape[0]):
        #             roi_single = numpy.unique(ds_fs[f])
        #             if roi_single[0] == 0:
        #                 roi_single = roi_single[1:]
        #             ds_fs_onehot[f, roi_single.astype(int)] = 1
        #     else:
        #         for f, roi_fiber in enumerate(ds_fs):
        #             roi_unique = numpy.unique(roi_fiber)
        #             assert set(roi_unique).issubset(set(roi_map[0]))
        #             for roi in roi_unique:
        #                 roi_new = roi_map[1][roi_map[0] == roi]
        #                 roi_fiber[roi_fiber == roi] = roi_new
        #             roi_single = numpy.unique(roi_fiber)
        #             if roi_single[0] == 0:
        #                 roi_single = roi_single[1:]
        #             ds_fs_onehot[f, roi_single.astype(int)] = 1
        #     return  ds_fs_onehot
        # ds_fs_onehot=one_hot_encoding(d_roi)
        # subject_id=(os.path.basename(input_pd_fnames[i])).split("_")[0]
        #
        # numpy.save('/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/test_save/x_array_{}.npy'.format(subject_id),
        #            x_array)
        # numpy.save('/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/test_save/x_roi_{}.npy'.format(subject_id),
        #            ds_fs_onehot)
        fiber_surfs_ve.append(fiber_surf_ve)
        fiber_surfs_dk.append(fiber_surf_dk)
        fiber_surfs_des.append(fiber_surf_des)

        input_pds.append(input_pd)
        x_arrays.append(x_array)
        if isinstance(d_roi,list):
            d_rois.extend(d_roi)
        else:
            d_rois.append(d_roi)

    #d=numpy.array(d_rois).reshape((len(d_roi) * num_pd,d_roi.shape[1],d_roi.shape[2],d_roi.shape[3]))
    # roi_uniques=numpy.array([])
    # for roi in d_rois:
    #     roi_unique=numpy.unique(roi)
    #     #print(len(roi_unique))
    #     roi_uniques=numpy.concatenate((roi_uniques,roi_unique))
    #     roi_uniques=numpy.unique(roi_uniques)


    # roi_origianl=[41,43,44,46,47,49,50,51,52,53,54,58,60,62,63,1100,2000,2100,3000,3100,4000,4100,5002]
    # roi_origianl.extend(list(range(2001,2036)))
    # roi_origianl.extend(list(range(3001, 3036)))
    # roi_origianl.extend(list(range(4001, 4036)))
    # roi_replace=[2,4,5,7,8,10,11,12,13,17,18,26,28,30,31,1000,1000,1000,1000,1000,1000,1000,5001]
    # roi_replace.extend(list(range(1001,1036)))
    # roi_replace.extend(list(range(1001,1036)))
    # roi_replace.extend(list(range(1001, 1036)))
    # #roi_check=numpy.stack((numpy.array(roi_origianl),numpy.array(roi_replace)),0)
    # import copy
    # roi_unique=numpy.unique(ds_fs)
    # roi_merge=copy.deepcopy(roi_unique)
    # for i,roi1 in enumerate(roi_origianl):
    #     roi_merge[roi_unique==roi1]=roi_replace[i]
    # roi_uni_merge = numpy.unique(roi_merge)
    # roi_relabel=numpy.zeros(len(roi_unique))
    # for i,roi in enumerate(roi_uni_merge):
    #     roi_relabel[roi_merge==roi]=i
    # # for ido, idr in zip(list(roi_origianl),list(roi_replace)):
    # #     ds_fs[ds_fs == ido] = idr
    # # roi_id = numpy.unique(ds_fs)
    # ds_fs1 = copy.deepcopy(ds_fs)
    # for i, id in zip(list(roi_relabel),list(roi_unique)):
    #     ds_fs[ds_fs == id] = i
    # roi_map = numpy.stack((roi_unique,roi_relabel),0)
    # numpy.save('relabel_map.npy', roi_map)
    x_arrays = numpy.array(x_arrays).reshape((-1, x_array.shape[1], x_array.shape[2]))
    roi_map = numpy.load('relabel_map.npy')
    # roi_map=numpy.concatenate((roi_map1,numpy.array([[80],[64]])),1)
    # numpy.save('relabel_map.npy',roi_map)
    ds_fs_onehot = numpy.zeros((len(x_arrays), len(numpy.unique(roi_map[1])))).astype(numpy.float32)
    if len(d_rois)==num_pd:
        ds_fs = numpy.array(d_rois).reshape((len(x_array) * num_pd, -1))
        roi_unique = numpy.unique(ds_fs)
        assert set(roi_unique).issubset(set(roi_map[0]))
        for roi in roi_unique:
            roi_new = roi_map[1][roi_map[0] == roi]
            ds_fs[ds_fs == roi] = roi_new
        for f in range(ds_fs.shape[0]):
            roi_single = numpy.unique(ds_fs[f])
            if roi_single[0] == 0:
                roi_single=roi_single[1:]
            ds_fs_onehot[f, roi_single.astype(int)] = 1
    elif len(d_rois)==x_arrays.shape[0]:
        for f,roi_fiber in enumerate(d_rois):
            roi_unique = numpy.unique(roi_fiber)
            assert set(roi_unique).issubset(set(roi_map[0]))
            for roi in roi_unique:
                roi_new = roi_map[1][roi_map[0] == roi]
                roi_fiber[roi_fiber == roi] = roi_new
            roi_single = numpy.unique(roi_fiber)
            if roi_single[0] == 0:
                roi_single=roi_single[1:]
            ds_fs_onehot[f, roi_single.astype(int)] = 1

    fiber_surfs_ve = numpy.array(fiber_surfs_ve).reshape((-1, 2))
    fiber_surfs_dk = numpy.array(fiber_surfs_dk).reshape((-1, 2))
    fiber_surfs_des = numpy.array(fiber_surfs_des).reshape((-1, 2))

    def surf_encoding(fiber_surf_ve, fiber_surf_dk, fiber_surf_des):
        fiber_surfs = fiber_surf_ve.astype(int)
        surf_labels = numpy.unique(fiber_surfs)
        surf_map = numpy.load('ve_map.npy')
        for surf_label in surf_labels:
            fiber_surfs[numpy.where(fiber_surfs == surf_label)] = numpy.where(surf_map == surf_label)
        ds_surf_onehot_ve = numpy.zeros((len(fiber_surfs), len(surf_map)))
        for s in range(len(fiber_surfs)):
            ds_surf_onehot_ve[s, fiber_surfs[s]] = 1

        fiber_surfs = fiber_surf_dk.astype(int)
        surf_labels = numpy.unique(fiber_surfs)
        surf_map = numpy.load('dk_map.npy')
        for surf_label in surf_labels:
            fiber_surfs[numpy.where(fiber_surfs == surf_label)] = numpy.where(surf_map == surf_label)
        ds_surf_onehot_dk = numpy.zeros((len(fiber_surfs), len(surf_map)))
        for s in range(len(fiber_surfs)):
            ds_surf_onehot_dk[s, fiber_surfs[s]] = 1

        fiber_surfs = fiber_surf_des.astype(int)
        surf_labels = numpy.unique(fiber_surfs)
        surf_map = numpy.load('des_map.npy')
        for surf_label in surf_labels:
            fiber_surfs[numpy.where(fiber_surfs == surf_label)] = numpy.where(surf_map == surf_label)
        ds_surf_onehot_des = numpy.zeros((len(fiber_surfs), len(surf_map)))
        for s in range(len(fiber_surfs)):
            ds_surf_onehot_des[s, fiber_surfs[s]] = 1
        return ds_surf_onehot_ve, ds_surf_onehot_dk, ds_surf_onehot_des

    ds_surf_onehot_ve, ds_surf_onehot_dk, ds_surf_onehot_des = surf_encoding(fiber_surfs_ve, fiber_surfs_dk,
                                                                             fiber_surfs_des)
    # from sklearn import preprocessing
    # ds_fs = preprocessing.MaxAbsScaler().fit_transform(ds_fs)*100
    #ds_fs=ds_fs.reshape(ds_fs.shape[0],d_roi.shape[1],d_roi.shape[2],d_roi.shape[3])
    # ds_train=ds_train.transpose((0,3,1,2))
    # ds_trainf = ds_trainf.transpose((0, 3, 1, 2))
    #ds_fs = ds_fs.transpose((0, 3, 1, 2))
    #ds_train1 = numpy.mean((ds_train, ds_trainf), axis=0)
    #ds_train1=numpy.concatenate((ds_train,ds_fs),axis=1)
    return  input_pds,x_arrays,ds_fs_onehot,ds_surf_onehot_ve, ds_surf_onehot_dk, ds_surf_onehot_des

if __name__ == "__main__":

    import argparse
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torchvision import datasets, models, transforms
    import os
    import math
    import fnmatch
    import nets
    import utils
    from training_functions_fiber_pair import calculate_predictions_roi
    from torch.utils.tensorboard import SummaryWriter

    # Translate string entries to bool for parser
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument(
        '-indir',action="store", dest="inputDirectory",default="/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/train_save",
        help='A file of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument(
        '-indirv',action="store", dest="inputDirectoryv",default="/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/validation",
        help='A file of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument(
        '-indirt',action="store", dest="inputDirectoryt",default="/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/test2",
        help='A file of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument(
        '-outdir',action="store", dest="outputDirectory",default="../torch_DFC/results",
        help='Output folder of clustering results.')
    # parser.add_argument(
    #     '-mp',action="store", dest="model_pretrained",default="nets/CAE_pair_130_pretrained.pt",
    #     help='Output folder of clustering results.')
    parser.add_argument(
        '-trf', action="store", dest="numberOfFibers_train", type=int, default=None,
        help='Number of fibers of each training data to analyze from each subject.')
    parser.add_argument(
        '-l', action="store", dest="fiberLength", type=int, default=40,
        help='Minimum length (in mm) of fibers to analyze. 60mm is default.')
    parser.add_argument(
        '-p', action="store", dest="numberOfFiberPoints", type=int, default=14,
        help='Number of points in each fiber to process. 10 is default.')
    parser.add_argument('--test', default=False, type=str2bool, help='whether perform experiment on testing data')
    parser.add_argument('--fs', default=True, type=str2bool, help='inporparating freesurfer information')
    parser.add_argument('--surf', default=True, type=str2bool, help='inporparating cortical information')
    parser.add_argument('--ro', default=True, type=str2bool, help='outlier removal')
    parser.add_argument('--num_clusters', default=800, type=int, help='number of clusters')
    parser.add_argument('--embedding_dimension', default=10, type=int, help='number of embeddings')
    parser.add_argument('--epochs', default=1, type=int, help='clustering epochs')
    parser.add_argument('--epochs_pretrain', default=300, type=int, help='pretraining epochs')

    parser.add_argument('--mode', default='train_full', choices=['train_full', 'pretrain'], help='mode')
    parser.add_argument('--tensorboard', default=True, type=bool, help='export training stats to tensorboard')
    parser.add_argument('--pretrain', default=False, type=str2bool, help='perform autoencoder pretraining')
    parser.add_argument('--idx', default=True, type=str2bool, help='idx for dgcnn')
    parser.add_argument('--k', default=5, type=int, help='k for dgcnn')
    parser.add_argument('--pretrained_net', default=1, help='index or path of pretrained net')
    parser.add_argument('--net_architecture', default='DGCNN', choices=['CAE_3', 'CAE_pair','CAE_bn3', 'CAE_4', 'CAE_bn4', 'CAE_5', 'DGCNN','PointNet','CAE_DG_pair','GCN'], help='network architecture used')
    parser.add_argument('--dataset', default='Fiber',
                        choices=['MNIST-train', 'custom', 'MNIST-test', 'MNIST-full','Fiber','FiberMap','FiberCom'],
                        help='custom or prepared dataset')
    parser.add_argument('--data', default='HCP',
                        choices=['HCP', 'PPMI', 'open_fMRI'],
                        help='custom or prepared dataset')
    #parser.add_argument('--dataset_path', default='data', help='path to dataset')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--rate', default=0.00001, type=float, help='learning rate for clustering')
    parser.add_argument('--rate_pretrain', default=0.0003, type=float, help='learning rate for pretraining')
    parser.add_argument('--weight', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--weight_pretrain', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--sched_step', default=200, type=int, help='scheduler steps for rate update')
    parser.add_argument('--sched_step_pretrain', default=200, type=int,
                        help='scheduler steps for rate update - pretrain')
    parser.add_argument('--sched_gamma', default=0.1, type=float, help='scheduler gamma for rate update')
    parser.add_argument('--sched_gamma_pretrain', default=0.1, type=float,
                        help='scheduler gamma for rate update - pretrain')
    parser.add_argument('--printing_frequency', default=10, type=int, help='training stats printing frequency')
    parser.add_argument('--gamma', default=0.1, type=float, help='clustering loss weight')
    parser.add_argument('--update_interval', default=100, type=int, help='update interval for target distribution')
    parser.add_argument('--tol', default=1e-2, type=float, help='stop criterium tolerance')
    parser.add_argument('--custom_img_size', default=[128, 128, 3], nargs=3, type=int, help='size of custom images')
    parser.add_argument('--leaky', default=True, type=str2bool)
    parser.add_argument('--neg_slope', default=0.01, type=float)
    parser.add_argument('--activations', default=False, type=str2bool)
    parser.add_argument('--bias', default=True, type=str2bool)
    args = parser.parse_args()
    print(args)

    if args.mode == 'pretrain' and not args.pretrain:
        print("Nothing to do :(")
        exit()

    board = args.tensorboard

    # Deal with pretraining option and way of showing network path
    pretrain = args.pretrain
    net_is_path = True
    if not pretrain:
        try:
            int(args.pretrained_net)
            idx = args.pretrained_net
            net_is_path = False
        except:
            pass
    params = {'pretrain': pretrain}
    #params['model_pretrained']=args.model_pretrained
    # Directories
    # Create directories structure
    dirs = ['runs', 'reports', 'nets']
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

    import re
    # Net architecture
    model_name = args.net_architecture
    # Indexing (for automated reports saving) - allows to run many trainings and get all the reports collected
    if pretrain or (not pretrain and net_is_path):
        reports_list = sorted(os.listdir('nets'), reverse=True)
        if reports_list:
            for file in reports_list:
                # print(file)
                if fnmatch.fnmatch(file, model_name + '*'):
                    #idx = int(str(file)[-7:-4]) + 1
                    idx=int("".join(re.findall(r'\d', file)))+1
                    break
        try:
            idx
        except NameError:
            idx = 1

    # Base filename
    name = model_name + '_' + str(idx).zfill(3)
    # if args.fs:
    #     name=name+'_fs'

    # Filenames for report and weights
    name_txt = name + '.txt'
    if args.pretrain:
        name_txt = name + '_pretrain.txt'
    name_net = name
    pretrained = name + '_pretrained.pt'

    # Arrange filenames for report, network weights, pretrained network weights
    if args.pretrain:
        name_txt = os.path.join('reports', name_txt)
    else:
        name_txt = os.path.join('reports', name + '_{}.txt'.format(args.epochs))
    print('report path:',  name_txt)

    name_net = os.path.join('nets', name_net)
    if net_is_path and not pretrain:
        pretrained = args.pretrained_net
    else:
        pretrained = os.path.join('nets', pretrained)
    if not pretrain and not os.path.isfile(pretrained):
        print("No pretrained weights, try again choosing pretrained network or create new with pretrain=True")

    model_files = [name_net, pretrained]
    print(model_files)
    params['model_files'] = model_files

    # Open file
    if pretrain:
        f = open(name_txt, 'w')
    else:
        f = open(name_txt, 'a')
    params['txt_file'] = f

    # Delete tensorboard entry if exist (not to overlap as the charts become unreadable)
    try:
        os.system("rm -rf runs/" + name)
    except:
        pass

    # Initialize tensorboard writer
    if board:
        writer = SummaryWriter('runs/' + name)
        if args.pretrain:
            writer = SummaryWriter('runs/' + name+'_pretrained')
        print('event path:', 'runs/' + name)
        params['writer'] = writer
    else:
        params['writer'] = None

    # Hyperparameters

    # Used dataset
    dataset = args.dataset

    # Batch size
    batch = args.batch_size
    params['batch'] = batch
    # Number of workers (typically 4*num_of_GPUs)
    workers = 4
    # Learning rate
    rate = args.rate
    rate_pretrain = args.rate_pretrain
    # Adam params
    # Weight decay
    weight = args.weight
    weight_pretrain = args.weight_pretrain
    # Scheduler steps for rate update
    sched_step = args.sched_step
    sched_step_pretrain = args.sched_step_pretrain
    # Scheduler gamma - multiplier for learning rate
    sched_gamma = args.sched_gamma
    sched_gamma_pretrain = args.sched_gamma_pretrain

    # Number of epochs
    epochs = args.epochs
    pretrain_epochs = args.epochs_pretrain
    params['pretrain_epochs'] = pretrain_epochs

    # Printing frequency
    print_freq = args.printing_frequency
    params['print_freq'] = print_freq

    # Clustering loss weight:
    gamma = args.gamma
    params['gamma'] = gamma

    # Update interval for target distribution:
    update_interval = args.update_interval
    params['update_interval'] = update_interval

    # Tolerance for label changes:
    tol = args.tol
    params['tol'] = tol

    # Number of clusters
    num_clusters = args.num_clusters


    # Report for settings
    tmp = "Training the '" + model_name + "' architecture"
    utils.print_both(f, tmp)
    tmp = "\n" + "The following parameters are used:"
    utils.print_both(f, tmp)
    tmp = "Batch size:\t" + str(batch)
    utils.print_both(f, tmp)
    tmp = "Number of workers:\t" + str(workers)
    utils.print_both(f, tmp)
    tmp = "Learning rate:\t" + str(rate)
    utils.print_both(f, tmp)
    tmp = "Pretraining learning rate:\t" + str(rate_pretrain)
    utils.print_both(f, tmp)
    tmp = "Weight decay:\t" + str(weight)
    utils.print_both(f, tmp)
    tmp = "Pretraining weight decay:\t" + str(weight_pretrain)
    utils.print_both(f, tmp)
    tmp = "Scheduler steps:\t" + str(sched_step)
    utils.print_both(f, tmp)
    tmp = "Scheduler gamma:\t" + str(sched_gamma)
    utils.print_both(f, tmp)
    tmp = "Pretraining scheduler steps:\t" + str(sched_step_pretrain)
    utils.print_both(f, tmp)
    tmp = "Pretraining scheduler gamma:\t" + str(sched_gamma_pretrain)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of training:\t" + str(epochs)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of pretraining:\t" + str(pretrain_epochs)
    utils.print_both(f, tmp)
    tmp = "Clustering loss weight:\t" + str(gamma)
    utils.print_both(f, tmp)
    tmp = "Update interval for target distribution:\t" + str(update_interval)
    utils.print_both(f, tmp)
    tmp = "Stop criterium tolerance:\t" + str(tol)
    utils.print_both(f, tmp)
    tmp = "Number of clusters:\t" + str(num_clusters)
    utils.print_both(f, tmp)
    tmp = "Leaky relu:\t" + str(args.leaky)
    utils.print_both(f, tmp)
    tmp = "Leaky slope:\t" + str(args.neg_slope)
    utils.print_both(f, tmp)
    tmp = "Activations:\t" + str(args.activations)
    utils.print_both(f, tmp)
    tmp = "Bias:\t" + str(args.bias)
    utils.print_both(f, tmp)

    # Data preparation
    if dataset == 'MNIST-train':
        # Uses slightly modified torchvision MNIST class
        import mnist
        tmp = "\nData preparation\nReading data from: MNIST train dataset"
        utils.print_both(f, tmp)
        img_size = [28, 28, 1]
        tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
        utils.print_both(f, tmp)

        dataset = mnist.MNIST('../data', train=True, download=True,
                              transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           # transforms.Normalize((0.1307,), (0.3081,))
                                                           ]))

        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch, shuffle=False, num_workers=workers)

        dataset_size = len(dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)

    elif dataset == 'MNIST-test':
        import mnist
        tmp = "\nData preparation\nReading data from: MNIST test dataset"
        utils.print_both(f, tmp)
        img_size = [28, 28, 1]
        tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
        utils.print_both(f, tmp)

        dataset = mnist.MNIST('../data', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  # transforms.Normalize((0.1307,), (0.3081,))
                              ]))

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch, shuffle=False, num_workers=workers)

        dataset_size = len(dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)

    elif dataset == 'MNIST-full':
        import mnist
        tmp = "\nData preparation\nReading data from: MNIST full dataset"
        utils.print_both(f, tmp)
        img_size = [28, 28, 1]
        tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
        utils.print_both(f, tmp)

        dataset = mnist.MNIST('../data', full=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ]))

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch, shuffle=False, num_workers=workers)

        dataset_size = len(dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)
    elif dataset == 'Fiber':
        data_dir = args.inputDirectory
        data_dirv = args.inputDirectoryv
        data_dirt=args.inputDirectoryt
        tmp = "\nData preparation\nReading data from:\t./" + data_dir
        utils.print_both(f, tmp)
        # x_arrays=numpy.load(os.path.join(args.inputDirectory,'x_array.npy'))
        # x_roi = numpy.load(os.path.join(args.inputDirectory, 'x_roi.npy'))
        # x_surf_dk = numpy.load(os.path.join(args.inputDirectory, 'x_surf_dk.npy'))
        input_pds,x_arrays,x_roi,x_surf_ve,x_surf_dk,x_surf_des=read_data(data_dir)
        # numpy.save('/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/train_save_15/x_array.npy',x_arrays)
        # numpy.save('/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/train_save_15/x_roi.npy',x_roi)
        # numpy.save('/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/train_save_15/x_surf_dk.npy',x_surf_dk)
        num_points = args.numberOfFiberPoints
        tmp = "numner of points used: {}".format(num_points)
        utils.print_both(f, tmp)
        import mnist
        dataset = mnist.Fiber_pair(x_arrays,x_roi,x_surf_dk,transform=transforms.Compose([transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch, shuffle=True, num_workers=workers)
        dataloader1 = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)
        if pretrain:
            input_pdsv, x_arraysv, x_roiv,x_surf_vev,x_surf_dkv,x_surf_desv = read_data(data_dirv)
            datasetv = mnist.Fiber_pair(x_arraysv, x_roiv,x_surf_dkv, transform=transforms.Compose([transforms.ToTensor()]))
            dataloaderv = torch.utils.data.DataLoader(datasetv, batch_size=batch, shuffle=False, num_workers=workers)
        else:
            dataloaderv=dataloader1
            x_arraysv=x_arrays
            x_roiv=x_roi
            x_surfv=x_surf_dk
        dataset_size = len(dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)
        #test_arrays=list_files(data_dirt,'x_array')
        #test_rois = list_files(data_dirt, 'x_roi')
        test_data = os.listdir(data_dirt)

    elif dataset == 'FiberMap':
        data_dir = args.inputDirectory
        data_dirv = args.inputDirectoryv
        data_dirt=args.inputDirectoryt
        tmp = "\nData preparation\nReading data from:\t./" + data_dir
        utils.print_both(f, tmp)
        x_arrays = numpy.load(os.path.join(args.inputDirectory, 'x_array.npy'))
        x_roi = numpy.load(os.path.join(args.inputDirectory, 'x_roi.npy'))
        x_surf_dk = numpy.load(os.path.join(args.inputDirectory, 'x_surf_dk.npy'))
        #input_pds,x_arrays,x_roi=read_data(data_dir)
        img_size = [28, 28, 3]
        tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
        utils.print_both(f, tmp)
        import mnist
        dataset = mnist.FiberMap_pair(x_arrays,x_roi,x_surf_dk,transform=transforms.Compose([transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch, shuffle=False, num_workers=workers)
        dataloader1 = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=workers)
        if pretrain:
            input_pdsv, x_arraysv, x_roiv,x_surf_vev,x_surf_dkv,x_surf_desv = read_data(data_dirv)
            datasetv = mnist.FiberMap_pair(x_arraysv, x_roiv, x_surf_dkv,transform=transforms.Compose([transforms.ToTensor()]))
            dataloaderv = torch.utils.data.DataLoader(datasetv, batch_size=batch, shuffle=False, num_workers=workers)
        else:
            dataloaderv=dataloader1
            x_arraysv=x_arrays
            x_roiv=x_roi
            x_surfv=x_surf_dk
        dataset_size = len(dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)
        # test_arrays=list_files(data_dirt,'x_array')
        # test_rois = list_files(data_dirt, 'x_roi')
        test_data = os.listdir(data_dirt)
    elif dataset == 'FiberCom':
        data_dir = args.inputDirectory
        data_dirv = args.inputDirectoryv
        data_dirt = args.inputDirectoryt
        tmp = "\nData preparation\nReading data from:\t./" + data_dir
        utils.print_both(f, tmp)
        x_arrays = numpy.load(os.path.join(args.inputDirectory, 'x_array.npy'))
        x_roi = numpy.load(os.path.join(args.inputDirectory, 'x_roi.npy'))
        # input_pds,x_arrays,x_roi=read_data(data_dir)
        img_size = [28, 28, 3]
        tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
        utils.print_both(f, tmp)
        import mnist

        dataset = mnist.FiberCom_pair(x_arrays, x_roi, transform=transforms.Compose([transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)
        dataloader1 = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)
        if pretrain:
            input_pdsv, x_arraysv, x_roiv = read_data(data_dirv)
            datasetv = mnist.FiberCom_pair(x_arraysv, x_roiv, transform=transforms.Compose([transforms.ToTensor()]))
            dataloaderv = torch.utils.data.DataLoader(datasetv, batch_size=batch, shuffle=False, num_workers=workers)
        else:
            dataloaderv = dataloader1
        dataset_size = len(dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)
        test_arrays = list_files(data_dirt, 'x_array')
        test_rois = list_files(data_dirt, 'x_roi')
    else:
        # Data folder
        data_dir = args.inputDirectory
        tmp = "\nData preparation\nReading data from:\t./" + data_dir
        utils.print_both(f, tmp)

        # Image size
        custom_size = math.nan
        custom_size = args.custom_img_size
        if isinstance(custom_size, list):
            img_size = custom_size

        tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
        utils.print_both(f, tmp)

        # Transformations
        data_transforms = transforms.Compose([
                transforms.Resize(img_size[0:2]),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Read data from selected folder and apply transformations
        image_dataset = datasets.ImageFolder(data_dir, data_transforms)
        # Prepare data for network: schuffle and arrange batches
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch,
                                                      shuffle=False, num_workers=workers)

        # Size of data sets
        dataset_size = len(image_dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)

    params['dataset_size'] = dataset_size

    # GPU check
    #device="cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmp = "\nPerforming calculations on:\t" + str(device)
    utils.print_both(f, tmp + '\n')
    params['device'] = device

    if args.net_architecture == 'DGCNN':
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


    # Evaluate the proper model
    if args.net_architecture=='CAE_pair':
        to_eval = "nets." + model_name + "(img_size, num_clusters=num_clusters,embedding_dimension=args.embedding_dimension, leaky = args.leaky, neg_slope = args.neg_slope)"
    elif args.net_architecture=='DGCNN' :
        to_eval = "nets." + model_name + "(k=args.k,input_channel=3,num_clusters=num_clusters,embedding_dimension=args.embedding_dimension,idx=idx)"
    elif args.net_architecture == 'PointNet' or args.net_architecture == 'GCN':
        to_eval = "nets." + model_name + "(input_channel=3,num_clusters=num_clusters,embedding_dimension=args.embedding_dimension)"
    elif args.net_architecture == 'CAE_DG_pair':
        to_eval = "nets." + model_name + "(img_size, num_clusters=num_clusters,embedding_dimension=args.embedding_dimension, leaky = args.leaky, " \
                                         "neg_slope = args.neg_slope,input_channel=3)"
    model = eval(to_eval)

    # Tensorboard model representation
    # if board:
    #     writer.add_graph(model, torch.autograd.Variable(torch.Tensor(batch, img_size[2], img_size[0], img_size[1])))

    model = model.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_pa=count_parameters(model)
    print('number of parameters:',num_pa)

    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #params = sum([numpy.prod(p.size()) for p in model_parameters])
    #print(params)

    # Reconstruction loss
    criterion_1 = nn.MSELoss(size_average=True)
    # Clustering loss
    criterion_2 = nn.KLDivLoss(size_average=False)

    criteria = [criterion_1, criterion_2]

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate, weight_decay=weight)

    optimizer_pretrain = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate_pretrain, weight_decay=weight_pretrain)

    optimizers = [optimizer, optimizer_pretrain]

    scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
    scheduler_pretrain = lr_scheduler.StepLR(optimizer_pretrain, step_size=sched_step_pretrain, gamma=sched_gamma_pretrain)
    #scheduler_pretrain = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_pretrain, 5,T_mult=2)

    schedulers = [scheduler, scheduler_pretrain]

    if args.mode == 'train_full':
        model_pretrained, model,preds_initial, preds_final,probs_final= training_functions_fiber_pair.train_model(model, dataloader, dataloaderv,dataloader1, #,preds_initial, preds
                                                                                   criteria, optimizers, schedulers, epochs, params,x_roi,args.fs,x_surf_dk,args.surf)
        list_all=[]
        #n_random=numpy.random.randint(0,num_clusters,100)
        for n in range(800):
            # print(n)
            probs_cluster=probs_final[preds_final==n]
            list_all.append(probs_cluster)
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(figsize=(50,6))
        flierprops=dict(marker='+',markersize=3)
        axes.boxplot(list_all,patch_artist=True,widths=0.8,flierprops=flierprops)
        plt.ylim([0, 0.6])
        #plt.savefig('probablities_clusters_train.png')

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
            utils.print_both(f,  'DB: {0:.4f}\tWMPG: {1:.4f}\tTAPC: {2:.4f}\tTSPC: {3:.4f}'.format(DB_score, wmpg_train, tapc_train,tspc_train))
            return DB_score,wmpg_train,tapc_train,tspc_train
        if args.fs:
            if args.surf:
                name_net_save = name_net + '_{}_fs_surf.pt'.format(epochs)
            else:
                name_net_save=name_net + '_{}_fs.pt'.format(epochs)
            torch.save(model.state_dict(), name_net_save)
            print(name_net_save)
        def roi_cluster_uptate(num_clusters, preds, x_fs):
            roi_cluster = numpy.zeros([num_clusters, x_fs.shape[1]])
            for i in range(num_clusters):
                t = x_fs[preds == i]
                t1 = numpy.sum(t, 0)
                roi_all = numpy.where(t1 > t.shape[0] * 0.4)[0]
                if 0 in roi_all:
                    roi_all = roi_all[1:]
                roi_cluster[i, roi_all] = 1
            return roi_cluster
        def surf_cluster_uptate(num_clusters, preds, x_surf):
            surf_cluster = numpy.zeros([num_clusters, x_surf.shape[1]])
            for i in range(num_clusters):
                t = x_surf[preds == i]
                t1 = numpy.sum(t, 0)
                surf_cluster[i] = t1 / t1.sum()
            return surf_cluster

        if args.fs:
            roi_cluster = roi_cluster_uptate(model.num_clusters, preds_final, x_roi)
            numpy.save('roi_cluster_fs.npy', roi_cluster)
        #if args.surf:
            surf_cluster = surf_cluster_uptate(model.num_clusters, preds_final, x_surf_dk)
            numpy.save('surf_cluster_fs.npy', surf_cluster)
        metrics_calculation(preds_initial, x_arrays, x_roi,x_surf_dk)
        metrics_calculation(preds_final, x_arrays, x_roi,x_surf_dk)

        cluster_colors = numpy.random.randint(0, 255, (num_clusters, 3))
        appender = vtk.vtkAppendPolyData()
        for pd in input_pds:
            if (vtk.vtkVersion().GetVTKMajorVersion() >= 6.0):
                appender.AddInputData(pd)
            else:
                appender.AddInput(pd)
        appender.Update()
        input_data = appender.GetOutput()
        outdir = args.outputDirectory
        if not os.path.exists(outdir):
            os.makedirs(outdir)

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
        if args.ro:
            #num_stds = [0.4,0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,0.01,0.015,0.02,0.025,0.03,0.035,0.040,0.045,0.050]
            num_stds=[0.7]
            DB_score_ro_thr = numpy.zeros(len(num_stds))
            wmpg_ro_thr = numpy.zeros(len(num_stds))
            tapc_ro_thr = numpy.zeros(len(num_stds))
            tspc_ro_thr = numpy.zeros(len(num_stds))
            rate_left_thr = numpy.zeros(len(num_stds))
            for i, num_std in enumerate(num_stds):
                if num_stds[i] < 0.1:
                    id_reject = numpy.where(probs_fs_surf < num_stds[i])[0]
                else:
                    id_reject = []
                    preds_fs_surf = preds_final
                    probs_fs_surf = probs_final
                    mean_atlas = numpy.zeros(num_clusters)
                    std_atlas = numpy.zeros(num_clusters)
                    for ic in range(num_clusters):
                        index = numpy.where(preds_fs_surf == ic)[0]
                        probc = probs_fs_surf[index]
                        #numpy.save('probabilities/pobc_{:03d}.npy'.format(ic), probc)
                        mean_atlas[ic] = probc.mean()
                        std_atlas[ic] = probc.std()
                        if len(probc) > 0:
                            index1 = numpy.where((probc.mean() - probc) > num_std*probc.std())[0]
                            if len(index1) > 0:
                                id_rejectc = index[index1]
                                id_reject.extend(id_rejectc)
                    #numpy.save('threshold.npy', mean_atlas - std_atlas)
                    id_reject = numpy.array(id_reject)
                probs_reject = probs_fs_surf[id_reject]
                if id_reject is not None:
                    temp = numpy.ones(len(preds_fs_surf))
                    temp[id_reject] = 0
                    mask = temp > 0
                    x_array_ro = x_arrays[mask]
                    preds_fs_surf_ro = preds_fs_surf[mask]
                    ds_fs_onehot_ro = x_roi[mask]
                    ds_surf_onehot_ro = x_surf_dk[mask]
                    DB_score_ro, wmpg_ro, tapc_ro, tspc_ro = metrics_calculation(preds_fs_surf_ro,x_array_ro, ds_fs_onehot_ro,ds_surf_onehot_ro)
                    rate_left = 1-len(preds_fs_surf_ro) / len(preds_fs_surf)
                    print('fiber_removed:', rate_left)
                    DB_score_ro_thr[i] = DB_score_ro
                    wmpg_ro_thr[i] = wmpg_ro
                    tapc_ro_thr[i] = tapc_ro
                    tspc_ro_thr[i] = tspc_ro
                    rate_left_thr[i] = rate_left
                    if num_stds[i] > 0.1:
                        print('masking removed fibers')
                        input_pd = wma.filter.mask(input_data, mask, preserve_point_data=True, preserve_cell_data=True,verbose=False)
                        print('masking clusters')
                        print(input_pd.GetNumberOfLines(),preds_fs_surf_ro.shape)
                        pd_c_list = wma.cluster.mask_all_clusters(input_pd, preds_fs_surf_ro,num_clusters,preserve_point_data=True,
                                                                              preserve_cell_data=True, verbose=False)
                        cluster_save(pd_c_list, outdir, input_data, preds_fs_surf_ro, num_clusters, cluster_colors)


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
        def fiber_cluster_save(input_pds,predicted,outfolder,x_arrays,x_fs):
            def roi_cluster_uptate(num_clusters, preds, x_fs):
                roi_cluster = numpy.zeros([num_clusters, x_fs.shape[1]])
                for i in range(num_clusters):
                    t = x_fs[preds == i]
                    t1 = numpy.sum(t, 0)
                    roi_all = numpy.where(t1 > t.shape[0] * 0.4)[0]
                    if 0 in roi_all:
                        roi_all = roi_all[1:]
                    roi_cluster[i, roi_all] = 1
                return roi_cluster
            # def roi_cluster_uptate(num_clusters, preds, x_fs):
            #     # Initialise roi distribution
            #     n_roi=len(numpy.unique(x_fs))
            #     roi_cluster = -numpy.ones((num_clusters, n_roi))
            #     for i in range(num_clusters):
            #         roi_onehot = numpy.zeros(n_roi)
            #         t = x_fs[preds == i]
            #         roi_all = numpy.unique(t)
            #         if 0 in roi_all:
            #             index_0 = roi_all != 0
            #             roi_all = roi_all[index_0]
            #         for roi in list(roi_all):
            #             x,y = numpy.where(t == roi)
            #             n = len(numpy.unique(x))
            #             # print(n)
            #             if n > t.shape[0] * 0.4:
            #                 roi_onehot[int(roi)] = 1
            #         roi_cluster[i] = roi_onehot
            #     return roi_cluster
            roi_cluster = roi_cluster_uptate(model.num_clusters, predicted, x_fs)
            loss_fn = DiceScore()
            # ds_fs_onehot = numpy.zeros((x_fs.shape[0], roi_cluster.shape[1]))
            # for f in range(x_fs.shape[0]):
            #     roi_single = numpy.unique(x_fs[f])
            #     if roi_single[0] == 0:
            #         roi_single=roi_single[1:]
            #     ds_fs_onehot[f, roi_single.astype(int)] = 1
            def tapc_calculation(preds, roi_fs,roi_cluster):
                #roi_cluster = numpy.load('roi_cluster.npy')
                roi_preds = roi_cluster[preds]
                tapc = loss_fn(roi_fs, roi_preds)
                return tapc

            tapc_train = tapc_calculation(predicted, x_fs,roi_cluster)
            DB_score = DB_index(x_arrays, predicted)
            n_detected=0
            for n in range(num_clusters):
                n_fiber=numpy.sum(predicted==n)
                if n_fiber>=20:
                    n_detected+=1
            wmpg_train=n_detected/num_clusters
            print(outfolder,DB_score,wmpg_train,tapc_train)
            #predicted[6] = 800
            # if args.fs:
            #     numpy.save('roi_cluster_fs.npy', roi_cluster)
            # else:
            #     numpy.save('roi_cluster.npy', roi_cluster)
            cluster_colors = numpy.random.randint(0, 255, (num_clusters, 3))
            # appender = vtk.vtkAppendPolyData()
            # for pd in input_pds:
            #     if (vtk.vtkVersion().GetVTKMajorVersion() >= 6.0):
            #         appender.AddInputData(pd)
            #     else:
            #         appender.AddInput(pd)
            # appender.Update()
            # input_data = appender.GetOutput()
            #outdir = os.path.join(args.outputDirectory, outfolder)
            #outdir = '/media/annabelchen/DataShare/deepClustering/cluster_dcec/journal_atlas'
            # if not os.path.exists(outdir):
            #     os.makedirs(outdir)
            # numpy.save(outdir + '/predicted.npy', predicted)
            # print('outdir:', outdir)
            # Save final model
            #torch.save(model.state_dict(), name_net + '.pt')
            # print(name)
            # pd_c_list = wma.cluster.mask_all_clusters(input_data, predicted,
            #                                                                             num_clusters,
            #                                                                             preserve_point_data=True,
            #                                                                             preserve_cell_data=True,
            #                                                                             verbose=False)
            # cluster_save(pd_c_list, outdir, input_data, predicted,
            #              num_clusters, cluster_colors)
        # fiber_cluster_save(input_pdsv,preds_km,'km',x_roiv)
        # fiber_cluster_save(input_pdsv, preds_initial, 'initial',x_arraysv,x_roiv)
        #fiber_cluster_save(input_pds, preds_final, 'final',x_arrays,x_roi)

        #args.test=False
        if args.test:
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
            rate_left_all=[]
            input_pd_fnames = wma.io.list_vtk_files(data_dirt)
            for i in range(len(input_pd_fnames)):
                subject_id = os.path.split(input_pd_fnames[i])[1].split('.')[0]
                # if os.path.exists(os.path.join('/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/test_save_PPMI',
                #                  subject_id + '_data.npz')):
                #     print(subject_id,'continue')
                #     continue
                input_pd, x_array, d_roi, fiber_surf_ve, fiber_surf_dk, fiber_surf_des = \
                    convert_fiber_to_array(input_pd_fnames[i], numberOfFibers=None,fiberLength=args.fiberLength,numberOfFiberPoints=args.numberOfFiberPoints, preproces=True,data=args.data)
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
                            # print(roi_unique)
                            # print(roi_map[0])
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
            #     numpy.savez(os.path.join('/media/annabelchen/DataShare/deepClustering/HCPTestingData/tractography_yc/test_save_PPMI',
            #                      subject_id + '_data.npz'), x_array, ds_fs_onehot, surf_dk_onehot)

            # for id,test_subject in enumerate(test_data):
            #     subject_id=test_subject.split('_')[0]
            #     print(subject_id)
            #     # if os.path.exists('../dataFolder/HCPTestingData/tractography_yc/tractography_labeled/metrics_DFC2/metrics_fs_surf/'+subject_id+'_measure.npz'):
            #     #     continue
            #     data=numpy.load(os.path.join(args.inputDirectoryt,test_subject))
            #     x_array=data['arr_0']
            #     ds_fs_onehot=data['arr_1']
            #     if not args.data=='HCP':
            #         surf_dk_onehot=data['arr_2']
            #     else:
            #         surf_dk = data['arr_3']
            #         def surf_encoding(fiber_surf_dk):
            #             fiber_surfs = fiber_surf_dk.astype(int)
            #             surf_labels = numpy.unique(fiber_surfs)
            #             surf_map = numpy.load('dk_map.npy')
            #             for surf_label in surf_labels:
            #                 fiber_surfs[numpy.where(fiber_surfs == surf_label)] = numpy.where(surf_map == surf_label)
            #             ds_surf_onehot_dk = numpy.zeros((len(fiber_surfs), len(surf_map)))
            #             for s in range(len(fiber_surfs)):
            #                 ds_surf_onehot_dk[s, fiber_surfs[s]] = 1
            #             return ds_surf_onehot_dk
            #         surf_dk_onehot=surf_encoding(surf_dk)

                if args.dataset == 'FiberMap':
                    dataset = mnist.FiberMap_pair(x_array, ds_fs_onehot,surf_dk_onehot,
                                               transform=transforms.Compose([transforms.ToTensor()]))
                elif args.dataset == 'Fiber':
                    dataset = mnist.Fiber_pair(x_array, ds_fs_onehot,surf_dk_onehot, transform=transforms.Compose([transforms.ToTensor()]))
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)
                dataset_size = len(dataset)
                print("Testing set size:\t" + str(dataset_size))

                # print(surf_cluster.shape)
                print('surf_dk_onehot',surf_dk_onehot.shape)
                #roi_cluster=numpy.load('roi_cluster_fs_1.npy')
                roi_cluster = torch.tensor(roi_cluster).to(device)
                surf_cluster = torch.tensor(surf_cluster).to(device)
                time_all = []
                for it in range(10):
                    since = time.time()
                    if args.fs:
                        preds, probs = calculate_predictions_roi(model, dataloader, params, roi_cluster=roi_cluster,
                                                                 surf_cluster=surf_cluster, surf_flag=args.surf)
                    else:
                        preds, probs = calculate_predictions_test(model, dataloader, params)

                # if id==0:
                #     list_all=[]
                #     for n in range(num_clusters):
                #         #print(n)
                #         probs_cluster = probs[preds == n]
                #         list_all.append(probs_cluster)
                #     import matplotlib.pyplot as plt
                #     figure, axes = plt.subplots(figsize=(50, 6))
                #     flierprops = dict(marker='+', markersize=3)
                #     axes.boxplot(list_all, patch_artist=True, widths=0.8, flierprops=flierprops)
                #     plt.ylim([0, 0.6])
                #     plt.xticks(rotation=90)
                #     plt.tick_params(axis='x', labelsize=4)
                    #plt.savefig('probablities_clusters_test1.png')

                # input_pd=add_prob(input_pd, probs)
                # cluster_colors = numpy.random.randint(0, 255, (num_clusters, 3))
                # pd_c_list = wma.cluster.mask_all_clusters(input_pd, preds,
                #                                           num_clusters,
                #                                           preserve_point_data=True,
                #                                           preserve_cell_data=True,
                #                                           verbose=False)
                # outdir='/media/annabelchen/DataShare/deepClustering/cluster_dcec/101006/DFC-ro'
                # cluster_save(pd_c_list, outdir, input_pd, preds,num_clusters, cluster_colors)

                    elapsed = time.time() - since
                    time_all.append(elapsed)
                time_mean = numpy.array(time_all).mean()
                print('prediction time:', time_mean)
                preds_pre, probs_pre = calculate_predictions_test(model_pretrained, dataloader, params)
                DB_score_pre, wmpg_pre, tapc_pre,tspc_pre = metrics_calculation(preds_pre, x_array, ds_fs_onehot,surf_dk_onehot)
                DB_score, wmpg, tapc,tspc = metrics_calculation(preds,x_array, ds_fs_onehot,surf_dk_onehot)
                DB_all.append(DB_score)
                wmpg_all.append(wmpg)
                tapc_all.append(tapc)
                tspc_all.append(tspc)
                DB_all_pre.append(DB_score_pre)
                wmpg_all_pre.append(wmpg_pre)
                tapc_all_pre.append(tapc_pre)
                tspc_all_pre.append(tspc_pre)

                # outdir = args.outputDirectory + '/metrics'
                # if not os.path.exists(outdir):
                #     os.makedirs(outdir)
                # print(outdir)
                # numpy.savez(os.path.join(outdir, subject_id + '_measure.npz'), DB_score_pre, wmpg_pre, tapc_pre, tspc_pre)
                # outdir = args.outputDirectory + '/metrics_fs'
                # if not os.path.exists(outdir):
                #     os.makedirs(outdir)
                # print(outdir)
                # numpy.savez(os.path.join(outdir, subject_id + '_measure.npz'), DB_score, wmpg, tapc, tspc)

                if args.ro:
                    #num_stds = [0.83] #[0.7,0.75,0.8,0.85]
                    num_stds=[0.7] #HCP  f_IRM; 0.83; open_fMRI: 0.85
                    #num_stds = [0.7, 0.045]
                    #num_stds = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
                    #num_stds=[0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,0.01,0.015,0.02,0.025,0.03,0.035,0.040,0.045]
                    DB_score_ro_thr=numpy.zeros(len(num_stds))
                    wmpg_ro_thr=numpy.zeros(len(num_stds))
                    tapc_ro_thr=numpy.zeros(len(num_stds))
                    tspc_ro_thr=numpy.zeros(len(num_stds))
                    rate_left_thr=numpy.zeros(len(num_stds))
                    preds_fs_surf = preds
                    probs_fs_surf = probs
                    for i,num_std in enumerate(num_stds):
                        rate_removed_clu = numpy.zeros(num_clusters)
                        if num_stds[i]<0.1:
                            id_reject = numpy.where(probs_fs_surf < num_stds[i])[0]
                            # for ic in range(num_clusters):
                            #     index = numpy.where(preds_fs_surf == ic)[0]
                            #     probc = probs_fs_surf[index]
                            #     num_removed=len(numpy.where(probc<num_stds[i])[0])
                            #     rate_removed_clu[ic]=num_removed/len(probc)
                        else:
                            id_reject = []
                            probs_rejected=[]
                            threshold=numpy.zeros(num_clusters)
                            for ic in range(num_clusters):
                                index = numpy.where(preds_fs_surf == ic)[0]
                                probc = probs_fs_surf[index]
                                #numpy.save('probabilities/testing/pobc_{:03d}.npy'.format(ic), probc)
                                if len(probc) > 0:
                                    mean=mean_atlas[ic]
                                    std=std_atlas[ic]
                                    #threshold[ic]=probc.mean() -probc.std()
                                    #index1 = numpy.where((probc.mean() - probc) > num_std * probc.std())[0]
                                    index1 = numpy.where((mean - probc) > num_std * std)[0]
                                    if len(index1) > 0:
                                        id_rejectc = index[index1]
                                        id_reject.extend(id_rejectc)
                                        prob_rejected=probc[index1]
                                        probs_rejected.append(prob_rejected)
                                        #rate_removed_clu[ic] = len(id_rejectc) / len(probc)
                                    # else:
                                    #     print(ic)
                            #numpy.save('threshold_test.npy',threshold)
                            id_reject = numpy.array(id_reject)
                        #numpy.save('debug_ro/rate_removed_clu_{}.npy'.format(i), rate_removed_clu)
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
                            rate_left = 1-len(preds_fs_surf_ro) / len(preds_fs_surf)
                            print('fiber_removed:', rate_left)
                            DB_score_ro_thr[i]=DB_score_ro
                            wmpg_ro_thr[i]=wmpg_ro
                            tapc_ro_thr[i]=tapc_ro
                            tspc_ro_thr[i]=tspc_ro
                            rate_left_thr[i]=rate_left
                            DB_all_ro.append(DB_score_ro_thr)
                            wmpg_all_ro.append(wmpg_ro_thr)
                            tapc_all_ro.append(tapc_ro_thr)
                            tspc_all_ro.append(tspc_ro_thr)
                            rate_left_all.append(rate_left_thr)
                        if num_stds[i] < 0.1:
                            outdir = args.outputDirectory + '/metrics_fs_ro_p'
                            print(outdir)
                            if not os.path.exists(outdir):
                                os.makedirs(outdir)
                            #numpy.savez(os.path.join(outdir, subject_id + '_measure.npz'), DB_score_ro, wmpg_ro,tapc_ro, tspc_ro, rate_left)
                        else:
                            outdir = args.outputDirectory + '/metrics_fs_surf_ro'
                            print(outdir)
                            if not os.path.exists(outdir):
                                os.makedirs(outdir)
                            #numpy.savez(os.path.join(outdir, subject_id + '_measure.npz'), DB_score_ro, wmpg_ro, tapc_ro, tspc_ro, rate_left)

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
                            #outdir = '/media/annabelchen/DataShare/deepClustering/cluster_dcec/101006/DFC_ro_{}'.format(num_stds[i])
                            #outdir = '/media/annabelchen/DataShare/deepClustering/cluster_dcec/101006/PPMI'
                            outdir=args.outputDirectory+'/'+subject_id
                            cluster_save(pd_c_list, outdir, input_pd1, preds_fs_surf_ro, num_clusters, cluster_colors)

            print('pretrained results:')
            DB_score_pre = numpy.array(DB_all_pre).mean()
            wmpg_pre = numpy.array(wmpg_all_pre).mean()
            tapc_pre = numpy.array(tapc_all_pre).mean()
            tspc_pre = numpy.array(tspc_all_pre).mean()
            print(numpy.array(DB_all_pre), numpy.array(wmpg_all_pre), numpy.array(tapc_all_pre), numpy.array(tspc_all_pre))
            print(DB_score_pre, wmpg_pre,tapc_pre,tspc_pre)
            utils.print_both(f,'DB: {0:.4f}\tWMPG: {1:.4f}\tTAPC: {2:.4f}\tTSPC: {3:.4f}'.format(DB_score_pre, wmpg_pre, tapc_pre,tspc_pre))

            print('final results:')
            DB_score=numpy.array(DB_all).mean()
            wmpg=numpy.array(wmpg_all).mean()
            tapc=numpy.array(tapc_all).mean()
            tspc = numpy.array(tspc_all).mean()
            print(numpy.array(DB_all), numpy.array(wmpg_all), numpy.array(tapc_all),numpy.array(tspc_all))
            print(DB_score, wmpg, tapc,tspc)
            utils.print_both(f,
                              'DB: {0:.4f}\tWMPG: {1:.4f}\tTAPC: {2:.4f}\tTSPC: {3:.4f}'.format(DB_score, wmpg, tapc,tspc))


            # print('outlier removal results:')
            # DB_score_ro=numpy.array(DB_all_ro).mean()
            # wmpg_ro=numpy.array(wmpg_all_ro).mean()
            # tapc_ro=numpy.array(tapc_all_ro).mean()
            # tspc_ro = numpy.array(tspc_all_ro).mean()
            # rate_left_ro = numpy.array(rate_left_all).mean()
            # print(numpy.array(DB_all_ro), numpy.array(wmpg_all_ro), numpy.array(tapc_all_ro),numpy.array(tspc_all_ro),numpy.array(rate_left_all))
            # print(DB_score_ro, wmpg_ro, tapc_ro,tspc_ro,rate_left_ro)
            # utils.print_both(f,
            #                  'DB: {0:.4f}\tWMPG: {1:.4f}\tTAPC: {2:.4f}\tTSPC: {3:.4f}\tfiber_left: {4:.4f}'.format(DB_score_ro, wmpg_ro, tapc_ro,tspc_ro,rate_left_ro))
            if args.ro:
                print('outlier removal results:')
                DB_score_ro=numpy.mean(numpy.array(DB_all_ro),0)
                wmpg_ro=numpy.mean(numpy.array(wmpg_all_ro),0)
                tapc_ro=numpy.mean(numpy.array(tapc_all_ro),0)
                tspc_ro = numpy.mean(numpy.array(tspc_all_ro),0)
                rate_left_ro = numpy.mean(numpy.array(rate_left_all),0)
                #print(numpy.array(DB_all_ro), numpy.array(wmpg_all_ro), numpy.array(tapc_all_ro),numpy.array(tspc_all_ro),numpy.array(rate_left_all))
                print(DB_score_ro)
                print(wmpg_ro)
                print(tapc_ro)
                print(tspc_ro)
                print(rate_left_ro)
                # utils.print_both(f,
                #                  'DB: {0:.4f}\tWMPG: {1:.4f}\tTAPC: {2:.4f}\tTSPC: {3:.4f}\tfiber_left: {4:.4f}'.format(DB_score_ro, wmpg_ro, tapc_ro,tspc_ro,rate_left_ro))




    elif args.mode == 'pretrain':
        model = training_functions_fiber_pair.pretraining(model, dataloader,dataloaderv, criteria[0], optimizers[1], schedulers[1], pretrain_epochs, params)
        # Save final model
        #torch.save(model.state_dict(), name_net + '_{}.pt'.format(epochs))
        #torch.save(model.state_dict(), name_net + '.pt')
        print(name)


    # Close files
    f.close()
    if board:
        writer.close()