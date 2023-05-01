import utils
import time
import torch
import numpy as np
import numpy
import copy
from sklearn.cluster import KMeans
import torch.nn as nn
import fiber_distance
import tracemalloc

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
    DB=numpy.mean(DB_all)
    return  DB
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
def metrics_calculation(num_clusters,predicted,x_arrays,x_fs):
    def roi_cluster_uptate(num_clusters, preds, x_fs):
        roi_cluster = np.zeros([num_clusters, x_fs.shape[1]])
        for i in range(num_clusters):
            t = x_fs[preds == i]
            t1 = np.sum(t, 0)
            roi_all = np.where(t1 > t.shape[0] * 0.4)[0]
            if 0 in roi_all:
                roi_all = roi_all[1:]
            roi_cluster[i, roi_all] = 1
        return roi_cluster
    roi_cluster = roi_cluster_uptate(num_clusters, predicted, x_fs)
    loss_fn = DiceScore()
    def tapc_calculation(preds, roi_fs, roi_cluster):
        # roi_cluster = numpy.load('roi_cluster.npy')
        roi_preds = roi_cluster[preds]
        tapc = loss_fn(roi_fs, roi_preds)
        return tapc
    tapc_train = tapc_calculation(predicted, x_fs, roi_cluster)
    DB_score = DB_index(x_arrays, predicted)
    n_detected = 0
    for n in range(num_clusters):
        n_fiber = np.sum(predicted == n)
        if n_fiber >= 30:
            n_detected += 1
    wmpg_train = n_detected / num_clusters
    return DB_score, wmpg_train, tapc_train

# Training function (from my torch_DCEC implementation, kept for completeness)
def train_model(model, dataloader,dataloaderv, dataloader1, criteria, optimizers, schedulers, num_epochs, params,x_fs,fs_flag,x_surf,surf_flag):
    #tracemalloc.start()
    # Note the time
    since = time.time()
    loss_fn = DiceLoss()
    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    #pretrained =  params['model_files'][1] #' #   # 'nets/CAE_3_pair_001_pretrained.pt' params['model_pretrained'] #
    pretrained = 'nets/DGCNN_002_1_fs_surf.pt'
    #pretrained = 'nets/GCN_166.pt'
    #pretrained='nets/GCN_039_1_fs_surf.pt'
    #pretrained = 'nets/CAE_3_pair_001_fs_1.pt'
    #pretrained= '/home/annabelchen/PycharmProjects/torch_DCEC_MICCAI/nets/CAE_3_pair_001_fs_1.pt'#'nets/CAE_3_pair_001_fs_1.pt' #/home/annabelchen/PycharmProjects/torch_DCEC_MICCAI/nets/CAE_3_pair_130_pretrained.pt'#
    #pretrained = 'nets/DGCNN_019_1_fs.pt'
    pretrain = params['pretrain']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']
    pretrain_epochs = params['pretrain_epochs']
    gamma = params['gamma']
    update_interval = params['update_interval']
    tol = params['tol']

    #model.load_state_dict(torch.load(pretrained))
    dl = dataloader
    dlv = dataloaderv

    # model_pretrained = copy.deepcopy(model)
    # model_pretrained.load_state_dict(torch.load('nets/DGCNN_019_1.pt'))
    # Pretrain or load weights
    if pretrain:
        while True:
            pretrained_model = pretraining(model, copy.deepcopy(dl),copy.deepcopy(dlv), criteria[0], optimizers[1], schedulers[1], pretrain_epochs, params)
            if pretrained_model:
                break
            else:
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        model = pretrained_model
        utils.print_both(txt_file, '\nInitializing cluster centers based on K-means')
        kmeans(model, copy.deepcopy(dl), params )
    else:
        try:
            model.load_state_dict(torch.load(pretrained))
            #weights = torch.load('weights_GCN.pt')
            #model.clustering.set_weight(weights.to(params['device']))
            utils.print_both(txt_file, 'Pretrained weights loaded from file: ' + str(pretrained))
        except:
            print("Couldn't load pretrained weights")

    # Initialise clusters
    # if pretrain:
    #     preds_km=kmeans(model, copy.deepcopy(dl), params,pretrained)
    # else:
    #     preds_km = np.load('preds_km.npy')
    #     weights = torch.load('weights.pt')
    #     model.clustering.set_weight(weights.to(params['device']))

    utils.print_both(txt_file, '\nBegin clusters training')

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    best_DB = 10000.0
    best_epoch=0

    # Initial target distribution
    utils.print_both(txt_file, '\nUpdating target distribution')
    preds_initial, probs_initial = calculate_predictions_test(model, copy.deepcopy(dataloader1), params)
    #if num_epochs>0:
        #output_distribution, preds_initial = calculate_predictions(model, copy.deepcopy(dataloader1), params)
        #target_distribution = target(output_distribution)
        #DB_score, wmpg_train, tapc_train = metrics_calculation(model.num_clusters, preds_initial, x_arraysv, x_fsv)
        # epoch=0
        # writer.add_scalar('/DB', DB_score, epoch)
        # writer.add_scalar('/wmpg', wmpg_train, epoch)
        # writer.add_scalar('/tapc', tapc_train, epoch)
        # utils.print_both(txt_file, 'DB: {0:.4f}\tWMPG: {1:.4f}\tTAPC: {2:.4f}'.format(DB_score, wmpg_train, tapc_train))

    preds_uptated = torch.tensor(preds_initial).to(device)
    if fs_flag:
        def roi_cluster_uptate(num_clusters, preds, x_fs, device):
            roi_cluster = torch.zeros([num_clusters, x_fs.shape[1]])
            for i in range(num_clusters):
                t = x_fs[preds == i]
                t1 = torch.sum(t, 0)
                roi_all = (t1 > t.shape[0] * 0.4).nonzero()
                if 0 in roi_all:
                    roi_all = roi_all[1:]
                roi_cluster[i, roi_all.long()] = 1
            # Initialise roi distribution
            # roi_cluster = -torch.ones([num_clusters,len(torch.unique(x_fs))])
            # for i in range(num_clusters):
            #     roi_onehot = torch.zeros([len(torch.unique(x_fs))])
            #     t=x_fs[preds==i]
            #     roi_all=torch.unique(t)
            #     if 0 in roi_all:
            #         index_0=roi_all!=0
            #         roi_all=roi_all[index_0]
            #     for roi in list(roi_all):
            #         x=(t==roi).nonzero()[:,0]
            #         n=len(torch.unique(x))
            #         #print(n)
            #         if n>t.shape[0]*0.4:
            #             roi_onehot[roi.long()]=1
            #     roi_cluster[i] = roi_onehot
            roi_cluster = roi_cluster.to(device)
            return roi_cluster
        preds_km = torch.tensor(preds_initial).to(device)
        x_fs = torch.tensor(x_fs).to(device)
        roi_cluster = roi_cluster_uptate(model.num_clusters, preds_km, x_fs, device)
    if surf_flag:
        def surf_cluster_uptate(num_clusters, preds, x_surf, device):
            surf_cluster = torch.zeros([num_clusters, x_surf.shape[1]])
            for i in range(num_clusters):
                t = x_surf[preds == i]
                t1 = torch.sum(t, 0)
                if t1.sum()==0:
                    continue
                surf_cluster[i] = t1/t1.sum()
            surf_cluster = surf_cluster.to(device)
            return surf_cluster
        preds_km = torch.tensor(preds_initial).to(device)
        x_surf = torch.tensor(x_surf).to(device)
        surf_cluster = surf_cluster_uptate(model.num_clusters, preds_km, x_surf, device)

    model_pretrained=copy.deepcopy(model)
    finished = False
    # Go through all epochs
    for epoch in range(num_epochs):
        utils.print_both(txt_file, 'Epoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file,  '-' * 10)

        schedulers[0].step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_clust = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            input1,input2,sim,roi_bat,surf_bat,index = data
            input1 = input1.to(device)
            input2 = input2.to(device)
            sim = sim.to(device)
            roi_bat = roi_bat.to(device)
            index = index.to(device)
            surf_bat = surf_bat.to(device)

            # Uptade target distribution, chack and print performance
            if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0):
                utils.print_both(txt_file, '\nUpdating target distribution:')
                #preds_epoch, _ = calculate_predictions_test(model, copy.deepcopy(dataloader1), params)
                # output_distribution,  preds_epoch = calculate_predictions(model, dataloader1, params)
                # target_distribution = target(output_distribution)
                if fs_flag:
                    roi_cluster = roi_cluster_uptate(model.num_clusters, preds_uptated,x_fs, device)
                if surf_flag:
                    surf_cluster = surf_cluster_uptate(model.num_clusters, preds_uptated, x_surf, device)
                # if batch_num==1:
                #     preds_epoch, _ = calculate_predictions_test(model, copy.deepcopy(dataloaderv), params)
                #  s   DB_score, wmpg_train, tapc_train = metrics_calculation(model.num_clusters, preds_epoch, x_arraysv, x_fsv)
                #     writer.add_scalar('/DB', DB_score, epoch)
                #     writer.add_scalar('/wmpg', wmpg_train, epoch)
                #     writer.add_scalar('/tapc', tapc_train, epoch)
                #     utils.print_both(txt_file, 'DB: {0:.4f}\tWMPG: {1:.4f}\tTAPC: {2:.4f}'.format(DB_score, wmpg_train, tapc_train))

                # if fs_flag:
                #     roi_cluster = roi_cluster_uptate(model.num_clusters, preds_uptated,x_fs, device)
            #     # check stop criterion
            #     delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
            #     preds_prev = np.copy(preds)
            #     if delta_label < tol:
            #         utils.print_both(txt_file, 'Label divergence ' + str(delta_label) + '< tol ' + str(tol))
            #         utils.print_both(txt_file, 'Reached tolerance threshold. Stopping training.')
            #         finished = True
            #         break

            # tar_dist = target_distribution[((batch_num - 1) * batch):(batch_num*batch), :]
            # tar_dist = torch.from_numpy(tar_dist).to(device)
            #print(tar_dist)

            # zero the parameter gradients
            optimizers[0].zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                outputs, clusters, _,_,_,dis_point = model(input1,input2)
                # if fs_flag:
                #     roi_3D = roi_bat.unsqueeze(1).to(device)
                #     roi_3D = roi_3D.repeat(1, model.num_clusters, 1)
                #     roi_3D = torch.reshape(roi_3D, (-1, roi_3D.shape[-1]))
                #     roi_cluster_3D = roi_cluster.repeat(outputs.shape[0], 1)
                #     dis_roi1 = loss_fn(roi_3D, roi_cluster_3D)
                #     dis_roi = torch.reshape(dis_roi1, (outputs.shape[0], model.num_clusters))
                #     # calculate new p
                #     # alpha = dis_point.max()
                #     _,pred_point=dis_point.min(1)
                #     _, pred_roi = dis_roi.min(1)
                #     index1=(pred_point!=pred_roi).nonzero()
                #     index_dif=[]
                #     for i in index1:
                #         if dis_roi[i, pred_roi[i]] != dis_roi[i, pred_point[i]]:
                #             index_dif.append(i)
                #     #print(len(index_dif))
                #     x = 1.0 + dis_point * dis_roi
                #     x = 1.0 / x
                #     x = torch.t(x) / torch.sum(x, dim=1)
                #     x = torch.t(x)
                #     clusters = x

                if fs_flag:
                    roi_3D = roi_bat.unsqueeze(1).to(device)
                    roi_3D = roi_3D.repeat(1, model.num_clusters, 1)
                    roi_3D = torch.reshape(roi_3D, (-1, roi_3D.shape[-1]))
                    roi_cluster_3D = roi_cluster.repeat(outputs.shape[0], 1)
                    dis_roi1 = loss_fn(roi_3D, roi_cluster_3D)
                    dis_roi = torch.reshape(dis_roi1, (outputs.shape[0], model.num_clusters))
                if surf_flag:
                    #temp=torch.mm(surf_bat.float(),surf_cluster.t())
                    # temp[temp<0]=0
                    # temp[temp > 1] = 1
                    # dis_surf = 1 -temp
                    #dis_surf = 1 - temp / temp.max()
                    dis_surf=1-torch.mm(surf_bat.float(),surf_cluster.t())/1.5
                    # dis_surf=torch.ones((dis_roi.shape[0],dis_roi.shape[1]))
                    # for i_s in range(surf_bat.shape[0]):
                    #     surf_fiber=surf_bat[i_s]
                    #     inds=surf_fiber.nonzero().squeeze()
                    #     if inds.shape==torch.Size([0]):  #no
                    #         pass
                    #     elif inds.shape==torch.Size([]):  #one
                    #         dis_surf[i_s]=1-surf_cluster[:,inds]*surf_fiber[inds]
                    #         #print(surf_fiber[inds])
                    #     elif inds.shape==torch.Size([2]):    #two
                    #         dis_surf[i_s] = 1-torch.sum(surf_cluster[:, inds],1)
                    #     else:
                    #         print('Error: more than two labels for a fiber',inds)
                    #         exit()
                    dis_surf=dis_surf.to(device)
                    #print(surf_cluster.max())
                if fs_flag or surf_flag:
                    if fs_flag and not surf_flag:
                        # dis_point=1000
                        x = 1.0 + dis_point * dis_roi
                        # predsp=torch.argmin(dis_point,1)
                        # predsr=torch.argmin(dis_roi,1)
                        # print(predsp,predsr)
                        #print('fs')
                    elif not fs_flag and surf_flag:
                        # dis_point=1000
                        x = 1.0 + dis_point * dis_surf
                        #print('surf')
                    else:
                        x = 1.0 + dis_point * dis_roi * dis_surf
                        #print('fs,surf')
                    x = 1.0 / x
                    x = torch.t(x) / torch.sum(x, dim=1)
                    x = torch.t(x)
                    clusters = x

                _, preds = torch.max(clusters, 1)
                # dis_sort, dis_index = torch.sort(dis_point, 1)
                # #dis_sort=torch.nn.functional.normalize(dis_sort)
                # margin=200
                # loss_clust=gamma*torch.mean(torch.clamp(dis_sort[:,0]-dis_sort[:,1]+margin,min=0))

                # probs, preds = torch.max(clusters, 1)
                # loss=torch.sum(-torch.log(probs))/batch
                #print(torch.sum(preds!=pred_point))
                preds_uptated[index] = preds
                #output_distribution[index.cpu().numpy()]=clusters.detach().cpu().numpy()
                loss_rec = criteria[0](outputs, sim)
                tar_dist = target_distribution(clusters)
                loss_clust = gamma *criteria[1](torch.log(clusters), tar_dist) / batch
                loss = loss_rec + loss_clust
                loss.backward()
                optimizers[0].step()

            # For keeping statistics
            running_loss += loss.item() * input1.size(0)
            running_loss_rec += loss_rec.item() * input1.size(0)
            running_loss_clust += loss_clust.item() * input1.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_batch_rec = loss_rec.item()
            loss_batch_clust = loss_clust.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + input1.size(0))
            loss_accum_rec = running_loss_rec / ((batch_num - 1) * batch + input1.size(0))
            loss_accum_clust = running_loss_clust / ((batch_num - 1) * batch + input1.size(0))

            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'Epoch: [{0}][{1}/{2}]\t'
                                           'Loss {3:.4f} ({4:.4f})\t'
                                           'Loss_recovery {5:.4f} ({6:.4f})\t'
                                           'Loss clustering {7:.4f} ({8:.4f})\t'.format(epoch + 1, batch_num,
                                                                                        len(dataloader),
                                                                                        loss_batch,
                                                                                        loss_accum, loss_batch_rec,
                                                                                        loss_accum_rec,
                                                                                        loss_batch_clust,
                                                                                        loss_accum_clust))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('/Loss', loss_accum, niter)
                    writer.add_scalar('/Loss_recovery', loss_accum_rec, niter)
                    writer.add_scalar('/Loss_clustering', loss_accum_clust, niter)
            batch_num = batch_num + 1
            # Print image to tensorboard
            # if batch_num == len(dataloader) and (epoch+1) % 5:
            #     inp = utils.tensor2img(inputs)
            #     out = utils.tensor2img(outputs)
            #     if board:
            #         img = np.concatenate((inp, out), axis=1)
            #         #writer.add_image('Clustering/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
            #         img_counter += 1

        #preds_epoch, _ = calculate_predictions_test(model, copy.deepcopy(dl1), params)

        if finished: break

        epoch_loss = running_loss / dataset_size
        epoch_loss_rec = running_loss_rec / dataset_size
        epoch_loss_clust = running_loss_clust / dataset_size

        if board:
            writer.add_scalar('/Loss' + '/Epoch', epoch_loss, epoch + 1)
            writer.add_scalar('/Loss_rec' + '/Epoch', epoch_loss_rec, epoch + 1)
            writer.add_scalar('/Loss_clust' + '/Epoch', epoch_loss_clust, epoch + 1)

        utils.print_both(txt_file, 'Loss: {0:.4f}\tLoss_recovery: {1:.4f}\tLoss_clustering: {2:.4f}'.format(epoch_loss,
                                                                                                            epoch_loss_rec,
                                                                                                            epoch_loss_clust))

        # If wanted to do some criterium in the future (for now useless)
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        # if DB_score < best_DB:
        #     best_DB = DB_score
        #     best_epoch=epoch+1
        #     best_wmpg=wmpg_train
        #     best_tapc = tapc_train
        #     best_model_wts = copy.deepcopy(model.state_dict())

        # if pretrain:
        #     name_model = pretrained[:-13] + str(epochs) + '.pt'
        # else:
        # name_model = params['model_files'][0] + '_{}.pt'.format(epoch)
        # torch.save(model.state_dict(), name_model)
        # utils.print_both(txt_file, '')

        # utils.print_both(txt_file, '')
        # model.eval()  # 不启用batch normalization（直接采用训练好的参数）和dropout
        # with torch.no_grad():
        #     lossv_sum = 0
        #     count = 0
        #     for data in dataloaderv:
        #         #input1, input2, fiber1, fiber2, sim, _, _ = data
        #         input1, input2, sim, _, _,_ = data
        #         input1 = input1.to(device)
        #         input2 = input2.to(device)
        #         # fiber1 = fiber1.to(device)
        #         # fiber2 = fiber2.to(device)
        #         sim = sim.to(device)
        #         outputs, clusters, _, _, _, _ = model(input1, input2)
        #         lossb = criteria[0](outputs, sim)
        #         lossv_sum += lossb
        #         count += 1
        #     lossv = lossv_sum / count
        #
        #     utils.print_both(txt_file, 'Validation:\t Loss: {:.4f}'.format(lossv))
        #     writer.add_scalar('Validation/Loss' + '/Epoch', lossv, epoch + 1)


    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # utils.print_both(txt_file, 'memory usage:'+str(tracemalloc.get_traced_memory()[1]))
    # tracemalloc.stop()

    # load best model weights
    #preds_uptated = preds_uptated.cpu().numpy()
    print(fs_flag,surf_flag)
    if fs_flag:
        if surf_flag:
            preds, probs = calculate_predictions_roi(model, dataloader1, params, roi_cluster=roi_cluster,surf_cluster=surf_cluster,surf_flag=True)
        else:
            preds, probs = calculate_predictions_roi(model, dataloader1, params, roi_cluster=roi_cluster)
    else:
        if num_epochs>0:
            preds, probs = calculate_predictions_test(model, dataloader1, params)
        else:
            preds = preds_initial
            probs=probs_initial
    return model_pretrained,model,preds_initial, preds,probs


# Pretraining function for recovery loss only
def pretraining(model, dataloader,dataloaderv, criterion, optimizer, scheduler, num_epochs, params):
    # Note the time
    since = time.time()

    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = params['model_files'][1]
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    since_epoch=time.time()
    # Go through all epochs
    batch_num_all=0
    for epoch in range(num_epochs):
        time_epoch = time.time() - since_epoch
        print('time_epoch:', time_epoch)
        utils.print_both(txt_file, 'time_epoch: {}'.format(time_epoch))
        since_epoch = time.time()
        utils.print_both(txt_file, 'Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file, '-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        cur_lr = optimizer.param_groups[-1]['lr']
        print('cur_lr:', cur_lr)

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        # Images to show
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            #input1, input2,fiber1,fiber2,sim_score,_,_ = data
            input1, input2, sim_score, _, _,_ = data
            input1 = input1.to(device)
            input2 = input2.to(device)
            # fiber1 = fiber1.to(device)
            # fiber2 = fiber2.to(device)
            sim_score = sim_score.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, _, _,_,_,_ = model(input1,input2)
                loss = criterion(outputs, sim_score)
                #print('outputs',outputs)
                #print('pseudo gt', sim_score)
                #loss=loss*100
                #loss= -torch.log(1-abs(sim_score-outputs))
                loss.backward()
                optimizer.step()

            # For keeping statistics
            running_loss += loss.item() * input1.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + input1.size(0))

            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'Pretraining:\tEpoch: [{0}][{1}/{2}]\t'
                           'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloader),
                                                             loss_batch,
                                                             loss_accum))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('Pretraining/Loss', loss_accum, niter)
            batch_num = batch_num + 1
            batch_num_all = batch_num_all + 1
            #print(batch_num_all)
            # if batch_num_all==25000:
            #     break

            # if batch_num in [len(dataloader), len(dataloader)//2, len(dataloader)//4, 3*len(dataloader)//4]:
            #     inp = utils.tensor2img(inputs)
            #     out = utils.tensor2img(outputs)
            #     if board:
            #         img = np.concatenate((inp, out), axis=1)
            #         #writer.add_image('Pretraining/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
            #         img_counter += 1
        # if epoch % 10==0 and epoch>0:
        #     name_model = params['model_files'][0] + '_pretrained_{}.pt'.format(epoch)
        #     torch.save(model.state_dict(), name_model)
        epoch_loss = running_loss / dataset_size
        if epoch == 0: first_loss = epoch_loss
        if epoch == 4 and epoch_loss / first_loss > 1:
            utils.print_both(txt_file, "\nLoss not converging, starting pretraining again\n")
            return False

        if board:
            writer.add_scalar('Pretraining/Loss' + '/Epoch', epoch_loss, epoch + 1)

        utils.print_both(txt_file, 'Pretraining:\t Loss: {:.4f}'.format(epoch_loss))

        # If wanted to add some criterium in the future
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        utils.print_both(txt_file, '')
        model.eval()  # 不启用batch normalization（直接采用训练好的参数）和dropout
        with torch.no_grad():
            lossv_sum = 0
            count = 0
            for data in dataloaderv:
                #input1, input2, fiber1, fiber2, sim, _, _ = data
                input1, input2, sim, _, _,_ = data
                input1 = input1.to(device)
                input2 = input2.to(device)
                # fiber1 = fiber1.to(device)
                # fiber2 = fiber2.to(device)
                sim = sim.to(device)
                outputs, clusters, _, _, _, _ = model(input1, input2)
                lossb = criterion(outputs, sim)
                lossv_sum += lossb
                count += 1
            lossv = lossv_sum / count

            utils.print_both(txt_file, 'Validation:\t Loss: {:.4f}'.format(lossv))
            writer.add_scalar('Validation/Loss' + '/Epoch', lossv, epoch + 1)
        # if batch_num_all == 25000:
        #     break

        # if lossv < best_loss:
        #     best_loss = lossv
        #     best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Pretraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.pretrained = True
    #torch.save(model.state_dict(), pretrained)

    return model


# K-means clusters initialisation
def kmeans(model, dataloader, params):
    #pretrained=params['model_files'][1]
    pretrained = params['model_files'][1]
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    output_array = None
    model.eval()
    x_input=[]
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        input1,input2,sim_score,_,_,_ = data
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])
        _, _, _,outputs,_ ,_= model(input1,input2)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
        #print(output_array.shape)
        #if output_array.shape[0] > 50000: break
        #x_input.append(inputs.detach().cpu())

    # Perform K-means
    predicted=km.fit_predict(output_array)
    # x = torch.cat(x_input).numpy()
    # x_sc=np.reshape(x.transpose((0,2,1)),(x.shape[0],-1))
    # s=metrics.silhouette_score(x_sc, predicted, metric='euclidean')
    # print('initial s:',s)

    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    model.clustering.set_weight(weights.to(params['device']))
    torch.save(weights, "weights.pt")
    np.save('preds_km.npy', predicted)
    torch.save(model.state_dict(), pretrained)
    print(pretrained)
    # torch.cuda.empty_cache()
    return predicted


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, params):
    output_array = None
    x_recs=None
    embeddings=None
    model.eval()
    for data in dataloader:
        input1,input2,sim_score,_,_,_ = data
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])
        x_rec, outputs, _,embedding,_,_ = model(input1,input2)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            x_recs= np.concatenate(( x_recs, x_rec.cpu().detach().numpy()), 0)
            embeddings= np.concatenate(( embeddings, embedding.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
            x_recs = x_rec.cpu().detach().numpy()
            embeddings=embedding.cpu().detach().numpy()

    preds = np.argmax(output_array.data, axis=1)
    # print(output_array.shape)
    return output_array, preds #x_recs, embeddings
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
            preds=np.concatenate((preds, preds_single.cpu().detach().numpy()), 0)
            probs = np.concatenate((probs, probs_single.cpu().detach().numpy()), 0)
        else:
            preds = preds_single.cpu().detach().numpy()
            probs = probs_single.cpu().detach().numpy()

    return  preds,probs #x_recs, embeddings
def calculate_predictions_roi(model, dataloader, params,surf_flag=False,surf_type=None,roi_cluster=None,surf_cluster=None):
    loss_fn=DiceLoss()
    import numpy
    if surf_flag:
        if surf_cluster is None:
            #surf_cluster = numpy.load('cluster_{}/surf_cluster_fs_surf_{}.npy'.format(model.num_clusters,surf_type))
            surf_cluster = numpy.load('surf_cluster_fs.npy')
            surf_cluster = torch.tensor(surf_cluster).to(params['device'])
        if roi_cluster is None:
            #roi_cluster = numpy.load('cluster_{}/roi_cluster_fs_surf_{}.npy'.format(model.num_clusters,surf_type))
            surf_cluster = numpy.load('surf_cluster_fs.npy')
            roi_cluster = torch.tensor(roi_cluster).to(params['device'])
    else:
        if roi_cluster is None:
            roi_cluster=numpy.load('roi_cluster_fs.npy'.format(model.num_clusters))
            roi_cluster=torch.tensor(roi_cluster).to(params['device'])

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
            preds=np.concatenate((preds, preds_single.cpu().detach().numpy()), 0)
            probs = np.concatenate((probs, probs_single.cpu().detach().numpy()), 0)
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
# def calculate_predictions_roi(model, dataloader, params):
#     loss_fn=DiceLoss()
#     import numpy
#     roi_cluster=numpy.load('roi_cluster_fs.npy')
#     roi_cluster=torch.tensor(roi_cluster).to(params['device'])
#     model.eval()
#     preds=None
#     probs=None
#     for data in dataloader:
#         input1,input2,sim_score,roi_bat,_,_ = data
#         input1 = input1.to(params['device'])
#         input2 = input2.to(params['device'])
#         roi_bat = roi_bat.to(params['device'])
#         outputs, clusters, _,embedding,_,dis_point = model(input1,input2)
#         n_label=64
#         # roi_batch = torch.zeros([outputs.shape[0], n_label])
#         # for j in range(outputs.shape[0]):
#         #     roi_onehot = torch.zeros([n_label]).to(params['device'])
#         #     roi_label = torch.unique(roi_bat[j])
#         #     if 0 in roi_label:
#         #         index_0 = roi_label != 0
#         #         roi_label = roi_label[index_0]
#         #     roi_onehot[roi_label.long()] = 1
#         #     roi_batch[j] = roi_onehot
#         roi_3D = roi_bat.unsqueeze(1).to(params['device'])
#         roi_3D = roi_3D.repeat(1, model.num_clusters, 1)
#         roi_3D = torch.reshape(roi_3D, (-1, roi_3D.shape[-1]))
#         roi_cluster_3D = roi_cluster.repeat(outputs.shape[0], 1)
#         dis_roi1 = loss_fn(roi_3D, roi_cluster_3D)
#         dis_roi = torch.reshape(dis_roi1, (outputs.shape[0], model.num_clusters))
#         # calculate new p
#         # alpha = dis_point.max()
#         _, pred_point = dis_point.min(1)
#         _, pred_roi = dis_roi.min(1)
#         x = 1.0 + dis_point * dis_roi
#         x = 1.0 / x
#         x = torch.t(x) / torch.sum(x, dim=1)
#         x = torch.t(x)
#         clusters = x
#         #_, preds = torch.max(clusters, 1)
#
#         probs_single,preds_single=torch.max(clusters,1)
#         if preds is not None:
#             preds=np.concatenate((preds, preds_single.cpu().detach().numpy()), 0)
#             probs = np.concatenate((probs, probs_single.cpu().detach().numpy()), 0)
#         else:
#             preds = preds_single.cpu().detach().numpy()
#             probs = probs_single.cpu().detach().numpy()
#
#         # if output_array is not None:
#         #     output_array = np.concatenate((output_array, clusters.cpu().detach().numpy()), 0)
#         # else:
#         #     output_array = clusters.cpu().detach().numpy()
#
#     # preds = np.argmax(output_array.data, axis=1)
#     # probs = np.max(output_array.data, axis=1)
#     # print(output_array.shape)
#     return  preds,probs

# Calculate target distribution
def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist
def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
