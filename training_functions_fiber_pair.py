import utils
import time
import torch
import numpy as np
import copy
from sklearn.cluster import KMeans
import torch.nn as nn

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
        loss = 1.5 - loss#.sum() / N
        return loss

# Training function (from my torch_DCEC implementation, kept for completeness)
def train_model(model, dataloader, criteria, optimizers, schedulers, num_epochs, params,x_fs,fs_flag):

    # Note the time
    since = time.time()
    loss_fn = DiceLoss()
    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = 'nets/CAE_3_pair_130_pretrained.pt'#params['model_files'][1] #' #   # 'nets/CAE_3_pair_001_pretrained.pt'
    pretrain = params['pretrain']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']
    pretrain_epochs = params['pretrain_epochs']
    gamma = params['gamma']
    update_interval = params['update_interval']
    tol = params['tol']

    dl = dataloader

    # Pretrain or load weights
    if pretrain:
        while True:
            pretrained_model = pretraining(model, copy.deepcopy(dl), criteria[0], optimizers[1], schedulers[1], pretrain_epochs, params)
            if pretrained_model:
                break
            else:
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        model = pretrained_model
    else:
        try:
            model.load_state_dict(torch.load(pretrained))
            utils.print_both(txt_file, 'Pretrained weights loaded from file: ' + str(pretrained))
        except:
            print("Couldn't load pretrained weights")

    # Initialise clusters
    utils.print_both(txt_file, '\nInitializing cluster centers based on K-means')
    #preds_km=kmeans(model, copy.deepcopy(dl), params)
    preds_km = np.load('preds_km.npy')
    weights = torch.load('weights.pt')
    model.clustering.set_weight(weights.to(params['device']))

    utils.print_both(txt_file, '\nBegin clusters training')

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Initial target distribution
    utils.print_both(txt_file, '\nUpdating target distribution')
    preds_initial,_ = calculate_predictions_test(model, copy.deepcopy(dl), params)
    #output_distribution, preds_initial = calculate_predictions(model, copy.deepcopy(dl), params)
    preds_uptated = torch.tensor(preds_initial).to(device)
    #target_distribution = target(output_distribution)

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
        preds_km = torch.tensor(preds_km).to(device)
        x_fs = torch.tensor(x_fs).to(device)
        roi_cluster = roi_cluster_uptate(model.num_clusters, preds_km, x_fs, device)

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
            input1,input2,sim,roi_bat,index = data
            input1 = input1.to(device)
            input2 = input2.to(device)
            sim = sim.to(device)
            roi_bat = roi_bat.to(device)
            index = index.to(device)

            # Uptade target distribution, chack and print performance
            if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0):
            #     utils.print_both(txt_file, '\nUpdating target distribution:')
            #     output_distribution,  preds = calculate_predictions(model, dataloader, params)
                #target_distribution = target(output_distribution)
                if fs_flag:
                    roi_cluster = roi_cluster_uptate(model.num_clusters, preds_uptated,x_fs, device)
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
            # print(tar_dist)

            # zero the parameter gradients
            optimizers[0].zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                outputs, clusters, _,_,_,dis_point = model(input1,input2)
                if fs_flag:
                    roi_3D = roi_bat.unsqueeze(1).to(device)
                    roi_3D = roi_3D.repeat(1, model.num_clusters, 1)
                    roi_3D = torch.reshape(roi_3D, (-1, roi_3D.shape[-1]))
                    roi_cluster_3D = roi_cluster.repeat(outputs.shape[0], 1)
                    dis_roi1 = loss_fn(roi_3D, roi_cluster_3D)
                    dis_roi = torch.reshape(dis_roi1, (outputs.shape[0], model.num_clusters))
                    # calculate new p
                    # alpha = dis_point.max()
                    _,pred_point=dis_point.min(1)
                    _, pred_roi = dis_roi.min(1)
                    index1=(pred_point!=pred_roi).nonzero()
                    index_dif=[]
                    for i in index1:
                        if dis_roi[i, pred_roi[i]] != dis_roi[i, pred_point[i]]:
                            index_dif.append(i)
                    #print(len(index_dif))
                    x = 1.0 + dis_point * dis_roi
                    x = 1.0 / x
                    x = torch.t(x) / torch.sum(x, dim=1)
                    x = torch.t(x)
                    clusters = x
                _, preds = torch.max(clusters, 1)
                #print(torch.sum(preds!=pred_point))
                preds_uptated[index] = preds
                #output_distribution[index.cpu().numpy()]=clusters.detach().cpu().numpy()
                tar_dist = target_distribution(clusters)
                loss_rec = criteria[0](outputs, sim)
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

        utils.print_both(txt_file, '')

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    preds_uptated = preds_uptated.cpu().numpy()
    #output_distribution, preds = calculate_predictions(model, dataloader, params)
    if fs_flag:
        preds_km=preds_km.cpu().numpy()
    return model,preds_km, preds_uptated


# Pretraining function for recovery loss only
def pretraining(model, dataloader, criterion, optimizer, scheduler, num_epochs, params):
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

    # Go through all epochs
    for epoch in range(num_epochs):
        utils.print_both(txt_file, 'Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file, '-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        # Images to show
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            input1, input2,sim_score,_,_ = data
            input1 = input1.to(device)
            input2 = input2.to(device)
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

            # if batch_num in [len(dataloader), len(dataloader)//2, len(dataloader)//4, 3*len(dataloader)//4]:
            #     inp = utils.tensor2img(inputs)
            #     out = utils.tensor2img(outputs)
            #     if board:
            #         img = np.concatenate((inp, out), axis=1)
            #         #writer.add_image('Pretraining/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
            #         img_counter += 1

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

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Pretraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.pretrained = True
    torch.save(model.state_dict(), pretrained)

    return model


# K-means clusters initialisation
def kmeans(model, dataloader, params):
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    output_array = None
    model.eval()
    x_input=[]
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        input1,input2,sim_score,_,_ = data
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
    # torch.cuda.empty_cache()
    return predicted


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, params):
    output_array = None
    x_recs=None
    embeddings=None
    model.eval()
    for data in dataloader:
        input1,input2,sim_score,_,_ = data
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
        input1,input2,sim_score,_,_ = data
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
def calculate_predictions_roi(model, dataloader, params):
    loss_fn=DiceLoss()
    import numpy
    roi_cluster=numpy.load('roi_cluster_fs_1.npy')
    roi_cluster=torch.tensor(roi_cluster).to(params['device'])
    model.eval()
    preds=None
    probs=None
    for data in dataloader:
        input1,input2,sim_score,roi_bat,_ = data
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])
        roi_bat = roi_bat.to(params['device'])
        outputs, clusters, _,embedding,_,dis_point = model(input1,input2)
        n_label=64
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
        # calculate new p
        # alpha = dis_point.max()
        _, pred_point = dis_point.min(1)
        _, pred_roi = dis_roi.min(1)
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