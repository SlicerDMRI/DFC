import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy

# Clustering layer definition (see DCEC article for equations)
class ClusterlingLayer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x_dis = x
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x,x_dis

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
def get_graph_feature(x, k=20, idx=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    # print(idx.shape)
    # print(idx_base.shape)
    if idx.shape[0]>idx_base.shape[0]:
        idx=idx[idx_base.shape[0]]
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature
class DGCNN(nn.Module):
    # from DGCNN's repo
    def __init__(self, k=5, input_channel=3,num_clusters=800,features_len=1024,embedding_dimension=10,bias=True,idx=None):
        super(DGCNN, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(features_len)

        self.conv1 = nn.Sequential(nn.Conv2d(input_channel*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, features_len, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(features_len * 2, 256, bias=bias)
        self.linear2 = nn.Linear(256, 64, bias=bias)
        self.linear3 = nn.Linear(64, embedding_dimension, bias=bias)
        #self.embedding = nn.Linear(features_len*2, embedding_dimension, bias=bias)
        self.clustering = ClusterlingLayer(embedding_dimension, num_clusters)
        self.num_clusters = num_clusters
        self.idx=idx
        if self.idx is None:
            print('original idx')
        else:
            print('redefined idx')
        print(k)

    def forward(self, x1,x2):
        def forward_single(self, x):
            batch_size = x.size(0)
            x = get_graph_feature(x, k=self.k,idx=self.idx)
            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            x = get_graph_feature(x1, k=self.k,idx=self.idx)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            x = get_graph_feature(x2, k=self.k,idx=self.idx)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]

            x = get_graph_feature(x3, k=self.k,idx=self.idx)
            x = self.conv4(x)
            x4 = x.max(dim=-1, keepdim=False)[0]

            x = torch.cat((x1, x2, x3, x4), dim=1)

            x = self.conv5(x)
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            x = torch.cat((x1, x2), 1)
            x=self.linear1(x)
            x = self.linear2(x)
            extra_out = self.linear3(x)
            #extra_out = self.embedding(x)
            #print(x.size())
            clustering_out, x_dis = self.clustering(extra_out)
            return clustering_out, extra_out, x_dis
        clustering_out1, extra_out1,x_dis1= forward_single(self,x1)
        clustering_out2, extra_out2,x_dis2 = forward_single(self,x2)
        sim_score=nn.functional.pairwise_distance(extra_out1, extra_out2, p=2)
        return sim_score, clustering_out1, clustering_out2, extra_out1,extra_out2,x_dis1
