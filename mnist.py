from __future__ import print_function
import torch.utils.data as data
import numpy as np
import torch
from utils.tract_feat import _feat_to_3D
from utils.fiber_distance import fiber_pair_similarity

class Fiber_pair(data.Dataset):
    def __init__(self,vec,roi,surf,transform=None):
        self.vec=vec #8000*14*3
        self.roi=roi
        self.transform = transform
        self.surf=surf

    def __getitem__(self, index: int):
        index1=index
        index2=np.random.randint(0,len(self.vec))
        fiber1=self.vec[index1]
        fiber2 = self.vec[index2]
        roi1=self.roi[index1]
        surf1=self.surf[index1]
        similarity=fiber_pair_similarity(fiber1, fiber2)
        fiber1=torch.tensor(fiber1.T, dtype=torch.float)
        fiber2 = torch.tensor(fiber2.T, dtype=torch.float)
        similarity = torch.tensor(similarity, dtype=torch.float)
        return fiber1,fiber2,similarity,roi1,surf1,index

    def __len__(self) -> int:
        return len(self.vec)

class FiberMap_pair(data.Dataset):
    def __init__(self,vec,roi,surf,transform=None):
        #vec=np.reshape(vec,(len(ds),-1,ds.shape[1]))
        self.vec=vec #8000*14*3
        self.roi=roi
        self.transform = transform
        self.surf = surf

    def __getitem__(self, index: int):
        index1=index
        index2=np.random.randint(0,len(self.vec))
        fiber1=self.vec[index1]
        fiber2 = self.vec[index2]
        img1= _feat_to_3D(np.expand_dims(fiber1,0), repeat_time=14).squeeze()
        img1=img1.transpose(2,0,1)
        img2= _feat_to_3D(np.expand_dims(fiber2,0), repeat_time=14).squeeze()
        img2=img2.transpose(2,0,1)
        roi1=self.roi[index1]
        surf1 = self.surf[index1]
        similarity=fiber_pair_similarity(fiber1, fiber2)
        img1=torch.tensor(img1, dtype=torch.float)
        img2 = torch.tensor(img2, dtype=torch.float)
        similarity = torch.tensor(similarity, dtype=torch.float)
        return img1,img2,similarity,roi1,surf1,index

    def __len__(self) -> int:
        return len(self.vec)

