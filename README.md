# DFC (Deep Fiber Clustering)
__Deep Fiber Clustering: Anatomically Informed Unsupervised Deep Learning for Fast and Effective White Matter Parcellation__

This code implements a deep learning method for white matter fiber clustering using diffusion MRI data, as described in the following paper:

Yuqian Chen, Chaoyi Zhang, Yang Song, Nikos Makris, Yogesh Rathi, Weidong Cai, Fan Zhang, Lauren J. O’Donnell.
Deep Fiber Clustering: Anatomically Informed Unsupervised Deep Learning for Fast and Effective White Matter Parcellation. (MICCAI 2021, travel award)

![figure1](https://user-images.githubusercontent.com/59594831/160573732-fa1881b2-780d-41bb-86b6-c4eb6cae801d.png)

## Installation
The code has been tested with Python 3.7, Pytorch 1.7.1, CUDA 10.1 on Ubuntu 18.04.  
whitematteranalysis  
scikit-learn

## Usage
To train a model for fiber clustering with tractography data:
```
python train.py -indir <path of training data>
```

To evaluate the model with testing data:
```
python test.py -indir <path of testing data> -modeldir <path of trainign model>
```
Fast and effective fiber clustering was achieved with the proposed method. Below is a visualization of the obtained clusters.

![images](https://github.com/SlicerDMRI/DFC/blob/master/images/visualization%20of%20clusters.png)

The training model and testing dataset are available here: https://github.com/SlicerDMRI/DFC/releases

See our project page https://deepfiberclustering.github.io/ for more details.

## Reference
Chen, Yuqian, Chaoyi Zhang, Yang Song, Nikos Makris, Yogesh Rathi, Weidong Cai, Fan Zhang, and Lauren J. O’Donnell. ["Deep Fiber Clustering: Anatomically Informed Unsupervised Deep Learning for Fast and Effective White Matter Parcellation."](https://link.springer.com/chapter/10.1007/978-3-030-87234-2_47) International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2021.
