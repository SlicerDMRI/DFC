# DFC (Deep Fiber Clustering)
__Deep fiber clustering: Anatomically informed fiber clustering with self-supervised deep learning for fast and effective tractography parcellation__

This code implements a deep learning method for white matter fiber clustering using diffusion MRI data, as described in the following paper:

Chen Y, Zhang C, Xue T, Song Y, Makris N, Rathi Y, Cai W, Zhang F, O'Donnell LJ. Deep Fiber Clustering: Anatomically Informed Fiber Clustering with Self-supervised Deep Learning for Fast and Effective Tractography Parcellation. NeuroImage. 2023 Apr 3:120086.

![fig1](https://user-images.githubusercontent.com/59594831/235310542-1f329a8f-e0f1-448a-ae6f-314be4c9e75e.jpg)

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
python test.py -indir <path of testing data> -modeldir <path of training model>
```
Fast and effective fiber clustering was achieved with the proposed method. Below is a visualization of the obtained clusters.

![fig2](https://user-images.githubusercontent.com/59594831/187730397-26bfb67b-659e-4b47-b78f-8c4a1a711aea.PNG)

The training model and testing dataset are available here: https://github.com/SlicerDMRI/DFC/releases

See our project page https://deepfiberclustering.github.io/ for more details.

## Reference
Chen, Yuqian, Chaoyi Zhang, Tengfei Xue, Yang Song, Nikos Makris, Yogesh Rathi, Weidong Cai, Fan Zhang, and Lauren J. O'Donnell. "DFC: Anatomically Informed Fiber Clustering with Self-supervised Deep Learning for Fast and Effective Tractography Parcellation." arXiv preprint arXiv:2205.00627 (2022).
