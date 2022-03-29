# DFC (Deep Fiber Clustering)
__DFC: Anatomically Informed Unsupervised Deep Learning for Fast and Effective White Matter Fiber Clustering__

This code implements a deep learning method for white matter fiber clustering using diffusion MRI data, as described in the following paper:

Yuqian Chen, Chaoyi Zhang, Yang Song, Tengfei Xue, Nikos Makris, Yogesh Rathi, Weidong Cai, Fan Zhang, Lauren J. O’Donnell.
DFC: Anatomically Informed Unsupervised Deep Learning for Fast and Effective White Matter Fiber Clustering. (In submission soon)

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
python test.py -indir <path of testing data>
```
Effective fiber clustering was achieved with the proposed method. Below is a visualization of the obtained clusters.

![images](https://github.com/SlicerDMRI/DFC/blob/master/images/visualization%20of%20clusters.png)
