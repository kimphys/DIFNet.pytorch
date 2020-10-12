# DIFNet

## Introduction

This is a PyTorch implementation of [Unsupervised Deep Image Fusion With Structure Tensor Representations](https://ieeexplore.ieee.org/document/8962327) with multi-GPU training supports.

## Prerequisites

* PyTorch
* tqdm
* pillow

## How to train

* For training, you should get image pairs such as RGB-thermal, RGB-NIR or multi-focus pairs. And they must be registered each other.
* Set your training phase at args.py
* For multi-GPU training, you should set parameters as follows,
```
### args.py
# For GPU training
world_size = -1
rank = -1
dist_backend = 'nccl'
gpu = 0,1,2,3
multiprocessing_distributed = True
distributed = None
```
You can see details of these parameters at [tutorials of PyTorch official documents](https://pytorch.org/tutorials/intermediate/dist_tuto.html#distributed-training). 
* If you have pretrained models, you can transfer the training to them.
```
### args.py
# resume = "./models/samples.pt" # Transfer learning
resume = None # Train from scratch
```
* Make two txt files which contain paths of RGB and NIR/thermal images, respectively. For example,
```
# RGB images
/home/kim/images_visible/1.jpg
/home/kim/images_visible/2.jpg
/home/kim/images_visible/3.jpg
```
```
# Thermal images
/home/kim/images_thermal/1.jpg
/home/kim/images_thermal/2.jpg
/home/kim/images_thermal/3.jpg
```
* Run the command below
```
python train.py
```

## How to test

* Please add the path of datasets for test.
```
### args.py
# For testing
test_save_dir = "./"
test_visible = "./test_visible.txt"
test_lwir = "./test_lwir.txt"
```
* For training and testing, I recommand [KAIST Multispectral Pedestrian Detection Benchmark](https://soonminhwang.github.io/rgbt-ped-detection/).

## To-Do list
* [x] Add evaluation code
* [ ] Upload pretrained models and samples
* [ ] Update benchmark

## Reference

```
@ARTICLE{8962327,  
         author={H. {Jung} and Y. {Kim} and H. {Jang} and N. {Ha} and K. {Sohn}},  
         journal={IEEE Transactions on Image Processing},   
         title={Unsupervised Deep Image Fusion With Structure Tensor Representations},  
         year={2020},  
         volume={29},  
         number={},  
         pages={3845-3858},}
```
