# CDNNet

## 1. Introduction

We propose a method CDN-Net for single frame dim space object detection. The contribution of this paper are as follows:

1. We propose a soft segmentation supervised training method based on densely nested U-Net networks for dim target detection. The method has been optimized for the weak and small characteristics of the target.
2. Provide open-source celestial target simulated and real dataset, including FITS format image and accurate target location annotation.
3. The method outperforms traditional methods and existing deep learning methods in both simulation and real datasets.



## 2. Structure

The project structure are as follow:

* dataset: the code for building the dataset and the dataset results.
* eval: evaluation code.
* model: algorithm source code.
* result: weight parameter file of network training results.
* outfiles: output files result directory.



## 3. Prerequisite

* Tested on Ubuntu 16.04, with Python 3.7, PyTorch 1.7, Torchvision 0.8.1, CUDA 11.1, and 1x NVIDIA 3090 and also 
* Tested on Windows 10  , with Python 3.6, PyTorch 1.1, Torchvision 0.3.0, CUDA 10.0, and 1x NVIDIA 1080Ti.



## 4. Usage

#### 4.1 Train.

```bash
python train.py --model cdnnet --backbone resnet_18 --netdepth 4 --in_channels 3 --dataset fits_softCircle_modified --preprocess hierarchy --mask_type soft --split_method 2304_256 --base_size 256 --crop_size 256 --epochs 100 --train_batch_size 16 --test_batch_size 16 --loss_func SIoU+SFL --deep_supervision True
```

* for simulated image, set --dataset with fits_starSimu; for real image, set --dataset with fits_softCircle_modified.
* for resume training, set --resume_from with resume model directory.



#### 4.2. Test.

```bash
python infer.py --model cdnnet --backbone resnet_18 --netdepth 4 --in_channels 3 --dataset fits_softCircle_modified --preprocess hierarchy --split_method 2304_256 --base_size 256 --crop_size 256 --st_model ./result/modelPathDir --model_dir ./result/modelPathDir/modelPath.pth.tar
```




*This code is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.



## 6. References

1. Li B, Xiao C, Wang L, et al. Dense nested attention network for infrared small target detection[J]. IEEE Transactions on Image Processing, 2022, 32: 1745-1758. [[code]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) 

