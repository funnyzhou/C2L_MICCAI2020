# C2L_MICCAI2020
This is a repository for paper "Comparing to Learn: Surpassing ImageNet Pretraining on Radiographs By Comparing Image Representations" early accepted by MICCAI 2020

### Introduction

The goal of C2L is to provide an effective pretraining method by using 2D radiographs only. It is designed to be flexible in order to support rapid implementation. Specifically, you can run these experiments by simply configuring the dataset path.

## Citation

```
@inproceedings{zhou2020C2L,
  title={Comparing to Learn: Surpassing ImageNet Pretraining on Radiographs By Comparing Image Representations},
  author={Zhou, Hong-Yu and Yu, Shuang and Bian, Cheng and Hu, Yifan and Ma, Kai and Zheng, Yefeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2020},
  organization={Springer}
}
```

## Installation

We will demonstrate how to use C2L to train ResNet-18 and DenseNet-121.

### Dependency

Please install PyTorch (1.1 or 1.4) before you run the code. We strongly recommend you to install Anaconda3 where we use Python 3.6.

### Step 0

    git clone https://github.com/funnyzhou/C2L_MICCAI2020.git
    cd C2L_MICCAI2020
### Step 1

Please configure `opt.data_folder`, `opt.model_path`, `opt.tb_path` and `data_folder` in `train_C2L_res18.py` and `train_C2L_densenet121.py`, respectively. 

### Step 2

Replace `pretrained_datasets/file_names.txt` with your own data paths.

### Step 3
*The proposed C2L method mainly lies in `train_C2L` function. You can find it in both `train_C2L_res18.py` and `train_C2L_dense121.py`.*

To train ResNet18:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_C2L_res18.py --model resnet18 --batch_size 128 --num_workers 24
```

To train DenseNet121:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_C2L_dense121.py --model densenet121 --batch_size 128 --num_workers 24
```

## Model Weight

We also provide pretrained weights of [ResNet-18](https://drive.google.com/drive/folders/1qZkzBmv6LMAe0DHB0jKsM8fqcVdV1bdb?usp=sharing) using C2L.

## Acknowledgements

Part of this code is based on [CMC](https://github.com/HobbitLong/CMC).

