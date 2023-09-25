## The code has been sorted out. Once the paper is accepted, it will be uploaded immediately



# RAMIS: Increasing Robustness and Accuracy in Medical Image Segmentation with Hybrid CNN-Transformer Synergy

This is the offical repo of RAMIS
[[`Paper`](https://github.com/TianFangzheng/RAMIS)]
[[`Project page`](https://ramis.netlify.app)]


## Environment

```shell
Ubuntu: 20.04
Python: 3.8
CUDA: 11.3
Pytorch: 1.12.1
```

## Installation

### 1. Clone code

```shell
git clone https://github.com/TianFangzheng/RAMIS.git
cd RAMIS
```

### 2. Create a conda environment for this repo

```shell
conda create -n ramis python=3.8
conda activate ramis
```

### 3. Install PyTorch >= 1.12.1

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

### 4. Install other dependency python packages

```shell
pip install -r requirements.txt
```

### 5. Your directory tree should look like this

```
${ROOT}
├── code
├── datasets
├── model
├── readme.md
└── requirements.txt
```

### 6. Prepare dataset

Download [ISIC](https://challenge.isic-archive.com/data/), [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) and [Drive, Chase] from website.  Your datasets tree should look like this.

```
${ROOT/datasets}
|── train_set
│   └── rgb
|   |   └──isic2017
|   |   |   └──isic_training(1).png
|   |   |   └──isic_training(2).png
|   |   |   └──  ......
|   |   └──isic2018
|   |   |   └──isic_training(1).png
|   |   |   └──isic_training(2).png
|   |   |   └──  ......
|   |   └──drive
|   |   |   └──drive_training(1).png
|   |   |   └──drive_training(2).png
|   |   |   └──  ......
|   |   └──chase
|   |   |   └──chase_training(1).png
|   |   |   └──chase_training(2).png
|   |   |   └──  ......
|   |   └──busi
|   |   |   └──busi_training(1).png
|   |   |   └──busi_training(2).png
|   |   |   └──  ......
|   └── gt
|   |   └──isic2017
|   |   |   └──isic_training(1).png
|   |   |   └──isic_training(2).png
|   |   |   └──  ......
|   |   └──isic2018
|   |   |   └──isic_training(1).png
|   |   |   └──isic_training(2).png
|   |   |   └──  ......
|   |   └──drive
|   |   |   └──drive_training(1).png
|   |   |   └──drive_training(2).png
|   |   |   └──  ......
|   |   └──chase
|   |   |   └──chase_training(1).png
|   |   |   └──chase_training(2).png
|   |   |   └──  ......
|   |   └──busi
|   |   |   └──busi_training(1).png
|   |   |   └──busi_training(2).png
|   |   |   └──  ......
├── test_set
│   └── rgb
|   |   └──isic2017
|   |   |   └──isic_test(1).png
|   |   |   └──isic_test(2).png
|   |   |   └──  ......
|   |   └──isic2018
|   |   |   └──isic_test(1).png
|   |   |   └──isic_test(2).png
|   |   |   └──  ......
|   |   └──drive
|   |   |   └──drive_test(1).png
|   |   |   └──drive_test(2).png
|   |   |   └──  ......
|   |   └──chase
|   |   |   └──chase_test(1).png
|   |   |   └──chase_test(2).png
|   |   |   └──  ......
|   |   └──busi
|   |   |   └──busi_test(1).png
|   |   |   └──busi_test(2).png
|   |   |   └──  ......
|   └── gt
|   |   └──isic2017
|   |   |   └──isic_test(1).png
|   |   |   └──isic_test(2).png
|   |   |   └──  ......
|   |   └──isic2018
|   |   |   └──isic_test(1).png
|   |   |   └──isic_test(2).png
|   |   |   └──  ......
|   |   └──drive
|   |   |   └──drive_test(1).png
|   |   |   └──drive_test(2).png
|   |   |   └──  ......
|   |   └──chase
|   |   |   └──chase_test(1).png
|   |   |   └──chase_test(2).png
|   |   |   └──  ......
|   |   └──busi
|   |   |   └──busi_test(1).png
|   |   |   └──busi_test(2).png
|   |   |   └──  ......
```

## Quick start

### 1. Download trained model

1. Download pretrained models from ([here](https://drive.google.com/drive/folders/13x7Ta8yyiKgtPqGcVpNlJblVTsDQZ0BZ?usp=drive_link)) and make models directory look like this:

```
${ROOT/model}
|── model_self_distillation.pth
├── model_implicit_neural.pth
├── model_chaset.pt
└── model_busi.pt
```

### 2. Training

```
cd code
python main.py --train_set /path/to/train_set/directory --test_set /path/to/test_set/directory
```

Take the training Drive data set as an example

```
python main.py --train_set "../datasets/train_set/rgb/drive/*" --test_set "../datasets/test_set/rgb/drive/*"
```

### 3. Resume training

```
python main.py --train_set /path/to/train_set/directory --test_set /path/to/test_set/directory --load  /path/to/model/directory
```

Take the training Drive data set as an example

```
python main.py --train_set "../datasets/train_set/rgb/drive/*" --test_set "../datasets/test_set/rgb/drive/*" --load  "../model/2023-09-25-10:50:04/"
```

### 4. Testing

```
python main.py --train_set /path/to/train_set/directory --test_set /path/to/test_set/directory --load  /path/to/model/directory --test_only
```

Take the training Drive data set as an example

```
python main.py --train_set "../datasets/train_set/rgb/drive/*" --test_set "../datasets/test_set/rgb/drive/*" --load  "../model/2023-09-25-10:50:04/" --test_only
```



## Citation

https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation

https://github.com/facebookresearch/detr/blob/master/models/transformer.py

https://github.com/yinboc/liif

https://github.com/facebookresearch/dino
