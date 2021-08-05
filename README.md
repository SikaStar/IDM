![Python >=3.7](https://img.shields.io/badge/Python->=3.7-blue.svg)
![PyTorch >=1.1](https://img.shields.io/badge/PyTorch->=1.1-yellow.svg)

# Intermediate Domain Module (IDM)

This repository is the official implementation for [IDM: An Intermediate Domain Module for Domain Adaptive Person Re-ID](), which is accepted by [ICCV 2021 (Oral)](http://iccv2021.thecvf.com/node/44). 

`IDM` achieves state-of-the-art performances on the **unsupervised domain adaptation** task for person re-ID.

## Requirements

### Installation

```shell
git clone https://github.com/SikaStar/IDM.git
cd IDM/idm/evaluation_metrics/rank_cylib && make all
```

### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the person re-ID datasets [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [DukeMTMC-ReID](https://arxiv.org/abs/1701.07717), [MSMT17](https://arxiv.org/abs/1711.08565), [PersonX](https://github.com/sxzrt/Instructions-of-the-PersonX-dataset#data-for-visda2020-chanllenge), and
[UnrealPerson](https://github.com/FlyHighest/UnrealPerson).
Then unzip them under the directory like
```
IDM/examples/data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
│   └── MSMT17_V1
├── personx
│   └── PersonX
└── unreal
    ├── list_unreal_train.txt
    └── unreal_vX.Y
```

### Prepare ImageNet Pre-trained Models for IBN-Net
When training with the backbone of [IBN-ResNet](https://arxiv.org/abs/1807.09441), you need to download the ImageNet-pretrained model from this [link](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) and save it under the path of `logs/pretrained/`.
```shell
mkdir logs && cd logs
mkdir pretrained
```
The file tree should be
```
IDM/logs
└── pretrained
    └── resnet50_ibn_a.pth.tar
```
ImageNet-pretrained models for **ResNet-50** will be automatically downloaded in the python script.


## Training

We utilize 4 GTX-2080TI GPUs for training. **Note that**

+ The source and target domains are trained jointly.
+ For baseline methods, use `-a resnet50` for the backbone of ResNet-50, and `-a resnet_ibn50a` for the backbone of IBN-ResNet.
+ For IDM, use `-a resnet50_idm` to insert IDM into the backbone of ResNet-50, and `-a resnet_ibn50a_idm` to insert IDM into the backbone of IBN-ResNet.
+ For strong baseline, use `--use-xbm` to implement [XBM](https://arxiv.org/abs/1912.06798) (a variant of Memory Bank).


### Baseline Methods
To train the baseline methods in the paper, run  commands like:
```shell
# Naive Baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/run_naive_baseline.sh ${source} ${target} ${arch}

# Strong Baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/run_strong_baseline.sh ${source} ${target} ${arch}
```

**Some examples:**
```shell
### market1501 -> dukemtmc ###

# ResNet-50
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/run_strong_baseline.sh market1501 dukemtmc resnet50 

# IBN-ResNet-50
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/run_strong_baseline.sh market1501 dukemtmc resnet_ibn50a
```

### Training with IDM

To train the models with our IDM, run commands like:
```shell
# Naive Baseline + IDM
CUDA_VISIBLE_DEVICES=0,1,2,3 \
sh scripts/run_idm.sh ${source} ${target} ${arch} ${stage} ${mu1} ${mu2} ${mu3}

# Strong Baseline + IDM
CUDA_VISIBLE_DEVICES=0,1,2,3 \
sh scripts/run_idm_xbm.sh ${source} ${target} ${arch} ${stage} ${mu1} ${mu2} ${mu3}
```

+ Defaults: `--stage 0 --mu1 0.7 --mu2 0.1 --mu3 1.0`

**Some examples:**
```shell
### market1501 -> dukemtmc ###

# ResNet-50 + IDM
CUDA_VISIBLE_DEVICES=0,1,2,3 \
sh scripts/run_idm_xbm.sh market1501 dukemtmc resnet50_idm 0 0.7 0.1 1.0 

# IBN-ResNet-50 + IDM
CUDA_VISIBLE_DEVICES=0,1,2,3 \
sh scripts/run_idm_xbm.sh market1501 dukemtmc resnet_ibn50a_idm 0 0.7 0.1 1.0
```

## Evaluation

We utilize 1 GTX-2080TI GPU for testing. **Note that**

+ use `--dsbn` for domain adaptive models, and add `--test-source` if you want to test on the source domain;
+ use `-a resnet50` for the backbone of ResNet-50, and `-a resnet_ibn50a` for the backbone of IBN-ResNet.
+ use `-a resnet50_idm` for ResNet-50 + IDM, and `-a resnet_ibn50a_idm` for IBN-ResNet + IDM.

To evaluate the **baseline model** on the **target-domain** dataset, run:
```shell
CUDA_VISIBLE_DEVICES=0 \
python3 examples/test.py --dsbn -d ${dataset} -a ${arch} --resume ${resume} 
```

To evaluate the **baseline model** on the **source-domain** dataset, run:
```shell
CUDA_VISIBLE_DEVICES=0 \
python3 examples/test.py --dsbn --test-source -d ${dataset} -a ${arch} --resume ${resume} 
```

To evaluate the **IDM model** on the **target-domain** dataset, run:
```shell
CUDA_VISIBLE_DEVICES=0 \
python3 examples/test.py --dsbn-idm -d ${dataset} -a ${arch} --resume ${resume} --stage ${stage} 
```

To evaluate the **IDM model** on the **source-domain** dataset, run:
```shell
CUDA_VISIBLE_DEVICES=0 \
python3 examples/test.py --dsbn-idm --test-source -d ${dataset} -a ${arch} --resume ${resume} --stage ${stage} 
```


**Some examples:**
```shell
### market1501 -> dukemtmc ###

# evaluate the target domain "dukemtmc" on the strong baseline model
CUDA_VISIBLE_DEVICES=0 \
python3 examples/test.py --dsbn  -d dukemtmc -a resnet50 \
--resume logs/resnet50_strong_baseline/market1501-TO-dukemtmc/model_best.pth.tar 

# evaluate the source domain "market1501" on the strong baseline model
CUDA_VISIBLE_DEVICES=0 \
python3 examples/test.py --dsbn --test-source  -d market1501 -a resnet50 \
--resume logs/resnet50_strong_baseline/market1501-TO-dukemtmc/model_best.pth.tar 

# evaluate the target domain "dukemtmc" on the IDM model (after stage-0)
python3 examples/test.py --dsbn-idm  -d dukemtmc -a resnet50_idm \
--resume logs/resnet50_idm_xbm/market1501-TO-dukemtmc/model_best.pth.tar --stage 0

# evaluate the target domain "dukemtmc" on the IDM model (after stage-0)
python3 examples/test.py --dsbn-idm --test-source  -d market1501 -a resnet50_idm \
--resume logs/resnet50_idm_xbm/market1501-TO-dukemtmc/model_best.pth.tar --stage 0

```

## Acknowledgement
Our code is based on [MMT](https://github.com/yxgeee/MMT) and [SpCL](https://github.com/yxgeee/SpCL). Thanks for [Yixiao's](https://geyixiao.com/) wonderful works.

## Citation
If you find our work is useful for your research, please kindly cite our paper
```
@inproceedings{dai2021idm,
  title={IDM: An Intermediate Domain Module for Domain Adaptive Person Re-ID},
  author={Dai, Yongxing and Liu, Jun and Sun, Yifan and Tong, Zekun and Zhang, Chi and Duan, Ling-Yu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```
If you have any questions, please leave an issue or contact me: yongxingdai@pku.edu.cn


