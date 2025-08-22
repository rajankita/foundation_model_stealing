# Examining the Threat Landscape: Foundation Models and Model Stealing

This is the official PyTorch implementation for the paper "Examining the Threat Landscape: Foundation Models and Model Stealing" accepted in BMVC 2024.

## Abstract

Foundation models (FMs) for computer vision learn rich and robust representations, enabling their adaptation to task/domain-specific deployments with little to no fine-tuning. However, we posit that the very same strength can make applications based on FMs vulnerable to model stealing attacks. Through empirical analysis, we reveal that models fine-tuned from FMs harbor heightened susceptibility to model stealing, compared to conventional vision architectures like ResNets. We hypothesize that this behavior is due to the comprehensive encoding of visual patterns and features learned by FMs during pre-training, which are accessible to both the attacker and the victim. We report that an attacker is able to obtain 94.28% agreement (matched predictions with victim) for a Vision Transformer based victim model (ViT-L/16) trained on CIFAR-10 dataset, compared to only 73.20% agreement for a ResNet-18 victim, when using ViT-L/16 as the thief model. We arguably show, for the first time, that utilizing FMs for downstream tasks may not be the best choice for deployment in commercial APIs due to their susceptibility to model theft. We thereby alert model owners towards the associated security risks, and highlight the need for robust security measures to safeguard such models against theft.

## Installation

Install the environment via:
```
conda env create -n environment.yaml
```

### Datasets
We use five datasets to perform all the experiments in the paper:
* Victim datasets
   * CIFAR-10. 
   * Caltech256 ([Link](http://www.vision.caltech.edu/Image_Datasets/Caltech256/). Images in `data/256_ObjectCategories/<classname>/*.jpg`)
   * Indoor Scenes ([Link](http://web.mit.edu/torralba/www/indoor.html). Images in `data/indoor/Images/<classname>/*.jpg`)
 * Thief datasets
   * ImageNet ILSVRC 2012 ([Link](http://image-net.org/download-images). Images in `data/ILSVRC2012/training_imgs/<classname>/*.jpg`)
   * ImageNet-32. A downsampled version of ImageNet ILSVRC 2012, where images are resized to 32x32. ([Link](https://patrykchrabaszcz.github.io/Imagenet32/). Images in `data/ImageNet32/train/`)


## Train victim models

We can train victim models either by linear probing (LP) or full-fine-tuning (FFT) from pretrained models. To perform linear probing, use the argument --lp. The default setting is full fine tuning.  Below are some sample scripts for training victim models using LP. 

Note: While the paper mentions results on 9 victim models (seven in main paper, two more in supplementary), the code provides support for training several additional architectures.

### CIFAR-10

ResNet-18 pretrained on Imagenet 1k (use similar settings for ResNet-34, ResNet-50)

```
python train_victim.py --data CIFAR10 --arch resnet34 --save-dir victims/victims_cifar10_lp/resnet34 --batch-size 64 --epochs 100 --lr-step 30 --lr 0.002 --momentum 0.9 --img_size 224 --lp
```

ViT B/16 pretrained on ImageNet 21k (use similar settings for ViT-S/16, ViT-L/16)
```
python train_victim.py --data CIFAR10 --arch vit_b_16_21k --save-dir victims/victims_cifar10_lp/vit_b_16_21k --batch-size 64 --epochs 100 --lr-step 30 --lr 0.002 --momentum 0.9 --img_size 224 --lp

```
ViT-B/16 CLIP pretrained on LAION-2B, and finetuned on ImageNet-1K (use input image size 384)
```
python train_victim.py --data CIFAR10 --arch vit_b_16_clip --save-dir victims/victims_cifar10_lp/vit_b_16_clip --batch-size 64 --epochs 100 --lr-step 30 --lr 0.002 --momentum 0.9 --img_size 384 --lp
```

### Indoor-67 
ViT-B/16 pretrained on Imagenet 21K
```
python train_victim.py --data Indoor67 --data_root <path to dataset> --arch vit_b_16_21k --save-dir victims/victims_indoor67_lp/vit_b_16_21k --batch-size 64 --epochs 100 --img_size 224 --lr 0.002 --wd 0 --lr-step 30 --n-classes 67 --lp
```

### Caltech-256
ViT-B/16 pretrained on ImageNet 21K (set the data root accordingly)
```
python train_victim.py --data Caltech256 --data_root <path to dataset> --arch vit_b_16_21k --save-dir victims/victims_caltech256_lp/vit_b_16_21k --batch-size 64 --epochs 100 --img_size 224 --lr 0.002 --lr-step 30 --n-classes 256 --lp
```


## Train thief models

Run `train_thief.py`, and specify all training parameters in the config file. Do not forget to set mention whether the victim and the thief models are LP or FFT. We have provided a few sample configs in the `cfgs` directory, where the name of the config file specifies the victim model, by convention. Adjust all other parameters in the config file as required. 
```
python train_thief.py --cfg cfgs/cifar10/resnet18.yaml
```

## Visualization
For tSNE plots, use `plotting/tsne_visualization.py` and `plotting/tsne_visualization_vit.py` for ResNets and ViTs, respectively.


