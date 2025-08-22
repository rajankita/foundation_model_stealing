import os
import sys
import random
import math
import csv
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision.models import resnet34#, ResNet34_Weights
from sampler import SubsetSequentialSampler

from models.victim import Victim
from models.Simodel import *
from models.cifar10_models import resnet34 as cifar10_resnet34
import utils
from train_utils import agree
from conf import cfg, load_cfg_fom_args
import datasets
# import tent, norm

import utils
from train_utils import train_cutmix, train_mixup, train_augmix, testz, train_with_validation, train_with_kd, agree, dist, FocalLoss
from conf import cfg, load_cfg_fom_args
from loader_utils import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Thief(nn.Module):
    def __init__(self, model,input_size):
        super(Thief,self).__init__()
        self.model=model
        self.input_size=input_size
    def forward(self,x):
        x=torch.nn.functional.interpolate(x,size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        out=self.model(x)
        return out

if __name__ == "__main__":

    load_cfg_fom_args(description='Model Stealing')
    thief_model_dir = '/home/deepankar_v/scratch/MSA_results/CIFAR10_resnet18/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1'

    trial = 1
    cycle = 1
    
    if cfg.THIEF.DATASET != 'imagenet32':
        torch.multiprocessing.set_start_method("spawn")
    
    # Load victim dataset (test split only)
    testset, victim_normalization_transform, n_classes = load_victim_dataset(cfg, cfg.VICTIM.DATASET, cfg.VICTIM.ARCH, cfg.VICTIM.IMG_SIZE)
    test_loader = DataLoader(testset, batch_size=128, num_workers=8, shuffle=False, pin_memory=False)  
    print(f"Loaded target dataset of size {len(testset)} with {n_classes} classes")

    # Load victim model    
    target_model = load_victim_model(cfg.VICTIM.ARCH, cfg.VICTIM.PATH, cfg.VICTIM.LP, victim_normalization_transform, n_classes)

    # Evaluate target model on target dataset: sanity check
    target_model.eval()
    acc, f1 = testz(target_model, test_loader)
    print(f"Target model acc = {acc},f1 = {f1}")

    sys.exit(0)

    # Load trained thief model
    thief_model_path = os.path.join(thief_model_dir, f'trial_{trial}_cycle_{cycle}_best.pth')
    thief_model = load_thief_model(cfg, cfg.THIEF.ARCH, n_classes, cfg.ACTIVE.PRETRAINED_PATH)
    thief_model=Thief(thief_model,cfg.THIEF.IMG_SIZE)
    print('model dict: ', len(thief_model.state_dict().keys()))
    print('checkpoint dict: ', len(torch.load(thief_model_path)['state_dict'].keys()))
    # thief_model=torch.nn.DataParallel(thief_model)
    thief_model.load_state_dict(torch.load(thief_model_path)['state_dict'])

    thief_model = thief_model.cuda()

    print(f"Loaded thief model {thief_model_path}")

    # Compute accuracy and agreement on test dataset
    print('Thief model')
    thief_model.eval()
    acc, f1 = testz(thief_model, test_loader)
    agr = agree(target_model, thief_model, test_loader)
    print(f'Acc = {acc}, F1 Score ={f1}, Agr = {agr}')


    thief_model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = thief_model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')

    # Show the plot
    plt.show()