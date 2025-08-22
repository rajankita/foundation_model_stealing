import os
import sys
import random
import numpy as np
from tqdm import tqdm
import torch
import torchattacks
from sklearn.metrics import accuracy_score
from advertorch.attacks import PGDAttack, GradientSignAttack

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.models import resnet34

from train_utils import train_cutmix, train_mixup, train_augmix, testz, train_with_validation, agree, dist
from conf import cfg, load_cfg_fom_args
# from activethief import load_thief_dataset, load_thief_model, create_thief_loaders
# from models.victim import Victim
# from models.cifar10_models import resnet34 as cifar10_resnet34
# import datasets

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
    

def test_adversarial_v1(target_model, thief_model, test_loader, adversary):

    trues = []
    preds_victim_clean = []
    preds_victim = []
    preds_thief = []
    count = 0
    for data in (test_loader):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        
        # create adversarial images
        adv_images = adversary.perturb(inputs, labels)
        
        # evaluate the victim model on clean images
        scores = target_model(inputs)
        _, pred_victim_clean = torch.max(scores.data, 1)
        
        # evaluate thief model on adversarial images
        scores = thief_model(adv_images)
        _, pred_thief = torch.max(scores.data, 1)
        
        # evaluate victim model on adversarial images
        scores = target_model(adv_images)
        _, pred_victim = torch.max(scores.data, 1)

        preds_thief.append(pred_thief.cpu())
        preds_victim_clean.append(pred_victim_clean.cpu())
        preds_victim.append(pred_victim.cpu())
        trues.append(labels.cpu())
        
        count += len(labels.cpu())
        if count >= 1000:
            break


    preds_victim_clean = np.concatenate(preds_victim_clean)
    preds_victim = np.concatenate(preds_victim)
    preds_thief = np.concatenate(preds_thief)
    trues = np.concatenate(trues)
    # print(len(preds_victim))
    
    # acc_thief = accuracy_score(y_true=trues, y_pred=preds_thief)
    # print('Thief model misclassification rate = ', 1. - acc_thief)
    
    # acc_victim = accuracy_score(y_true=trues, y_pred=preds_victim)
    # print('Victim model misclassification rate = ', 1. - acc_victim)
    
    mislabeled_victim = 0
    mislabeled_thief = 0
    total = 0
    for y_gt, y_clean, y_adv, y_thief in zip(trues, preds_victim_clean, preds_victim, preds_thief):
        # if y_clean != y_gt:
        #     continue
        if y_adv != y_gt:
            mislabeled_victim += 1
        if y_thief != y_gt:
            mislabeled_thief += 1
        total += 1
    
    # asr = mislabeled / total
    # print(f"Victim ASR = {mislabeled}/{total} = {asr}")

    asr = mislabeled_victim / mislabeled_thief
    print(f"Victim ASR = {mislabeled_victim}/{mislabeled_thief} = {asr}")


def test_adversarial_v2(target_model, thief_model, test_loader, adversary):
    correct = 0.0
    total = 0.0
    target_model.eval()
    total_L2_distance = 0.0
    
    success, total = 0.0, 0.0
    for data in test_loader:
        inputs = data[0].cuda()
        labels = data[1].cuda()

        # initial prediction on clean test data
        outputs = target_model(inputs)
        _, predicted_clean = torch.max(outputs.data, 1)

        # generate adversarial sample using the thief model
        adv_inputs_ori = adversary.perturb(inputs, labels)

        # prediction of thief model on adv sample
        with torch.no_grad():
            outputs = thief_model(adv_inputs_ori)
            _, pred_thief = torch.max(outputs.data, 1)
        
        # prediction of victim model on adv sample
        with torch.no_grad():
            outputs = target_model(adv_inputs_ori)
            _, pred_victim = torch.max(outputs.data, 1)
        
        # count the images for which both thief and victim predictions are incorrect
        for (gt, pred1, pred2) in zip(labels, pred_thief, pred_victim):
            if pred1 != gt:   
                total += 1 
                if pred2 != gt:
                    success += 1
    
    asr = 100.*success/total
    print(f'Attack success rate = {success}/{total} = {asr}%')
    

def test_adversarial_v3(target_model, thief_model, test_loader, adversary):

    trues = []
    preds_victim_clean = []
    preds_victim = []
    preds_thief_clean = []
    preds_thief = []
    count = 0
    for data in (test_loader):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        
        # create adversarial images
        adv_images = adversary.perturb(inputs, labels)
        
        # evaluate the victim model on clean images
        scores = target_model(inputs)
        _, pred_victim_clean = torch.max(scores.data, 1)

        # evaluate victim model on adversarial images
        scores = target_model(adv_images)
        _, pred_victim = torch.max(scores.data, 1)

        # evaluate the thief model on clean images
        scores = thief_model(inputs)
        _, pred_thief_clean = torch.max(scores.data, 1)
        
        # evaluate thief model on adversarial images
        scores = thief_model(adv_images)
        _, pred_thief = torch.max(scores.data, 1)
            
        preds_victim_clean.append(pred_victim_clean.cpu())
        preds_victim.append(pred_victim.cpu())
        preds_thief_clean.append(pred_thief_clean.cpu())
        preds_thief.append(pred_thief.cpu())
        trues.append(labels.cpu())
        
        count += len(labels.cpu())
        if count >= 1000:
            break

    preds_victim_clean = np.concatenate(preds_victim_clean)
    preds_victim = np.concatenate(preds_victim)
    preds_thief_clean = np.concatenate(preds_thief_clean)
    preds_thief = np.concatenate(preds_thief)
    trues = np.concatenate(trues)

    # Compute thief ASR: proportion of adv samples misclassified by the thief 
    # (out of samples originially classified correctly)
    # mislabeled_thief, total = 0, 0
    # for y_gt, y_thief_clean, y_thief_adv in zip(trues, preds_thief_clean, preds_thief):
    #     if y_thief_clean != y_gt:
    #         continue
    #     if y_thief_adv != y_gt:
    #         mislabeled_thief += 1
    #     total += 1
    # asr_thief = mislabeled_thief / total
    # print(f"Thief ASR = {mislabeled_thief}/{total} = {asr_thief}")

    # Compute transferability: adv samples crafted by the thief that are misclassified by the victim
    # total, success = 0, 0
    # for (gt, pred1, pred2) in zip(trues, preds_thief, preds_victim):
    #         if pred1 != gt:   
    #             total += 1 
    #             if pred2 != gt:
    #                 success += 1        
    # asr = success / total
    # print(f"Transferability = {success}/{total} = {asr}")

    total, success = 0, 0
    for (gt, pred1, pred2) in zip(trues, preds_thief, preds_victim):
            if pred1 == pred2:   
                success += 1 
            total += 1
    asr = success / total
    print(f"Adversarial fidelity = {success}/{total} = {asr}")



if __name__ == "__main__":

    load_cfg_fom_args(description='Model Stealing')

    VICTIM_ARCHS = [
                    # 'resnet18', 'resnet34', 'resnet50', 
                    # 'resnet101', 'vit_s_16_21k', 'vit_b_16_21k', 
                    'vit_l_16_21k'
                    ]
    THIEF_ARCH = 'vit_l_16_21k'

    # CIFAR-10
    VICTIM_MODEL_PATHS = [
        # '/home/ankita/scratch/data_msa_encoders/victims/victims_cifar10_lp/resnet18/checkpoint.pth.tar',
        # '/home/ankita/scratch/data_msa_encoders/victims/victims_cifar10_lp/resnet34/checkpoint.pth.tar',
        # '/home/ankita/scratch/data_msa_encoders/victims/victims_cifar10_lp/resnet50/checkpoint.pth.tar',
        # '/home/ankita/scratch/data_msa_encoders/victims/victims_cifar10_lp/resnet101/checkpoint.pth.tar',
        # '/home/ankita/scratch/data_msa_encoders/victims/victims_cifar10_lp/vit_s_16_21k/checkpoint.pth.tar',
        # '/home/ankita/scratch/data_msa_encoders/victims/victims_cifar10_lp/vit_b_16_21k/checkpoint.pth.tar',
        '/home/ankita/scratch/data_msa_encoders/victims/victims_cifar10_lp/vit_l_16_21k/checkpoint.pth.tar',
    ]

    # Thieves for FFT victims
    THIEF_MODEL_PATHS = [
        # '/home/deepankar/scratch/MSA_results_vision03/CIFAR10_resnet18/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
        # '/home/deepankar/scratch/MSA_results_vision03/CIFAR10_resnet34/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
        # '/home/deepankar/scratch/MSA_results_vision03/CIFAR10_resnet50/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
        # '/home/deepankar/scratch/MSA_results_vision03/CIFAR10_resnet101/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
        # '/home/deepankar/scratch/MSA_results_vision03/CIFAR10_vit_s_16_21k/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
        # '/home/deepankar/scratch/MSA_results_vision03/CIFAR10_vit_b_16_21k/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
        '/home/deepankar/scratch/MSA_results_vision03/CIFAR10_vit_l_16_21k/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth'
    ]

    # # Thieves for LP victims
    # THIEF_MODEL_PATHS = [
    #     '/home/ankita/mnt/vision03_deepankar_ssl/CIFAR10_resnet18/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/CIFAR10_resnet34/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/CIFAR10_resnet50/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/CIFAR10_resnet101/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/CIFAR10_vit_s_16_21k/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/CIFAR10_vit_b_16_21k/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/CIFAR10_vit_l_16_21k/imagenet32_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth'
    # ]

    # # Indoor-67
    # VICTIM_MODEL_PATHS = [
    #     '/home/ankita/mnt/data_msa_encoders/victims/victims_indoor67_lp/resnet18/checkpoint.pth.tar',
    #     '/home/ankita/mnt/data_msa_encoders/victims/victims_indoor67_lp/resnet34/checkpoint.pth.tar',
    #     '/home/ankita/mnt/data_msa_encoders/victims/victims_indoor67_lp/resnet50/checkpoint.pth.tar',
    #     '/home/ankita/mnt/data_msa_encoders/victims/victims_indoor67_lp/resnet101/checkpoint.pth.tar',
    #     '/home/ankita/mnt/data_msa_encoders/victims/victims_indoor67_lp/vit_s_16_21k/checkpoint.pth.tar',
    #     '/home/ankita/mnt/data_msa_encoders/victims/victims_indoor67_lp/vit_b_16_21k/checkpoint.pth.tar',
    #     '/home/ankita/mnt/data_msa_encoders/victims/victims_indoor67_lp/vit_l_16_21k/checkpoint.pth.tar',
    # ]

    # # Thieves for FFT victims
    # THIEF_MODEL_PATHS = [
    #     '/home/ankita/mnt/vision03_deepankar_ssl/Indoor67_resnet18/imagenet_full_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/Indoor67_resnet34/imagenet_full_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/Indoor67_resnet50/imagenet_full_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/Indoor67_resnet101/imagenet_full_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/Indoor67_vit_s_16_21k/imagenet_full_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/Indoor67_vit_b_16_21k/imagenet_full_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth',
    #     '/home/ankita/mnt/vision03_deepankar_ssl/Indoor67_vit_l_16_21k/imagenet_full_vit_l_16_21k/SGD/5000_val500/random_v1/trial_1_cycle_1_best.pth'
    # ]

    
    # Load victim dataset (test split only)
    testset, victim_normalization_transform, n_classes = load_victim_dataset(cfg, cfg.VICTIM.DATASET, cfg.VICTIM.ARCH, 224)
    test_loader = DataLoader(testset, batch_size=64, num_workers=4, shuffle=False, pin_memory=False)  
    print(f"Loaded target dataset of size {len(testset)} with {n_classes} classes")

    # Load victim and thief models
    for (victim_arch, victim_path, thief_path) in zip(VICTIM_ARCHS, VICTIM_MODEL_PATHS, THIEF_MODEL_PATHS):
        print('\n\n')
        target_model = load_victim_model(victim_arch, victim_path, victim_normalization_transform, n_classes)

        # Evaluate target model on target dataset: sanity check
        acc, f1 = testz(target_model, test_loader)
        print(f"Target model acc = {acc}")

        # Load thief model            
        thief_model = load_thief_model(cfg, THIEF_ARCH, n_classes, cfg.ACTIVE.PRETRAINED_PATH)
        thief_model=Thief(thief_model,cfg.THIEF.IMG_SIZE)
        thief_model.load_state_dict(torch.load(thief_path)['state_dict'])

        thief_model = thief_model.cuda()
        # print('\n'+thief_path)
        
        # Evaluate thief model on target dataset
        acc, f1 = testz(thief_model, test_loader)
        print(f'Thief Acc = {acc}')

    # sys.exit(1)
    
        targeted = False
        # for attack in ['FGSM', 'PGD']:
        for attack in ['FGSM']:
            # for eps in [4/255, 8/255]:
            for eps in [8/255]:
                print(f'\n {attack}, targeted={targeted}, eps={eps}:')
                         
                # Attack
                thief_model.eval()

                # # PGD attack
                if attack == 'PGD':
                    attack_iters = 10
                    adversary = PGDAttack(
                        thief_model,
                        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                        eps=eps,
                        nb_iter=attack_iters, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
                        # nb_iter=attack_iters, eps_iter=0.1 * (8 / 255), clip_min=0.0, clip_max=1.0,
                        targeted=targeted)
                    
                elif attack == 'FGSM':
                    # FGSM attack
                    adversary = GradientSignAttack(
                                    thief_model,
                                    loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                    eps=eps,
                                    targeted=targeted)
                    
                test_adversarial_v3(target_model, thief_model, test_loader, adversary)
                    
                