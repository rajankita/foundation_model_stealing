import os
import random
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.optim.lr_scheduler as lr_scheduler

import utils
from train_utils import testz, train_with_validation, agree, dist
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
    
    if (cfg.THIEF.DATASET != 'imagenet32') and (cfg.THIEF.DATASET != 'imagenet32_soft'):
        torch.multiprocessing.set_start_method("spawn")
    
    # Load victim dataset (test split only)
    testset, victim_normalization_transform, n_classes = load_victim_dataset(cfg, cfg.VICTIM.DATASET, cfg.VICTIM.ARCH, cfg.VICTIM.IMG_SIZE)
    test_loader = DataLoader(testset, batch_size=128, num_workers=8, shuffle=False, pin_memory=False)  
    print(f"Loaded target dataset of size {len(testset)} with {n_classes} classes")

    # Load victim model  
    target_model = load_victim_model(cfg.VICTIM.ARCH, cfg.VICTIM.PATH, cfg.VICTIM.LP, victim_normalization_transform, n_classes)

    # Evaluate target model on target dataset: sanity check
    acc, f1 = testz(target_model, test_loader)
    print(f"Target model acc = {acc}")
    
    # Begin trials
    li = []
    results_arr = []
    uncertainty_arr = []
    for trial in range(cfg.TRIALS):

        # Load thief dataset ~ with and without augmentation (teacher and student versions)
        imagenet_id_labels_file = None
        thief_data, thief_data_aug = load_thief_dataset(cfg, cfg.THIEF.DATASET, cfg.THIEF.DATA_ROOT, target_model, cfg.VICTIM.DATASET, cfg.VICTIM.IMG_SIZE, id_labels_file=imagenet_id_labels_file)
        
        # Setup validation, initial labeled, and unlabeled sets
        indices = list(range(min(cfg.THIEF.NUM_TRAIN, len(thief_data))))
        random.shuffle(indices)
        # do not use the entire unlabeled set, use only SUBSET number of samples
        indices = indices[:cfg.THIEF.SUBSET]
        val_set = indices[:cfg.ACTIVE.VAL]
        labeled_set = indices[cfg.ACTIVE.VAL:cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL]
        unlabeled_set = indices[cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL:]
        
        # Create train, val and unlabeled dataloaders
        train_loader, val_loader, unlabeled_loader = create_thief_loaders(thief_data, thief_data_aug, labeled_set, 
                                                                        val_set, unlabeled_set, cfg.TRAIN.BATCH, 
                                                                        target_model)
        dataloaders  = {'train': train_loader, 'test': test_loader, 'val': val_loader, 'unlabeled': unlabeled_loader}
        
        for cycle in range(cfg.ACTIVE.CYCLES):
            
            # Load thief model            
            thief_model = load_thief_model(cfg, cfg.THIEF.ARCH, n_classes, cfg.ACTIVE.PRETRAINED_PATH, cfg.THIEF.LP)
            thief_model=Thief(thief_model,cfg.THIEF.IMG_SIZE)
            print("Thief model initialized successfully")
            total_params = sum(p.numel() for p in thief_model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {total_params}")
            # print(summary(thief_model, (3,224,224)))

            # Compute metrics on target dataset
            # acc, f1 = testz(thief_model, test_loader)
            # agr = agree(target_model, thief_model, test_loader)
            # print(f'Initial model on target dataset: acc = {acc:.4f}, agreement = {agr:.4f}, f1 = {f1:.4f}')
            
            # Compute metrics on validation dataset
            # acc, f1 = testz(thief_model, dataloaders['val'])
            # agr = agree(target_model, thief_model, dataloaders['val'])
            # print(f'Initial model on validation dataset: acc = {acc:.4f}, agreement = {agr:.4f}, f1 = {f1:.4f}')

            # Set up thief optimizer, scheduler
            criterion = nn.CrossEntropyLoss(reduction='none')
            if cfg.TRAIN.OPTIMIZER == 'SGD':
                non_frozen_parameters = [p for p in thief_model.parameters() if p.requires_grad]
                optimizer = optim.SGD(non_frozen_parameters, lr=cfg.TRAIN.LR, 
                                        momentum=cfg.TRAIN.MOMENTUM,weight_decay=cfg.TRAIN.WDECAY)
            elif cfg.TRAIN.OPTIMIZER == 'Adam':
                optimizer = optim.Adam(thief_model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WDECAY)
            else:
                raise AssertionError('Unknown optimizer')
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA)

            # Logit adjustment
            if cfg.ACTIVE.LA is True:
                logit_adjustments = utils.compute_adjustment(dataloaders['train'], tro=cfg.ACTIVE.TRO, num_classes=n_classes)
            else:
                logit_adjustments = None
            print('la: ', logit_adjustments)

            # Train thief model for current cycle
            train_with_validation(thief_model, criterion, optimizer, scheduler, dataloaders, 
                                cfg.TRAIN.EPOCH, trial, cycle, cfg.SAVE_DIR,
                                la=logit_adjustments)

            # Compute accuracy and agreement on target dataset
            acc,f1 = testz(thief_model, test_loader)
            agr = agree(target_model, thief_model, test_loader)
            print('Acc, agreement for latest model: ', acc, agr)

            # Load checkpoint for best model
            print("Load best checkpoint for thief model")
            best_model_path = os.path.join(cfg.SAVE_DIR, f'trial_{trial+1}_cycle_{cycle+1}_best.pth')
            best_state = torch.load(best_model_path)['state_dict']
            thief_model.load_state_dict(best_state)

            # Compute accuracy and agreement on target dataset
            acc, f1 = testz(thief_model, test_loader)
            agr = agree(target_model, thief_model, test_loader)
            print('Acc, agreement for best model: ', acc, agr)
            
            print('Trial {}/{} || Cycle {}/{} || Label set size {} || Test acc {} || Test agreement {}'.format(trial+1, cfg.TRIALS, cycle+1, cfg.ACTIVE.CYCLES, len(labeled_set), acc, agr))
            print("*"*100, "\n")

            # Select labeled subset for next cycle            
            if cycle!=cfg.ACTIVE.CYCLES-1:

                if cfg.ACTIVE.METHOD == 'random':
                    continue
                    
                elif cfg.ACTIVE.METHOD == 'entropy':
                    # Compute entropy for all unlabeled samples
                    uncertainty, indexes = utils.get_uncertainty_entropy(thief_model, thief_data, unlabeled_set)

                    # select samples with highest entropy
                    arg = np.argsort(uncertainty)
                    selected_index_list = indexes[arg][-(cfg.ACTIVE.ADDENDUM):].numpy().astype('int')
                
                elif cfg.ACTIVE.METHOD == 'kcenter':
                    from kcenter_greedy import kCenterGreedy
                    
                    # init greedy k center
                    sampler = kCenterGreedy(thief_model, thief_data)
                                        
                    # select unlabeled points farthest from all centers
                    selected_index_list = sampler.select_batch(labeled_set, unlabeled_set, N=cfg.ACTIVE.ADDENDUM)
                    
                else:
                    raise(AssertionError)
                
                l = list(selected_index_list)
                labeled_set += l
                unlabeled_set = [unlabeled_set[x] for x in range(len(unlabeled_set)) if unlabeled_set[x] not in selected_index_list]
                addendum_loader = DataLoader(Subset(thief_data_aug, l), batch_size=cfg.TRAIN.BATCH, 
                                                pin_memory=False, num_workers=8, shuffle=False)
                labeled_set = list(set(labeled_set))
                unlabeled_set = list(set(unlabeled_set))
                print('labeled set: ', len(labeled_set))
                print('unlabeled set: ', len(unlabeled_set))
                
                # Query the newly selected samples
                print("replacing addendum labels with victim labels")
                target_model.eval()
                with torch.no_grad():
                    for (d,l0,ind0) in tqdm(addendum_loader):
                        d = d.cuda()
                        label = target_model(d).argmax(axis=1,keepdim=False)
                        label = label.detach().cpu().tolist()
                        for ii, jj in enumerate(ind0):
                            thief_data_aug.samples[jj] = (thief_data_aug.samples[jj][0], label[ii])
                
                print('Label distribution of newly added samples after modification')
                label_dist_new = dist(l, addendum_loader)
                print(label_dist_new)

                # Update the labeled and unlabeled dataloaders
                dataloaders['train'] = DataLoader(Subset(thief_data_aug, labeled_set), batch_size=cfg.TRAIN.BATCH,
                                pin_memory=False, num_workers=8, shuffle=True)
                dataloaders['unlabeled'] = DataLoader(Subset(thief_data_aug, unlabeled_set), batch_size=cfg.TRAIN.BATCH, 
                                            pin_memory=False, num_workers=8, shuffle=True) 
                
                with open(f'{cfg.SAVE_DIR}/X_trial_{trial+1}_cycle_{cycle+1}_labeled_set.npy', 'wb') as f:
                    np.save(f, labeled_set)
                with open(f'{cfg.SAVE_DIR}/X_trial_{trial+1}_cycle_{cycle+1}_val_set.npy', 'wb') as f:
                    np.save(f, val_set) 
                

        # Final stats at the end of a trial
        trial_results = {}
        f = agree(target_model, thief_model, test_loader)
        trial_results['acc'] = acc
        trial_results['agr'] = f
        trial_results['label dist'] = dist(labeled_set, dataloaders['train']) #dist(val_set, val_loader) 
        results_arr.append(trial_results)
        li.append(f)
        
        with open(f'{cfg.SAVE_DIR}/X_trial_{trial+1}_cycle_{cycle+1}_labeled_set.npy', 'wb') as f:
            np.save(f, labeled_set)
        with open(f'{cfg.SAVE_DIR}/X_trial_{trial+1}_cycle_{cycle+1}_val_set.npy', 'wb') as f:
            np.save(f, val_set) 
            
    # Average agreement at the end of all trials
    li=np.array(li)
    print(np.mean(li),np.std(li))
    
    # Compile results of all trials
    df = pd.DataFrame.from_dict(results_arr)
    out_file = f'{cfg.SAVE_DIR}/results.csv'
    df.to_csv(out_file)
    print(df)

    print('Results saved to ', cfg.SAVE_DIR)