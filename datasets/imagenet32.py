import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


class ImageNet32(Dataset):
    def __init__(self, data_root, transform=None, val=False, num_filesets=10):
        super().__init__()
        if val:
            filenames = [os.path.join(data_root, 'val_data')]
        else:    
            filenames = [('train_data_batch_%d' % (i+1)) for i in range(num_filesets)]
            filenames = [os.path.join(data_root, f) for f in filenames]
        
        self.samples = []
        for filename in tqdm(filenames):
            if os.path.isfile(filename):
                res = unpickle(filename)
                Xs = res['data'].reshape((res['data'].shape[0],3,32,32))/255
                ys = res['labels']
                for i in range(len(ys)):
                    self.samples += [(Xs[i], ys[i]-1)]

        # convert X to tensor from ndarray
        # self.X = torch.tensor(self.X).float()
        # self.X = self.X.clone().detach()
        
        IdentityTransform = transforms.Lambda(lambda x: x)  
        if transform is not None:
            self.transform = transform
        else:
            self.transform = IdentityTransform

    def __getitem__(self, index):
        X, y = self.samples[index]
        X = torch.tensor(X).float()
        
        if self.transform is not None:
            X = self.transform(X)

        return X, y, index

    def __len__(self):
        return len(self.samples)
    

class ImageNet32_Soft(Dataset):
    def __init__(self, data_root, transform=None):
        super().__init__()
        filenames = [('train_data_batch_%d' % (i+1)) for i in range(10)]
        filenames = [os.path.join(data_root, f) for f in filenames]
        
        self.samples = []
        for filename in filenames:
            if os.path.isfile(filename):
                res = unpickle(filename)
                Xs = res['data'].reshape((res['data'].shape[0],3,32,32))/255
                ys = res['labels']
                ys_soft = np.zeros((len(ys), 10), dtype='float32')
                # ys_soft = res['labels']
                for i in range(len(ys)):
                    # self.samples += [(Xs[i], ys[i]-1, ys_soft[i]-1)]
                    self.samples += [(Xs[i], ys[i]-1, ys_soft[i])]

        IdentityTransform = transforms.Lambda(lambda x: x)  
        if transform is not None:
            self.transform = transform
        else:
            self.transform = IdentityTransform
            
    def __getitem__(self, index):
        X, y, y_soft = self.samples[index]
        X = torch.tensor(X).float()
        
        if self.transform is not None:
            X = self.transform(X)

        return X, y, y_soft, index
  
    def __len__(self):
        return len(self.samples)