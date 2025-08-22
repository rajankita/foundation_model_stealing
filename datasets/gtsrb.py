import sys
import os
import os.path as osp

from torchvision.datasets import GTSRB as TVGTSRB

# import knockoff.config as cfg
from torchvision.datasets.utils import check_integrity
import pickle


class GTSRB(TVGTSRB):
    def __init__(self, cfg, train=True, transform=None, target_transform=None, download=True):
        root = cfg.VICTIM.DATA_ROOT
        if isinstance(train, bool):
            split = 'train' if train else 'test'
        else:
            split = train
        super().__init__(root, split, transform, target_transform, download)
        self.n_classes = 43

    def get_image(self, index):
        return self.data[index]