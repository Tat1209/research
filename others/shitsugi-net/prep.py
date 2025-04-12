import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from PIL import ImageDraw
import pandas as pd

import random


import torch
from torchvision import transforms


class Prep:
    def __init__(self, val_range=None, seed=None):
        self.seed = seed


    def detasets(self, ds_name, transform):
        root = '/home/haselab/Documents/tat/app/shitsugi-net/data'
        # root = '/home/haselab/Documents/tat/app/shitsugi-net/data'
        match(ds_name): 
            case "cifar_train": return torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=transform)
            case "cifar_val": return torchvision.datasets.CIFAR100(root=root, train=False, download=False, transform=transform)
            case "caltech": return torchvision.datasets.Caltech101(root=root, target_type="category", transform=transform, download=False)
            case "imagenet": return torchvision.datasets.ImageNet(root=root, split=root, transform=transform)
            case _: raise Exception("Invalid name.")


    def dl(self, ds_name, transform, batch_size, in_range=1.0, ex_range=None):
        ds = self.detasets(ds_name, transform)
        if not hasattr(self, ds_name): setattr(self, ds_name, self.idx_setter(ds))
        data_num, idx_list = getattr(self, ds_name)
        
        if ex_range is None:
            if isinstance(in_range, float): in_range = (0, in_range)
            idx_range = (int(in_range[0] * data_num), int(in_range[1] * data_num))
            indices = idx_list[idx_range[0]:idx_range[1]]
            sds = torch.utils.data.Subset(ds, indices=indices)

        else:
            if isinstance(ex_range, float): ex_range = (0, ex_range)
            idx_range = (int(ex_range[0] * data_num), int(ex_range[1] * data_num))
            indices = idx_list[:idx_range[0]]+idx_list[idx_range[1]:]
            sds = torch.utils.data.Subset(ds, indices=indices)

        # print(len(sds), indices[:5])
        dl = self.fetch_loader(sds, batch_size)
        return dl

    
    def idx_setter(self, dataset):
        data_num = len(dataset)
        idx_list = list(range(data_num))
        if self.seed is not None: random.seed(self.seed)
        random.shuffle(idx_list)
        
        return data_num, idx_list


    def fetch_loader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        




