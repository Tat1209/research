import os
import random

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from PIL import ImageDraw
import pandas as pd

# from trans import Trans



class LabeledDS(torch.utils.data.Dataset):
    def __init__(self, fname):
        df = pd.read_csv(fname)
        print(df)
        self.feat = df.iloc[:, :-1]
        self.label = df.iloc[:, -1]

    def __getitem__(self, index):
        return self.feat.iloc[index], self.label.iloc[index]

    def __len__(self):
        return len(self.label)


class UnlabeledDS(torch.utils.data.Dataset):
    def __init__(self, fname):
        print(df)
        df = pd.read_csv(fname)
        self.feat = df

    def __getitem__(self, index):
        return self.feat.iloc[index]

    def __len__(self):
        return len(self.feat)



class Prep:
    def __init__(self, data_path, batch_size, val_range=0., seed=None):
        self.data_path = data_path
        self.batch_size = batch_size
        
        labeled_ds = LabeledDS(data_path["labeled"])
        # self.tr = Trans(info={'mean':[0.3108, 0.3108, 0.3108], 'std':[0.3188, 0.3188, 0.3188]})

        self.data_num = len(labeled_ds)
        self.rand_idxs = list(range(self.data_num))
        random.seed(seed)
        random.shuffle(self.rand_idxs)

        if not isinstance(val_range, tuple): self.val_range = (0, 0)
        else: self.val_range = val_range


    def fetch_train(self):
        ds = LabeledDS(self.data_path["labeled"])
        idx = (int(self.val_range[0] * self.data_num), int(self.val_range[1] * self.data_num))
        ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[:idx[0]]+self.rand_idxs[idx[1]:])
        dl = self.fetch_loader(ds)

        return dl


    def fetch_val(self):
        if self.val_range[0] == self.val_range[1]: return None
        ds = LabeledDS(self.data_path["labeled"])
        idx = (int(self.val_range[0] * self.data_num), int(self.val_range[1] * self.data_num))
        ds = torch.utils.data.Subset(ds, indices=self.rand_idxs[idx[0]:idx[1]])
        dl = self.fetch_loader(ds)

        return dl


    def fetch_test(self):
        ds = UnlabeledDS(self.data_path["unlabeled"])
        dl = self.fetch_loader(ds)
        return dl


    def fetch_loader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        



