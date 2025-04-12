import os
import random

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from PIL import Image
from PIL import ImageDraw



class Trans:
    def __init__(self, info=None, base_ds=None, batch_size=None):
        def blacken_region(x1, y1, x2, y2):
            def transform(image):
                draw = ImageDraw.Draw(image)
                draw.rectangle([x1, y1, x2, y2], fill=0)
                return image
            return transform

        def convert_to_rgb():
            def transform(image): return image.convert('RGB')
            return transform

        self.base = [
                transforms.CenterCrop(85),
                transforms.Lambda(blacken_region(0, 0, 24, 5)),
                transforms.Lambda(blacken_region(85-24, 0, 85-1, 5)),
                ]

        self.tsr = [transforms.ToTensor()]

        self.base_tsr = self.compose(self.base + self.tsr)

        if info is None: info = self.fetch_normal(base_ds, batch_size)
        self.norm = [transforms.Normalize(mean=info["mean"], std=info["std"])]

        self.roflip = [
                transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5), 
                ]
        self.flip90 = [transforms.Lambda(lambda image: rotate(image, 90))]
        self.flip180 = [transforms.Lambda(lambda image: rotate(image, 180))]
        self.flip270 = [transforms.Lambda(lambda image: rotate(image, 270))]
        # self.flip270 = [lambda image: rotate(image, 270)]
        self.hflip = [transforms.RandomHorizontalFlip(p=1)]
        self.vflip = [transforms.RandomVerticalFlip(p=1)]
        self.rflip = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
        self.color = [transforms.Lambda(convert_to_rgb())]
        
        self.gen = self.compose(self.base + self.tsr + self.norm)
        self.aug = self.compose(self.base + self.roflip + self.tsr + self.norm)
        self.flip_aug = self.compose(self.base + self.rflip + self.tsr + self.norm)

        # self.rgb = self.compose(self.base + self.color + self.tsr + self.norm)
        # self.rgbaug = self.compose(self.base + self.roflip + self.color + self.tsr + self.norm)




    def compose(self, args):
        return transforms.Compose(args)


    def fetch_normal(self, base_ds, batch_size):
        base_ds.transform = self.base_tsr
        base_dl = torch.utils.data.DataLoader(base_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')        # GPUが使える場合は、GPU使用モードにする。

        pv = None

        for input_b, label_b in base_dl:
            input_b = input_b.to(device)
            label_b = label_b.to(device)

            pv_list = [input_b[:,i,:,:].flatten() for i in range(input_b.shape[-3])]
            pv_tensor = torch.stack(pv_list, dim=0)
            if pv is None: pv = pv_tensor
            else: pv = torch.cat((pv_tensor, pv), dim=1)

        info = dict()
        info["mean"] = pv.mean(dim=1).cpu()
        info["std"] = pv.std(dim=1).cpu()
        
        print(info)

        return info