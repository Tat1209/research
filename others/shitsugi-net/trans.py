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
    def __init__(self):
        def convert_to_rgb():
            def transform(image): return image.convert('RGB')
            return transform

        tsr = [transforms.ToTensor()]
        cf_norm = [transforms.Normalize(mean=[0.5070751309394836, 0.48654884099960327, 0.44091784954071045], std=[0.2673342823982239, 0.2564384639263153, 0.2761504650115967], inplace=True)]
        in_norm = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)]

        color = [transforms.Lambda(convert_to_rgb())]
        res = [torchvision.transforms.Resize((32, 32))]
        res224 = [torchvision.transforms.Resize((224, 224))]
        rotate_flip = [transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5)]
        light_aug = [transforms.RandomRotation(degrees=(-45, 45), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5)]
        crop32 = [transforms.RandomCrop(32, padding=4,padding_mode='reflect'), transforms.RandomHorizontalFlip()]
        crop224 = [transforms.RandomCrop(224, padding=4,padding_mode='reflect'), transforms.RandomHorizontalFlip()]
        flip90 = [transforms.Lambda(lambda image: rotate(image, 90))]
        flip180 = [transforms.Lambda(lambda image: rotate(image, 180))]
        flip270 = [transforms.Lambda(lambda image: rotate(image, 270))]
        hflip = [transforms.RandomHorizontalFlip(p=1)]
        vflip = [transforms.RandomVerticalFlip(p=1)]
        rflip = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
        pil = [transforms.ToPILImage()]

        # flipとか回転は、PILの状態で操作しなければいけないものもある。テンソルにする前に行う必要がある。
        self.cf_gen = self.compose(tsr + cf_norm)
        self.cf_crop = self.compose(crop32 + tsr + cf_norm)
        # self.aug = self.compose(rotate_flip + tsr + norm)
        # self.laug = self.compose(light_aug + tsr + norm)
        # self.flip_aug = self.compose(rflip + tsr + norm)

        self.in_gen = self.compose(res224 + tsr + in_norm)

        # self.calgen = self.compose(color + res + tsr + norm)
        # self.calgen_2 = self.compose(color + res224 + tsr + norm)
        # self.calcrop = self.compose(color + res224 + crop224 + tsr + norm)


    def compose(self, args):
        return transforms.Compose(args)

