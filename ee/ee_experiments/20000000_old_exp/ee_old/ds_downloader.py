import random

import torch
import torchvision


working_dir = "/home/haselab/Documents/tat/Research/"
root = f"{working_dir}assets/datasets/"

torchvision.datasets.CIFAR10(root=root, train=True, download=True)
torchvision.datasets.CIFAR100(root=root, train=True, download=True)
torchvision.datasets.Caltech101(root=root, target_type="category", download=True)
torchvision.datasets.STL10(root=root, split="train", download=True)
