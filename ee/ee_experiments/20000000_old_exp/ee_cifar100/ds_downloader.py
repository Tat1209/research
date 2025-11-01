import sys
import random
from pathlib import Path

import torch
import torchvision


work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

root = f"{work_path}/assets/datasets/"

torchvision.datasets.CIFAR10(root=root, train=True, download=True)
torchvision.datasets.CIFAR100(root=root, train=True, download=True)
# torchvision.datasets.Caltech101(root=root, target_type="category", download=True)
# torchvision.datasets.STL10(root=root, split="train", download=True)
# torchvision.datasets.STL10(root=root, split="test", download=True)

from torchvision.datasets.utils import download_url

class CustomSTL10(torchvision.datasets.STL10):
    def download(self):
        if not self._check_integrity():
            download_url('http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz',
                         self.root, 'stl10_binary.tar.gz', '8d2783665e5c4d81f14d38fefbd25e8f')
            self.extract_file()

CustomSTL10(root=root, split="train", download=True)
CustomSTL10(root=root, split="test", download=True)
