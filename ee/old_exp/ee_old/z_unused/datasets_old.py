import random

import torch
import torchvision
from torch.utils.data import Dataset


class FixRandomDataset(Dataset):
    def __init__(self, size):
        super().__init__()

        torch.manual_seed(42)
        self.data = torch.rand(size)

    def __getitem__(self, index):
        data = self.data.clone().detach()
        target = index
        # target = 1
        return data, target

    def __len__(self):
        return 10000


class Datasets:
    '''
        Ex.)
        ds = Datasets(root='dir/to/datasets/')
        train_loader = ds.dl("cifar100_train", transform, batch_size)
        for input, label in train_loader:
        ...
    '''
    def __init__(self, root=None):
        self.root = root


    def detasets(self, ds_name, transform_l):
        transform = torchvision.transforms.Compose(transform_l)
        download=False
        match(ds_name): 
            case "cifar100_train": return torchvision.datasets.CIFAR100(root=self.root, train=True, download=download, transform=transform)
            case "cifar100_val": return torchvision.datasets.CIFAR100(root=self.root, train=False, download=download, transform=transform)
            case "cifar10_train": return torchvision.datasets.CIFAR10(root=self.root, train=True, download=download, transform=transform)
            case "cifar10_val": return torchvision.datasets.CIFAR10(root=self.root, train=False, download=download, transform=transform)
            case "caltech": return torchvision.datasets.Caltech101(root=self.root, target_type="category", transform=transform, download=download)
            case "imagenet": return torchvision.datasets.ImageNet(root=self.root, split='val', transform=transform)
            case "fix_rand": return FixRandomDataset((3, 32, 32))

            case _: raise Exception("Invalid name.")


    def dl(self, ds_name, transform_l, batch_size, seed=None, in_range=1.0, ex_range=None, shuffle=True):
        # seed はデータセットの順番 "fix" は並び替えなし
        # shuffle はデータロード時の並び替え
        # seed で順番を固定してから range でデータセットの範囲を選択する。shuffle はこの範囲内で読み込み時に並び替え

        ds = self.detasets(ds_name, transform_l)
        data_num, idx_list = self.idx_setter(ds, seed)
        
        if ex_range is None:
            if not isinstance(in_range, tuple): in_range = (0, in_range)
            idx_range = (int(in_range[0] * data_num), int(in_range[1] * data_num))
            indices = idx_list[idx_range[0]:idx_range[1]]

        else:
            if not isinstance(ex_range, tuple): ex_range = (0, ex_range)
            idx_range = (int(ex_range[0] * data_num), int(ex_range[1] * data_num))
            indices = idx_list[:idx_range[0]]+idx_list[idx_range[1]:]

        sds = torch.utils.data.Subset(ds, indices=indices)

        dl = self.fetch_loader(sds, batch_size, shuffle)

        return dl

    
    def idx_setter(self, dataset, seed):
        data_num = len(dataset)
        idx_list = list(range(data_num))
        if seed is not None: random.seed(seed)
        if seed != "fix": random.shuffle(idx_list)
        
        return data_num, idx_list


    def fetch_loader(self, dataset, batch_size, shuffle):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
        




