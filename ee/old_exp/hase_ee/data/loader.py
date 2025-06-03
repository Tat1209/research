import os
from PIL import Image
import numpy as np

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms as tt
from torch.utils.data import DataLoader

class InMemoryDataset(Dataset):
    def __init__(self, root_dir, transform=None, test=False, image_size=80):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.data = []
        self.labels = []
        
        with open(os.path.dirname(root_dir) + "/wnids.txt", "r") as f:
            wnids = f.read().splitlines()
        self.idmap = {n:i  for i, n in enumerate(wnids)}
        
        # すべての画像をメモリに読み込む
        if not test:
            for label_folder in os.listdir(root_dir):
                label_path = os.path.join(root_dir, label_folder)
                if os.path.isdir(label_path):
                    img_dir_path = os.path.join(label_path, 'images')
                    for img_name in os.listdir(img_dir_path):
                        img_path = os.path.join(img_dir_path, img_name)
                        image = Image.open(img_path).convert('RGB')
                        self.data.append(image)
                        self.labels.append(label_folder)  # ラベルをフォルダ名にする場合
        else:
            with open(self.root_dir + "/val_annotations.txt", "r") as f:
                lines = f.read().splitlines()
                lines = [line.split('\t') for line in lines]
            
            img_dir_path = os.path.join(root_dir, 'images')
            for i, line in enumerate(lines):
                img_name, label = line[0], line[1]
                img_path = os.path.join(img_dir_path, img_name)
                image = Image.open(img_path).convert('RGB')
                self.data.append(image)
                self.labels.append(label)
        
        # すべての画像をリサイズ
        self.data = [image.resize((self.image_size, self.image_size)) for image in self.data]
        
        self.labels_index = [self.idmap[label] for label in self.labels]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels_index[idx]
        return image, label
    
class CaltechDataset(Dataset):
    def __init__(self, filename, transform=None, test=False):
        self.filename = filename
        self.transform = transform
        z = np.load(filename)
        if test:
            self.data, self.labels = z['arr_1'], z['arr_3']
        else:
            self.data, self.labels = z['arr_0'], z['arr_2']
        if self.data.shape[1] == 1:
            self.data = [Image.fromarray(d[0]).convert('RGB') for d in self.data]
        else:
            self.data = [Image.fromarray(d) for d in self.data]
        self.labels = self.labels.astype(np.int64)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def prepare_dataloader_tinyimagenet(root_dir, batch_size, image_size=72):
    # データ前処理
    transform_train = tt.Compose([tt.RandomCrop(size=64),
                                  tt.RandomHorizontalFlip(),
                                  tt.RandomRotation(15),
                                  tt.ToTensor(),
                                  tt.Normalize(mean=[0.485, 0.456, 0.406],  # 通常のImageNetの平均と標準偏差
                                               std=[0.229, 0.224, 0.225])])
    transform_test = tt.Compose([tt.CenterCrop((64, 64)), 
                                 tt.ToTensor(), 
                                 tt.Normalize(mean=[0.485, 0.456, 0.406],  # 通常のImageNetの平均と標準偏差
                                              std=[0.229, 0.224, 0.225])])

    # カスタムデータセットの作成
    train_ds = InMemoryDataset(root_dir=root_dir + 'tiny-imagenet-200/train', transform=transform_train, test=False, image_size=image_size)
    test_ds = InMemoryDataset(root_dir=root_dir + 'tiny-imagenet-200/val', transform=transform_test, test=True, image_size=image_size)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl

def prepare_dataloader_cifar(root_dir, batch_size, cifar=10):
    transform_train = tt.Compose([
        tt.RandomCrop(32, padding=4),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if cifar == 10:
        train_ds = torchvision.datasets.CIFAR10(f'{root_dir}/CIFAR10', train=True, download=True, transform=transform_train)
        test_ds = torchvision.datasets.CIFAR10(f'{root_dir}/CIFAR10', train=False, download=True, transform=transform_test)
    elif cifar == 100:
        train_ds = torchvision.datasets.CIFAR100(f'{root_dir}/CIFAR10', train=True, download=True, transform=transform_train)
        test_ds = torchvision.datasets.CIFAR100(f'{root_dir}/CIFAR10', train=False, download=True, transform=transform_test)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)#, num_workers=4)
    return train_dl, test_dl

def prepare_dataloader(filename, batch_size, size=224, padding=16):
    transform_train = tt.Compose([tt.RandomCrop(size=size, padding=padding),
                                  tt.RandomHorizontalFlip(),
                                  tt.RandomRotation(15),
                                  tt.ToTensor(),
                                  tt.Normalize(mean=[0.485, 0.456, 0.406],  # 通常のImageNetの平均と標準偏差
                                               std=[0.229, 0.224, 0.225])])
    transform_test = tt.Compose([tt.ToTensor(), 
                                 tt.Normalize(mean=[0.485, 0.456, 0.406],  # 通常のImageNetの平均と標準偏差
                                              std=[0.229, 0.224, 0.225])])

    train_ds = CaltechDataset(filename, test=False, transform=transform_train)
    test_ds = CaltechDataset(filename, test=True, transform=transform_test)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)#, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)#, num_workers=4)
    return train_dl, test_dl

def get_dataloader(dataname, root_dir, batch_size, num_gpus):
    # データローダーの作成
    if dataname == "tiny":
        train_dl, test_dl = prepare_dataloader_tinyimagenet(root_dir, batch_size*num_gpus)
    elif dataname == "cifar10":
        train_dl, test_dl = prepare_dataloader_cifar(root_dir, batch_size*num_gpus, cifar=10)
    elif dataname == "cifar100":
        train_dl, test_dl = prepare_dataloader_cifar(root_dir, batch_size*num_gpus, cifar=100)
    elif dataname == "caltech101":
        train_dl, test_dl = prepare_dataloader(f"caltech101.npz", batch_size*num_gpus)
    elif dataname == "flowers102":
        train_dl, test_dl = prepare_dataloader(f"flowers102.npz", batch_size*num_gpus)
    elif dataname == "oxfordpet":
        train_dl, test_dl = prepare_dataloader(f"oxfordpet.npz", batch_size*num_gpus)
    elif dataname == "food101":
        train_dl, test_dl = prepare_dataloader(f"food101.npz", batch_size*num_gpus)
    elif dataname == "dtd":
        train_dl, test_dl = prepare_dataloader(f"dtd.npz", batch_size*num_gpus)
    elif dataname == "gtsrb":
        train_dl, test_dl = prepare_dataloader(f"gtsrb.npz", batch_size*num_gpus, size=32, padding=4)
    elif dataname == "domainnet_clipart":
        train_dl, test_dl = prepare_dataloader(f"DomainNet_clipart.npz", batch_size*num_gpus)
    elif dataname == "cub200":
        train_dl, test_dl = prepare_dataloader(f"cub200.npz", batch_size*num_gpus)
    elif dataname == "eurosat":
        train_dl, test_dl = prepare_dataloader(f"eurosat.npz", batch_size*num_gpus, size=64)
    elif dataname == "covidx":
        train_dl, test_dl = prepare_dataloader(f"covidx.npz", batch_size*num_gpus)
    elif dataname == "isic2019":
        train_dl, test_dl = prepare_dataloader(f"isic2019.npz", batch_size*num_gpus)
    return train_dl, test_dl

# データセットの一部を取得（先頭からnum件）
def get_subset_dataloader(loader, num):
    subset_loader = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=True)
    subset_loader.dataset.data = subset_loader.dataset.data[:num]
    if hasattr(subset_loader.dataset, "targets"):
        subset_loader.dataset.targets = subset_loader.dataset.targets[:num]
    if hasattr(subset_loader.dataset, "labels"):
        subset_loader.dataset.labels = subset_loader.dataset.labels[:num]
    return subset_loader