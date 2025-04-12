import pickle

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from trans import Trans


class FixRandomDataset(Dataset):
    def __init__(self, size):
        super().__init__()

        torch.manual_seed(42)
        self.data = torch.rand(size)

    def __getitem__(self, index):
        data = self.data.detach().clone()
        target = index
        # target = 1
        return data, target

    def __len__(self):
        return 10000


class PklToDataset(Dataset):
    def __init__(self, pkl_file, transform=None):
        with open(pkl_file, "rb") as f:
            (self.datas, self.targets) = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        target = self.targets[idx]
        if self.transform:
            data = self.transform(data)
        return data, target


class Tpl_Dataset(Dataset):
    def __init__(self, something, transform=None, target_transform=None):
        (self.datas, self.targets) = something.something()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        target = self.targets[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target


class Tpl_VisionDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        # super().__init__(root, transforms=transforms)
        (self.datas, self.targets) = root.something()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        target = self.targets[idx]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)

        return data, target


class Datasets:
    """
    Ex.)
    ds = Datasets(root='dir/to/datasets/')
    train_loader = dl(ds("cifar100_train", train_trans), batch_size, shuffle=True)
    for input, label in train_loader:
    ...
    """

    def __init__(self, root=None):
        self.root = root

    def fetch_ds(self, ds_str, transform=None):
        match (ds_str):
            case "cifar100_train":
                return torchvision.datasets.CIFAR100(root=self.root, train=True, transform=transform)
            case "cifar100_val":
                return torchvision.datasets.CIFAR100(root=self.root, train=False, transform=transform)
            case "cifar10_train":
                return torchvision.datasets.CIFAR10(root=self.root, train=True, transform=transform)
            case "cifar10_val":
                return torchvision.datasets.CIFAR10(root=self.root, train=False, transform=transform)
            case "stl10_train":
                return torchvision.datasets.STL10(root=self.root, split="train", transform=transform)
            case "caltech":
                return torchvision.datasets.Caltech101(root=self.root, target_type="category", transform=transform)
            case "imagenet":
                return torchvision.datasets.ImageNet(root=self.root, split="val", transform=transform)
            case "ai-step_train":
                return PklToDataset(f"{self.root}fukui_train_32_60_ver2.pkl", transform=transform)
            case "ai-step_test":
                return PklToDataset(f"{self.root}kanazawa_test_32_60_ver2.pkl", transform=transform)
            case "fix_rand":
                return FixRandomDataset((3, 32, 32))
            case _:
                raise Exception("Invalid name.")

    def __call__(self, ds_str, transform_l=[], seed=None, label_balance=False):
        # seed はデータセットの順番 "arange" は並び替えなし
        ds = self.fetch_ds(ds_str, None)
        ds.ds_str = ds_str
        ds.ds_name = ds.__class__.__name__

        if label_balance:
            indices = self._indices_perm_lb(ds, seed)
        else:
            indices = self._indices_perm(ds, seed)

        return DSInfo(ds, transform_l, indices)

    def _indices_perm(self, ds, seed):
        data_num = len(ds)

        if seed == "arange":
            indices = torch.arange(data_num)
        else:
            if seed is not None:
                torch.manual_seed(seed)
            indices = torch.randperm(data_num)

        indices = indices.tolist()

        return indices

    def _indices_perm_lb(self, ds, seed):
        if seed is not None and seed != "arange":
            torch.manual_seed(seed)

        label_d, len_d = self._fetch_label_data(ds)

        keys = list(len_d.keys())
        values = list(len_d.values())

        length = sum(values)
        len_array = torch.empty(length, dtype=torch.int)
        max_value = max(values)
        index = 0
        counter = torch.zeros(len(keys), dtype=torch.int)  # len_arrayに格納されたkeyの個数をカウントするテンソルを初期化する
        for _ in range(max_value):
            valid_keys = [key for key in keys if counter[key] < values[key]]  # len_dのkeyのうち、まだvalue個格納されていないものを選択する
            m = len(valid_keys)
            if m > 0:
                if seed == "arange":
                    perm = torch.arange(m)
                else:
                    perm = torch.randperm(m)  # valid_keysをランダムにシャッフルする
                shuffled_keys = [valid_keys[p] for p in perm]

                for key in shuffled_keys:  # シャッフルされたkeyに対してループする
                    len_array[index] = key
                    index += 1
                    counter[key] += 1
                    if index == length:
                        break
                if index == length:
                    break

        # label_dのすべてのkeyに対応するvalueのリストをランダムに並び替える
        shuffled_label_d = dict()
        for key in label_d.keys():
            if seed == "arange":
                perm = torch.arange(len(label_d[key]))
            else:
                perm = torch.randperm(len(label_d[key]))
            perm = perm.tolist()

            shuffled_label_d[key] = [label_d[key][p] for p in perm]

        indices = []
        for key in len_array:
            key = key.item()
            value = shuffled_label_d[key][0]  # label_d[key]のリストから先頭の要素を取り出す
            indices.append(value)
            shuffled_label_d[key] = shuffled_label_d[key][1:]  # label_d[key]のリストから先頭の要素を削除する

        return indices

    def _fetch_label_data(self, ds):
        try:
            label_d, len_d = torch.load(f"{self.root}{ds.ds_str}.ld")
        except FileNotFoundError:
            self.ds_save_labels(ds)
            label_d, len_d = torch.load(f"{self.root}{ds.ds_str}.ld")

        return label_d, len_d

    def ds_save_labels(self, ds, batch_size=100):
        # tmp_ds = self(ds.ds_str, transform_l=[Trans.color, Trans.tsr, Trans.resize(1, 1)], seed="arange")
        tmp_ds = self(
            ds.ds_str,
            transform_l=[Trans.color, Trans.tsr, Trans.resize(1, 1)],
            seed="arange",
        )
        tmp_ds = tmp_ds.apply()
        dl = DataLoader(
            tmp_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
        )

        label_d = self.dl_to_labels(dl)

        len_d = dict()
        for key, value in label_d.items():
            len_d[key] = len(value)

        save_obj = (label_d, len_d)
        save_path = self.root + dl.dataset.ds_str + ".ld"
        torch.save(save_obj, save_path)
        print(f"Saved label data to the following path: {save_path}")

        return save_obj

    def dl_to_labels(self, dl, output=False):
        labels = None  # label のリストを作成
        try:
            labels = torch.tensor(dl.dataset.dataset.targets)
        except AttributeError:
            for input_b, label_b in dl:
                if labels is None:
                    labels = label_b
                else:
                    labels = torch.cat([labels, label_b])

        # index と対応させ、label を key とし、index を item とした dict を作成
        label_d = dict()
        for idx, label in enumerate(labels):
            label = label.item()
            if label_d.get(label) is None:
                label_d[label] = [idx]
            else:
                label_d[label].append(idx)

        label_d = dict(sorted(label_d.items()))

        if output:
            for label in label_d.keys():
                print(f"{label}:{len(label_d[label])} items")

        return label_d


class DSInfo:
    def __init__(self, ds, transform_l, indices):
        self.ds = ds
        self.transform_l = transform_l
        self.indices = indices

    def in_range(self, range_t):
        transform_l_new = self.transform_l.copy()
        indices_new = self.indices.copy()

        data_num = len(indices_new)

        if not isinstance(range_t, tuple):
            range_t = (0, range_t)
        idx_range = (int(range_t[0] * data_num), int(range_t[1] * data_num))
        indices_new = indices_new[idx_range[0] : idx_range[1]]

        return DSInfo(self.ds, transform_l_new, indices_new)
        # return DSInfo(self.ds, indices_new, self.transform_l.copy())    # イニシャライザでコピーせず、関数内でコピーするならこっち

    def ex_range(self, range_t):
        data_num = len(self.indices)
        if not isinstance(range_t, tuple):
            range_t = (0, range_t)
        idx_range = (int(range_t[0] * data_num), int(range_t[1] * data_num))
        indices_new = self.indices[: idx_range[0]] + self.indices[idx_range[1] :]

        return DSInfo(self.ds, indices_new, self.transform_l)

    def transform(self, transform_l):
        return DSInfo(self.ds, self.indices, transform_l)

    def apply(self):
        self.ds.transform = torchvision.transforms.Compose(self.transform_l)

        sds = Subset(self.ds, indices=self.indices)
        sds.ds_str = self.ds.ds_str
        sds.ds_name = self.ds.ds_name

        return sds

    def __len__(self):
        return len(self.indices)


def dl(ds, batch_size, shuffle=True):
    if isinstance(ds, DSInfo):
        ds = ds.apply()

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

    # def _clone(self):
    #     new_di = self._clone()
    #     new_di.indices = indices
    #     return self.new_di()
    #     return copy.deepcopy(self)


# # torch.manual_seedを避けたいとき
# def _idx_setter(self, dataset, seed):
#     data_num = len(dataset)
#     idx_list = list(range(data_num))
#     if seed != "arange":
#         random.seed(seed)
#         random.shuffle(idx_list)

#     return data_num, idx_list


# return _Loader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
# class _Loader(DataLoader):
#     def __init__(self, *args, **kwargs):
#         self.args_init = args
#         self.kwargs_init = kwargs
#         super().__init__(*args, **kwargs)


#     def __repr__(self) -> str:
#         format_string = f'{self.__class__.__name__}(\n'

#         for value in self.args_init:
#             format_string += f'dataset = {value.__class__.__name__}()\n'    # 必ず dataset の引数が入る

#         for key, value in self.kwargs_init.items():
#             if key == 'dataset': format_string += f'{getattr(self, key).__class__.__name__}()\n'
#             else: format_string += f"{key} = {value}\n"
#         format_string += ')'
#         return format_string


# class _Subset(Subset):
#     def __init__(self, *args, **kwargs):
#         # self.args_init = args
#         # self.kwargs_init = kwargs
#         super().__init__(*args, **kwargs)
#         self.ds_name = self.dataset.ds_name

#     def __repr__(self) -> str:
#         format_string = self.__class__.__name__ + ' (\n'
#         for attr in dir(self):
#             if not attr.startswith("_") and not callable(getattr(self, attr)): # exclude special attributes and methods
#                 value = getattr(self, attr)
#                 format_string += f"{attr} = {value}\n"
#         format_string += ')'
#         return format_string


# except FileNotFoundError as e:
#     error_msg = f'\nWhen "label_balanced = True", the file "a" must exist, but it does not.\n'\
#             'Before executing, run the following code to create a file containing the label data.\n\n'\
#             'from datasets import Datasets, dl\n'\
#             f'ds = Datasets("{self.root}")\n'\
#             f'dl = dl(ds("{ds.ds_str}", seed="arange"), batch_size=100, shuffle=False)\n'\
#             f'ds.dl_to_labeldict(dl)\n'
#     print(e)
#     print(error_msg)
#     raise FileNotFoundError(f"{e}\n{error_msg}")


# ds.transformを更新する過程で結局.create()みたいなのがいる。このままSubsetで返すようにすると、subsetとしてふるまうがtransformが設定されていないやつが返ってきて分かりづらい。
# 区別するためにをDSInfoを代わりに用いる
# class MySubset(Subset):
#     def __init__(self, ds, indices, transform_l=[]):
#         self.ds = ds
#         self.indices = indices.copy()
#         self.transform_l = transform_l.copy()
#         super().__init__(ds, indices)


#     def in_range(self, range_t):
#         data_num = len(self)
#         if not isinstance(range_t, tuple): range_t = (0, range_t)
#         idx_range = (int(range_t[0] * data_num), int(range_t[1] * data_num))
#         self.indices = self.indices[idx_range[0]:idx_range[1]]

#         return MySubset(self.ds, self.indices, self.transform_l)


#     def ex_range(self, range_t):
#         data_num = len(self)
#         if not isinstance(range_t, tuple): range_t = (0, range_t)
#         idx_range = (int(range_t[0] * data_num), int(range_t[1] * data_num))
#         self.indices = self.indices[:idx_range[0]] + self.indices[idx_range[1]:]

#         return MySubset(self.ds, self.indices, self.transform_l)
