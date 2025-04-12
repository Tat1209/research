import pickle

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from trans import Trans


class FixRandomDataset(Dataset):
    def __init__(self, size, transform=None):
        super().__init__()

        torch.manual_seed(42)
        self.data = torch.rand(size)
        self.transform = transform

    def __getitem__(self, index):
        data = self.data.detach().clone()
        target = index
        # target = 1

        if self.transform is not None: data = self.transform(data)
        return data, target

    def __len__(self):
        return 10000


class FukuiDataset(Dataset):
    def __init__(self, pkl_file, transform=None):
        with open(pkl_file, 'rb') as f:
            (self.data, self.labels) = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform is not None:
            x = self.transform(x)
            # y = self.transform(y)
        return x, y


class Datasets:
    '''
        Ex.)
        ds = Datasets(root='dir/to/datasets/')
        train_loader = dl(ds("cifar100_train", train_trans), batch_size, shuffle=True)
        for input, label in train_loader:
        ...
    '''
    def __init__(self, root=None):
        self.root = root
        
    
    def _fetch_base_ds(self, ds_str, transform):
        match(ds_str): 
            case "cifar100_train": return torchvision.datasets.CIFAR100(root=self.root, train=True, transform=transform)
            case "cifar100_val": return torchvision.datasets.CIFAR100(root=self.root, train=False, transform=transform)
            case "cifar10_train": return torchvision.datasets.CIFAR10(root=self.root, train=True, transform=transform)
            case "cifar10_val": return torchvision.datasets.CIFAR10(root=self.root, train=False, transform=transform)
            case "caltech": return torchvision.datasets.Caltech101(root=self.root, target_type="category", transform=transform)
            case "imagenet": return torchvision.datasets.ImageNet(root=self.root, split='val', transform=transform)
            case "ai-step_train": return FukuiDataset("/home/haselab/Documents/tat/Research/app/ai-step2/fukui_train_32_60_ver2.pkl", transform=transform)
            case "ai-step_test": return FukuiDataset("/home/haselab/Documents/tat/Research/app/ai-step2/kanazawa_test_32_60_ver2.pkl", transform=transform)
            case "fix_rand": return FixRandomDataset((3, 32, 32), transform=transform)

            case _: raise Exception("Invalid name.")


    def __call__(self, ds_str, transform_l=[Trans.tsr], seed=None, label_balance=False, in_range=1.0, ex_range=None):
        # seed はデータセットの順番 "arange" は並び替えなし
        transform = torchvision.transforms.Compose(transform_l)
        ds = self._fetch_base_ds(ds_str, transform)
        ds.ds_str = ds_str

        if label_balance: data_num, indices = self._indices_perm_lb(ds, seed)
        else: data_num, indices = self._indices_perm(ds, seed)
        
        if ex_range is None:
            if not isinstance(in_range, tuple): in_range = (0, in_range)
            idx_range = (int(in_range[0] * data_num), int(in_range[1] * data_num))
            indices = indices[idx_range[0]:idx_range[1]]

        else:
            if not isinstance(ex_range, tuple): ex_range = (0, ex_range)
            idx_range = (int(ex_range[0] * data_num), int(ex_range[1] * data_num))
            indices = indices[:idx_range[0]]+indices[idx_range[1]:]

        sds = Subset(ds, indices=indices)
        sds.ds_name = ds.__class__.__name__
        sds.ds_str = ds_str
        
        return sds


    def _indices_perm(self, ds, seed):
        data_num = len(ds)
        
        if seed == "arange": indices = torch.arange(data_num)
        else:
            if seed is not None: torch.manual_seed(seed)
            indices = torch.randperm(data_num)
        
        indices = indices.tolist()
        
        return data_num, indices


    def _indices_perm_lb(self, ds, seed):
        data_num = len(ds)
        if seed is not None  and  seed != "arange": torch.manual_seed(seed)

        try: label_d, len_d = torch.load(f"{self.root}{ds.ds_str}.ld")
        except FileNotFoundError as e:
            self.ds_to_labeldict(ds)

            label_d, len_d = torch.load(f"{self.root}{ds.ds_str}.ld")


        keys = list(len_d.keys())
        values = list(len_d.values())

        length = sum(values)
        len_array = torch.empty(length, dtype=torch.int)
        max_value = max(values)
        index = 0
        counter = torch.zeros(len(keys), dtype=torch.int)      # len_arrayに格納されたkeyの個数をカウントするテンソルを初期化する

        for _ in range(max_value):
            valid_keys = [key for key in keys if counter[key] < values[key]]    # len_dのkeyのうち、まだvalue個格納されていないものを選択する
            m = len(valid_keys)
            if m > 0:
                if seed == "arange": perm = torch.arange(m)
                else: perm = torch.randperm(m)    # valid_keysをランダムにシャッフルする

                shuffled_keys = [valid_keys[p] for p in perm]
                
                for key in shuffled_keys:   # シャッフルされたkeyに対してループする
                    len_array[index] = key
                    
                    index += 1
                    counter[key] += 1
                    
                    if index == length: break
                if index == length: break
                
        # label_dのすべてのkeyに対応するvalueのリストをランダムに並び替える
        shuffled_label_d = dict()
        for key in label_d.keys():
            if seed == "arange": perm = torch.arange(len(label_d[key]))
            else: perm = torch.randperm(len(label_d[key]))
            perm = perm.tolist()
            
            shuffled_label_d[key] = [label_d[key][p] for p in perm]

        indices = []
        for key in len_array:
            key = key.item()
            value = shuffled_label_d[key][0] # label_d[key]のリストから先頭の要素を取り出す
            indices.append(value)
            shuffled_label_d[key] = shuffled_label_d[key][1:] # label_d[key]のリストから先頭の要素を削除する
            
        return data_num, indices


    def ds_to_labeldict(self, ds, batch_size=100):
        # tmp_ds = self(ds.ds_str, transform_l=[Trans.color, Trans.tsr, Trans.resize(1, 1)], seed="arange")
        tmp_ds = self(ds.ds_str, transform_l=[Trans.color, Trans.tsr, Trans.resize(1, 1)], seed="arange")
        dl = DataLoader(tmp_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

        labels = None   # label のリストを作成
        try: labels = torch.tensor(dl.dataset.dataset.targets)
        except AttributeError: 
            for input_b, label_b in dl:
                if labels is None: labels = label_b
                else: labels = torch.cat([labels, label_b])
                
        # index と対応させ、label を key とし、index を item とした dict を作成
        label_d = dict()
        for idx, label in enumerate(labels):
            label = label.item()
            if label_d.get(label) is None: label_d[label] = [idx]
            else: label_d[label].append(idx)

        label_d = dict(sorted(label_d.items()))
        # for label in label_d.keys(): print(f"{label}:{len(label_d[label])} items")
        len_d = dict()
        for key, value in label_d.items(): len_d[key] = len(value)
            
        save_obj = (label_d, len_d)
        save_path = self.root + dl.dataset.ds_str + ".ld"
        torch.save(save_obj, save_path)
        print(f"Saved it to the following path: {save_path}")
            
        return save_obj


def dl(ds, batch_size, shuffle=True):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)





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





