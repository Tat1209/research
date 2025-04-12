import sys

import torch
import torchvision

work_path = "/home/haselab/Documents/tat/Research/"
sys.path.append(f"{work_path}app/torch_libs/")

from datasets import Datasets, dl
from run_manager_old import RunManager, RunsManager
from trainer import Model, MultiTrain
from trans import Trans
import utils

work_path = "/home/haselab/Documents/tat/Research/"
ds = Datasets(root=f"{work_path}assets/datasets/")

train_ds, val_ds = ds("ai-step_l").split(ratio=0.7, shuffle=True, balance_label=True)

# base_ds = ds("ai-step_l").shuffle().mult_label({0: 1250, 1: 3, 2: 16, 4: 125})

# u_num_label = 4000
# labeled_ds = ds("ai-step_l").shuffle().mult_label({0: int(u_num_label / 2), 1: int(u_num_label / 931), 2: int(u_num_label / 159), 4: int(u_num_label / 20), 5: int(u_num_label / 3112)})

# val_range = (0.8, 1.0)
# train_ds = labeled_ds.ex_range(val_range).transform(Trans.as_da)
# val_ds = labeled_ds.in_range(val_range).transform(Trans.as_gen)
# train_loader = dl(train_ds, batch_size=200, shuffle=True)
# val_loader = dl(val_ds, batch_size=2000, shuffle=True)

# print(len(tmp_ds))
# label_l, label_d = train_ds.fetch_ld(output=True)
# label_l, label_d = val_ds.fetch_ld(output=True)


set1 = set(train_ds.indices)
set2 = set(val_ds.indices)

# 重複する要素を見つける
duplicates = set1.intersection(set2)
print(duplicates)
# print(label_d)
# print(label_l)


# print(base_ds.indices)
# print(tmp_ds.indices)
# print(tmp_ds2.indices)


def print_labels(dl, file=None):
    labels = None
    for input, label in dl:
        if labels is None:
            labels = label
        else:
            labels = torch.cat([labels, label])

    if file:
        with open(file, "w") as fh:
            print(labels.tolist(), file=fh)
    else:
        print(labels)
