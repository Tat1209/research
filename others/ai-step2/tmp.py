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

# base_ds = ds("ai-step_l").shuffle().mult_label({0: 1250, 1: 3, 2: 16, 4: 125})
labeled_ds = ds("ai-step_l").shuffle()
# labeled_ds = ds("ai-step_l").shuffle().mult_label({0: 1250, 1: 3, 2: 16, 4: 125})
label_l, label_d = labeled_ds.fetch_ld(output=True)
tsr = labeled_ds.fetch_weight()
# tmp_ds = base_ds.transform([Trans.tsr])
print(tsr)


labeled_ds = ds("ai-step_l").shuffle()

# val_range = (0.8, 1.0)
# train_ds = labeled_ds.ex_range(val_range).transform(Trans.as_da)
# val_ds = labeled_ds.in_range(val_range).transform(Trans.as_gen)
# train_loader = dl(train_ds, batch_size=200, shuffle=True)
# val_loader = dl(val_ds, batch_size=2000, shuffle=True)

# loader = dl(tmp_ds, batch_size=100)
# for i, l in loader:
#     pass


# print(len(tmp_ds))
# label_l, label_d = labeled_ds.fetch_ld(output=True)
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
