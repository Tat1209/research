from pathlib import Path

import torch
from datasets import Datasets, dl
from trans import Trans

from time import time

start = time()

work_path = '/home/haselab/Documents/tat/Research/'
ds = Datasets(root=f"{work_path}assets/datasets/")

# loader = dl(ds("fix_rand", [Trans.tsr], seed=5, in_range=(0, 0.001)), batch_size=5, shuffle=False)
loader = dl(ds("cifar100_train", [Trans.tsr], seed='arange', in_range=1), batch_size=100, shuffle=False)


labels = None
for input_b, label_b in loader:
    if labels is None: labels = label_b
    else: labels = torch.cat([labels, label_b])
    

label_d = dict()
for idx, label in enumerate(labels):
    label = label.item()
    if label_d.get(label) is None: label_d[label] = [idx]
    else: label_d[label].append(idx)

label_d = dict(sorted(label_d.items()))
    

    
print(len(torch.unique(labels)))

path = Path("/home/haselab/Documents/tat/Research/app/ee/labels/cifar100_label.py")
dict_name = "labels"

with open(path, "a") as fh:
    fh.write(f"{dict_name} = dict()\n")

    for key, value in label_d.items():
        # print(f"{key}:{len(value)} items")
        f_str = f"{dict_name}[{key}] = {value}\n"
        fh.write(f_str)

    
# print(label_d[0])


print(time() - start)


    



