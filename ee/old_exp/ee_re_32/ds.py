import time
import sys
from pathlib import Path

import torch

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets, dl
from trans import Trans

ds = Datasets(root=work_path / "assets/datasets/")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# base_train_ds = ds(train_ds_str, train_trans).balance_label(seed=0).limit_class(max_num=10)
# val_ds = ds(val_ds_str, val_trans).limit_class(base_train_ds.fetch_classes(list=True))

a = ds("tiny-imagenet_val").limit_class(max_num=100)
# a = ds("caltech101_trainval").limit_class(max_num=10).balance_label()
# label_l, label_d = a.fetch_ld(output=True)
a.fetch_ld(output=True)
# print(a.fetch_classes(listed=True))
# label_l, label_d = a.fetch_ld(output=True)


