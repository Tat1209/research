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

# print(len(ds("cars_train").balance_label().fetch_ld()[1]))
# print(len(ds("pets_train").balance_label().fetch_ld()[1]))
# print(len(ds("flowers_train").balance_label().fetch_ld()[1]))

a, b = ds("caltech101_trainval").split_ratio(0.7, balance_label=False, seed=0)
a.fetch_ld(output=True)
b.fetch_ld(output=True)
print(a.fetch_ld()[1][0])
print(b.fetch_ld()[1][0])
# print(len(a))
# print(len(b))
# ds("pets_train").balance_label().fetch_ld(output=True)
# ds("flowers_train").balance_label().fetch_ld(output=True)

# print(ds("mnist_train", transform_l=[Trans.tsr]).calc_min_max(formatted=True))
# print(ds("cifar10_train", transform_l=[Trans.tsr]).calc_min_max(formatted=True))
# print(ds("mnist_train", transform_l=[Trans.tsr]).calc_mean_std(formatted=True))
