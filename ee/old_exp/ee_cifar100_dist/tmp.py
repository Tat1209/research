from tsnecuda import TSNE
import time
import sys
from pathlib import Path

import torch

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets
from trans import Trans
from trainer import Trainer

ds = Datasets(root=work_path / "assets/datasets/")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_ds = ds("mnist_train")

print(train_ds.calc_classdist())


