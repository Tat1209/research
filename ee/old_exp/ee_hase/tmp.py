import sys
from pathlib import Path
import math

import torch
from torchvision import transforms

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets
from run_manager import RunManager, RunsManager, RunViewer
# from trainer import MultiTrainer as MyMultiTrain
# from trainer import Trainer
from trainer import Trainer, MyMultiTrain
from trans import Trans

from mytrainer import MyTrainer

# from models.resnet_ee import resnet18 as net
from models.resnet_git_ee import resnet18 as net

ds = Datasets(root=work_path / "assets/datasets/")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

max_lr = 0.0005
batch_size = 128

train_trans = [transforms.ToTensor()]
val_trans = [transforms.ToTensor()]

train_ds = ds("stl10_train", train_trans).fetch_base_ld()
val_ds = ds("stl10_val", val_trans)

# print(train_ds.calc_mean_std(formatted=True))
# train_ds = ds("stl10_train", train_trans)
# val_ds = ds("stl10_val", val_trans)
