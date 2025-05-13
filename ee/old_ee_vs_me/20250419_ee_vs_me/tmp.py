import sys
import math
from pathlib import Path

import torch
from torchvision import transforms

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "research"), None))
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

from datasets import Datasets
from run_manager import RunManager, RunsManager
from trainer import Trainer, MultiTrainer
from modules import CrossEntropyLossT
import utils

from models.resnet_ee import resnet18 as resnet18_ee
from models.resnet_git_ee import resnet18 as resnet18_git_ee
from models.resnet_git_ee import resnet50 as resnet50_git_ee

def print_stat(mode, tsr):
    disp_str = f"abs_mean: {tsr.abs().mean().item():8.6f}, shape: {tsr.shape}"
    # disp_str = f"mean: {tsr.mean().item():8.6f}, shape: {tsr.shape}"
    print(f"{mode} {disp_str}")

# デフォルトの初期化
mode = "def_init_EE"
conv = torch.nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
tsr = conv.weight.data
tsr = tsr[:32]
print_stat(mode, tsr)

mode = "def_init_ME"
conv = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
tsr = conv.weight.data
print_stat(mode, tsr)

# resnetの初期化
mode = "res_init_EE"
conv = torch.nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
torch.nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
tsr = conv.weight.data
tsr = tsr[:32]
print_stat(mode, tsr)

mode = "res_init_ME"
conv = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
torch.nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
tsr = conv.weight.data
print_stat(mode, tsr)

# resnetの初期化において、groupsを考慮したもの
mode = "org_init_EE"
conv = torch.nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
for w in conv.weight.chunk(conv.groups, dim=0):
    torch.nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
tsr = conv.weight.data
tsr = tsr[:32]
print_stat(mode, tsr)

mode = "org_init_ME"
conv = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
for w in conv.weight.chunk(conv.groups, dim=0):
    torch.nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
tsr = conv.weight.data
print_stat(mode, tsr)
