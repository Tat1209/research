import sys
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms

this_path = Path(__file__) if '__file__' in globals() else Path("<undefined>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

ee_tools_path_p = work_path / Path("ee")
sys.path.append(str(ee_tools_path_p))

from trainer import Trainer, Network
from datasets import fetch_handler

# from models.resnet_cifar import resnet18 as mine
# from resnet_ku import ResNet18 as ku
# from resnet_we import resnet18 as we

from torchvision.models import mobilenet_v2 as tv
from models.mobilenet_v2_cifar import mobilenet_v2 as mine

from ee_tools.models.mobilenet_v2_ee import mobilenet_v2 as ee

network = Network(ee(num_classes=100, ee_div=8))
net_name = "ee"

p = Path(this_path.parent / f"{net_name}.txt")
ti = network.torchinfo(input_size=(1, 3, 32, 32))
p.write_text(str(ti))
