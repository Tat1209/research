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

import utils
from datasets import fetch_handler
from run_manager import RunManager, RunsManager
from network import Network, Networks
from trainer import MultiTrainer, Trainer

from ee_tools.models.resnet_cifar_ee import resnet18 as resnet18_cifar_ee
from models.resnet_cifar import resnet18 as resnet18_cifar

network = Network(resnet18_cifar(num_classes=100))

network.torchinfo(input_size=(1, 3, 32, 32), output_path=this_path.parent / "off_arc_64.txt")