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

from models.model_factories import create_model_ensembles

model_config = {
    "name": "resnet18",
    "num_classes": 100,
    "T": 1,
    "pretrained": False,
    "div": 1,  # is_ee=True の場合、この引数は無視されます
    "ensembles": 16,  # YAMLの 'num_models: 4' に対応
    "for_cifar_customize": True,
    "is_ee": True,
    "is_he": False
}

model = create_model_ensembles(**model_config)

network = Network(model)

network.torchinfo(input_size=(1, 3, 32, 32), output_path=this_path.parent / "hase_arc.txt")

