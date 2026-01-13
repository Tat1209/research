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
from ee_tools.ee_refiner import EERefiner
from ee_tools.ee_trainer import EETrainer
from network import Network, Networks
from exp_manager import ExpManager
from torchvision import models
from trainer import MultiTrainer, Trainer


exp_name = "exp_model_bl_best"
exp = ExpManager(exp_path=this_path.parent / exp_name, exp_tpl="exp_tpl_ee")
exp.ref_results()