import sys
from pathlib import Path

this_path = Path(__file__) if '__file__' in globals() else Path("<undefined>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

ee_tools_path_p = work_path / Path("ee")
sys.path.append(str(ee_tools_path_p))

import torch
from ee_tools.ee_refiner import EERefiner
from network import Network, Refiner, comp_param_stat
# from torchvision.models.resnet import resnet18 as tv
# from torchvision.models.resnet import wide_resnet101_2 as tv
# from torchvision.models.convnext import convnext_base as tv
from torchvision.models.regnet import regnet_x_16gf as tv
# from torchvision.models.mobilenetv2 import mobilenet_v2 as tv
# from torchvision.models.efficientnet import efficientnet_b7 as tv

input_size = (2, 3, 32, 32)
# input_size = (1, 3, 64, 64)
# input_size = (1, 3, 224, 224)
torchinfo = False

networks = []
networks_names = []

base_network = tv(num_classes=100)
network = Network(base_network)
networks.append(network)
networks_names.append("tv")


base_network = tv(num_classes=100)
network = Network(EERefiner(base_network).multi_narrow(div=8, agg="both").init_weights().build()).to(device="cuda")
networks.append(network)
networks_names.append("ee")

from ee_tools.ee_refiner import ChunkMerge, EEWrapper, RepeatData
print("=== Checking agg settings ===")
found = False

for name, module in network.named_modules():
    # ChunkMergeが見つかったらそのaggを表示
    if isinstance(module, ChunkMerge):
        print(f"Layer: '{name}' | Type: ChunkMerge | agg: {module.agg}")
        found = True

    if isinstance(module, ChunkMerge):
        print(f"Layer: '{name}' | Type: ChunkMerge | chunks: {module.chunks}")
        found = True
    
    # EEWrapperが見つかったらそのaggを表示
    if isinstance(module, EEWrapper):
        print(f"Layer: '{name}' | Type: EEWrapper | agg: {module.agg}")
        found = True

    # EEWrapperが見つかったらそのaggを表示
    if isinstance(module, RepeatData):
        print(f"Layer: '{name}' | Type: RepeatData | n: {module.n}")
        found = True

if not found:
    print("ChunkMerge or EEWrapper not found in the model.")
    
base_network = tv(num_classes=100)
network = Network(EERefiner(base_network).cifar_style().multi_narrow(div=2, agg="both").init_weights().build()).to(device="cuda")
# network = Network(EERefiner(base_network).cifar_style().multi_narrow(div=2).init_weights().build()).to(device="cuda")
networks.append(network)
networks_names.append("ee_cifar")


for network, net_name in zip(networks, networks_names):
    p = Path(this_path.parent / f"{net_name}.txt")
    if torchinfo:
        ti = network.torchinfo(input_size=input_size)
        p.write_text(str(ti))
    else:
        network = network.to(device="cuda")
        out = network(torch.randn(input_size).to(device=network.device))
        p.write_text(str(network.base_model))


# comp_param_stat(networks, layer_width=50)