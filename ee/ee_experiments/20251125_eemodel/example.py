import torch
from ee_refiner import EERefiner
from pathlib import Path

from torchvision.models import resnet18 as tv
# from torchvision.models import wide_resnet101_2 as tv
# from torchvision.models import convnext_base as tv
# from torchvision.models import regnet_x_16gf as tv
# from torchvision.models import mobilenet_v2 as tv
# from torchvision.models import efficientnet_b7 as tv

from torchvision.models import resnet as tv

this_path = Path(__file__) if '__file__' in globals() else Path("<undefined>.ipynb").resolve()

# input_size = (1, 3, 32, 32)
input_size = (1, 3, 224, 224)

base_network = tv(num_classes=100)

# EERefiner でラップし，メソッドチェーンで変更を適用．最後にbuild() 記述が面倒なら，関数つくるのもあり (下の例参照)
network = EERefiner(base_network).multi_narrow(div=2).init_weights().build()
network = EERefiner(base_network).cifar_style().multi_narrow(div=2).init_weights().build() ##### cifar_style() は最初に呼ぶのを推奨


# 関数化の例



p = Path(this_path.parent / "modelarc.txt")
network = network.to(device="cuda")
out = network(torch.randn(input_size).to(device="cuda"))
p.write_text(str(network.base_model))
