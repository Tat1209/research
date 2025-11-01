import utils

# from models.resnet_hase import resnet18 as hasenet
# from models.resnet_hase_2d import ResNet18 as hasenet
from models.resnet_git import resnet18 as tananet
# from models.resnet_ee import resnet18 as tananet
from models.resnet_dd import make_resnet18k as net


hase = tananet(num_classes=100)
utils.arc_check(network=hase, out_file=True, file_name="tananet.txt", input_size=(100, 3, 32, 32))

# hase = hasenet(num=100)
# utils.arc_check(network=hase, out_file=True, file_name="hasenet.txt", input_size=(100, 3, 32, 32))

hase = net(num_classes=100)
utils.arc_check(network=hase, out_file=True, file_name="ddnet.txt", input_size=(100, 3, 32, 32))

