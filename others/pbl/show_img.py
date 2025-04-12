import sys
from pathlib import Path

import torch
from torchvision import transforms

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets, dl
from run_manager import RunManager, RunsManager, RunViewer
from trainer import Model, MyMultiTrain
from trans import Trans

# from torchvision.models import resnet18 as net
from models.resnet_1ch import resnet18 as net

ds = Datasets(root=work_path / "assets/datasets/")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# exp_name = "exp_pre1"
exp_name = "exp_tmp"

epochs = 100
batch_size = 128

train_ds_str = "mnist_syn"
val_ds_str = "mnist_val"
num_classes = 10

org_trans = [transforms.ToTensor()]
# org_trans = [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 360))]
gen_trans = [transforms.ToTensor(), transforms.Grayscale(num_output_channels=1), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 360))]
val_trans = [transforms.ToTensor()]

org_ds = ds("mnist_train", org_trans)
gen_ds = ds(train_ds_str, gen_trans)

import numpy as np
from PIL import Image
from pathlib import Path

def format_image(img):
    print(img.shape)
    img = np.array(img)
    # img = np.array(img, dtype=np.uint8)
    img = Image.fromarray(img)

def ds_to_imgs(ds, num):
    imgs = []
    ds_it = iter(ds)
    for _ in range(num):
        img, label = next(ds_it)
        img = format_image(img)
        imgs.append(img)
    
def ds_to_folder(ds, num, path):
    path = Path(path)
    path.mkdir(parents=True ,exist_ok=True)
    ds_it = iter(ds)
    for i in range(num):
        img, label = next(ds_it)
        img = format_image(img)
        img_name = f"{i}_label.png"
        img.save(path / Path(img_name), format="png")

ds_to_folder(gen_ds, 100, "/home/haselab/Documents/tat/Research/app/pbl/gen_ds_imgs")
ds_to_folder(org_ds, 100, "/home/haselab/Documents/tat/Research/app/pbl/org_ds_imgs")
