import os
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

from datasets import fetch_dataset

# ---------------------------------------------------------
target_datasets = [
    "cifar10_train",
    "cifar100_train",
    "stl10_train",
    "tiny-imagenet_train",
    "mnist_train",
    "fashion-mnist_train",
    "svhn_train",
    "caltech101_trainval",
    "oxford-pet_trainval",
    "flowers102_train",
    "cub200_train",
    "stanford-cars_train"
]

ds_root = work_path / "assets/datasets/"

print(f"\nDownload Destination: {ds_root}")
os.makedirs(ds_root, exist_ok=True)

print("Starting download sequence...\n")

for ds_name in target_datasets:
    print(f"--- Processing: {ds_name} ---")
    try:
        # fetch_dataset は内部で FileLock を使い、
        # 必要なら download=True で _datasets を呼び出します
        dset = fetch_dataset(str(ds_root), ds_name)
        
        # 成功確認
        if dset is not None:
            print(f"✔ OK: {ds_name} (Length: {len(dset)})")
        else:
            print(f"⚠ Warning: {ds_name} returned None (Check dataset.py logic)")
            
    except Exception as e:
        print(f"✖ Failed: {ds_name}")
        print(f"  Error: {e}")

print("All downloads processed.")
