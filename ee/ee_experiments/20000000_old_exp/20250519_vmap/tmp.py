import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

this_path = Path(__file__) if '__file__' in globals() else Path("<unknown>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

from datasets import Datasets
from run_manager import RunManager, RunsManager
from trainer import Network, Networks, Trainer, MultiTrainer, MergeEnsemble, MergeEnsembleMeta
from modules import CrossEntropyLossT
import utils

from models.resnet_git_ee import resnet18 as resnet18_git_ee

net = resnet18_git_ee
import torch
import time
from functorch import vmap

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 仮のモデル定義
# ユーザーの実装を以下で差し替えてください
# from your_module import net

# ベンチマーク共通設定
ensemble_size = 256
batch_size = 128
input_channels = 3
input_size = 32
repeat = 100

input_tensor = torch.randn(batch_size, input_channels, input_size, input_size).to(device)

# ===== ベンチマーク関数（時間 + メモリ） =====
def benchmark_model(model, input_tensor, repeat=100):
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)

    # CUDAメモリ統計リセット
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # ウォームアップ
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(input_tensor)
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.time() - start

    mem_peak = torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else 0  # MB
    return elapsed / repeat, mem_peak

# ===== vmap用関数 =====
def vmap_ensemble_forward(model, inputs, ensemble_size):
    repeated_inputs = inputs.unsqueeze(0).repeat(ensemble_size, 1, 1, 1, 1)  # (E, B, C, H, W)
    outputs = vmap(model)(repeated_inputs)
    return outputs

def benchmark_vmap(model_single, input_tensor, ensemble_size, repeat=100):
    model_single.eval()
    model_single.to(device)
    input_tensor = input_tensor.to(device)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # ウォームアップ
    with torch.no_grad():
        for _ in range(10):
            _ = vmap_ensemble_forward(model_single, input_tensor, ensemble_size)

    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = vmap_ensemble_forward(model_single, input_tensor, ensemble_size)
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.time() - start

    mem_peak = torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else 0  # MB
    return elapsed / repeat, mem_peak

# ===== モデルのインスタンス化と比較 =====
# vmap用単一モデル（groups=1）
model_single = net(num_classes=10, nb_fils=4, ee_groups=1)

# groups=Nモデル（フィルタ数をN倍、groups=N）
model_grouped = net(num_classes=10, nb_fils=4, ee_groups=ensemble_size)

# ===== 実行 =====
print("Running benchmark...")

time_vmap, mem_vmap = benchmark_vmap(model_single, input_tensor, ensemble_size, repeat)
time_group, mem_group = benchmark_model(model_grouped, input_tensor, repeat)

print(f"[vmap ensemble]")
print(f"  avg inference time: {time_vmap:.6f} sec")
print(f"  peak memory usage : {mem_vmap:.2f} MB")

print(f"[grouped conv ensemble]")
print(f"  avg inference time: {time_group:.6f} sec")
print(f"  peak memory usage : {mem_group:.2f} MB")

