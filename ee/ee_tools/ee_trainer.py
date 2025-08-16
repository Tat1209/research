import math
import sys
from pathlib import Path
from collections import Counter

import torch
from torchvision.transforms import v2 as transforms

this_path = Path(__file__) if '__file__' in globals() else Path("<unknown>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

from trainer import Trainer
from utils import TimeLog

@TimeLog()
class EETrainer(Trainer):
    def __init__(self, network, criterion, optimizer, scheduler=None, device=None):
        super().__init__(network, criterion, optimizer, scheduler, device)

    def train_agg(self, stats_l, mode=None):
        total_ndata = sum(stats["batch_ndata"] for stats in stats_l)

        loss = sum(stats["batch_loss"] for stats in stats_l) / total_ndata
        acc = sum(stats["batch_corr"] for stats in stats_l) / total_ndata

        loss_path = [sum(x) / total_ndata for x in zip(*(s["batch_loss_path"] for s in stats_l))]
        acc_path = [sum(x) / total_ndata for x in zip(*(s["batch_corr_path"] for s in stats_l))]

        param_norm = sum(stats["param_norm"] for stats in stats_l) / total_ndata
        grad_norm = sum(stats["grad_norm"] for stats in stats_l) / total_ndata
        
        param_norm_layer = {k: v / total_ndata for k, v in sum((Counter(s["param_norm_layer"]) for s in stats_l), Counter()).items()}
        grad_norm_layer = {k: v / total_ndata for k, v in sum((Counter(s["grad_norm_layer"]) for s in stats_l), Counter()).items()}

        g2w_ratio = sum(stats["g2w_ratio"] for stats in stats_l) / total_ndata
        last_momentum_norm = sum(stats["last_momentum_norm"] for stats in stats_l) / total_ndata
        
        aux_stats = {
            "param_norm": param_norm,
            "grad_norm": grad_norm,
            "param_norm_layer": param_norm_layer,
            "grad_norm_layer": grad_norm_layer,
            "g2w_ratio": g2w_ratio,
            "last_momentum_norm": last_momentum_norm,
            "loss_path": loss_path,
            "acc_path": acc_path
        }

        return loss, acc, aux_stats
        # return loss, acc, loss_path, acc_path

    def val_agg(self, stats_l, mode=None):
        total_ndata = sum(stats["batch_ndata"] for stats in stats_l)

        loss = sum(stats["batch_loss"] for stats in stats_l) / total_ndata
        acc = sum(stats["batch_corr"] for stats in stats_l) / total_ndata

        return loss, acc
        # return loss, acc, loss_path, acc_path
    
    @TimeLog("dur_train_core", mode="add")
    @TimeLog("dur_total_core", mode="add")
    def train_1batch(self, inputs, labels):
        # outputs = self.network(inputs)
        outputs, path_outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)
        preds, corr = self.eval_flow(outputs, labels)

        last_momentum_norm = self.get_momentum_norm()

        self.update_grad(loss)

        stats = {"batch_loss": loss.item() * len(inputs), "batch_corr": corr, "batch_ndata": len(inputs)}
        path_stats = self.path_stats(path_outputs, labels)
        params_stats = {
            "param_norm": self.network.param_stat(stat_f=lambda p: p.norm(p=2).item()) * len(inputs),
            "param_norm_layer": {k: v * len(inputs) for k, v in self.network.param_stat_layer(stat_f=lambda p: p.norm(p=2).item()).items()},
            "grad_norm": self.network.grad_stat(stat_f=lambda g: g.norm(p=2).item(), incl_if=lambda p: p.grad is not None) * len(inputs),
            "grad_norm_layer": {k: v * len(inputs) for k, v in self.network.grad_stat_layer(stat_f=lambda g: g.norm(p=2).item(), incl_if=lambda p: p.grad is not None).items()},
        }

        params_stats |= {
            "g2w_ratio": params_stats["grad_norm"] / params_stats["param_norm"] * len(inputs),
            "last_momentum_norm": last_momentum_norm * len(inputs),
        }

        return stats.copy() | path_stats.copy() | params_stats.copy()

    # @TimeLog("dur_val_core", mode="add")
    @TimeLog("dur_total_core", mode="add")
    def val_1batch(self, inputs, labels):
        outputs, path_outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)
        preds, corr = self.eval_flow(outputs, labels)

        stats = {"batch_loss": loss.item() * len(inputs), "batch_corr": corr, "batch_ndata": len(inputs)}
        path_stats = self.path_stats(path_outputs, labels)

        return stats.copy() | path_stats.copy()
 
    def path_stats(self, path_outputs, labels):
        path_losses = [self.criterion(path_output, labels).detach().item() * len(labels) for path_output in path_outputs]
        path_corrs = [self.eval_flow(path_output, labels)[1] for path_output in path_outputs]
        
        path_stats = {"batch_loss_path": path_losses, "batch_corr_path": path_corrs}
        return path_stats
    