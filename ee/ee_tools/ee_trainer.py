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

    @TimeLog("dur_train_core", mode="add")
    @TimeLog("dur_total_core", mode="add")
    def train_1batch(self, inputs, labels):
        # outputs = self.network(inputs)
        outputs, path_outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)
        preds, corr = self.eval_flow(outputs, labels)

        momentum_norm = self.get_momentum_norm()

        self.update_grad(loss)

        stats = {
            "batch_loss": loss.item() * len(inputs), 
            "batch_corr": corr, "batch_ndata": len(inputs)}
        path_stats = self.path_stats(path_outputs, labels)
        params_stats = {
            "param_norm": self.network.param_stat(stat_f=lambda p: p.norm(p=2).item()) * len(inputs),
            "param_norm_layer": {k: v * len(inputs) for k, v in self.network.param_stat_layer(stat_f=lambda p: p.norm(p=2).item()).items()},
            "grad_norm": self.network.grad_stat(stat_f=lambda g: g.norm(p=2).item(), incl_if=lambda p: p.grad is not None) * len(inputs),
            "grad_norm_layer": {k: v * len(inputs) for k, v in self.network.grad_stat_layer(stat_f=lambda g: g.norm(p=2).item(), incl_if=lambda p: p.grad is not None).items()},
        }
        params_stats |= {
            "g2w_ratio": params_stats["grad_norm"] / params_stats["param_norm"] * len(inputs),
            "momentum_norm": momentum_norm * len(inputs),
        }
        iter_stats = {
            "iter_ndata": len(inputs),
            "iter_loss": loss.item(),
            "iter_param_norm": self.network.param_stat(stat_f=lambda p: p.norm(p=2).item()),
            "iter_grad_norm": self.network.grad_stat(stat_f=lambda g: g.norm(p=2).item(), incl_if=lambda p: p.grad is not None),
            "iter_momentum_norm": momentum_norm,
        }

        return stats | path_stats | params_stats | iter_stats

    def train_agg(self, stats_l, mode=None):
        # stats_lは [dict_batch_1, dict_batch_2, ...] のようなリストが格納される．

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
        momentum_norm = sum(stats["momentum_norm"] for stats in stats_l) / total_ndata
        
        iter_ndata = [stats["iter_ndata"] for stats in stats_l]
        iter_loss = [stats["iter_loss"] for stats in stats_l]
        iter_param_norm = [stats["iter_param_norm"] for stats in stats_l]
        iter_grad_norm = [stats["iter_grad_norm"] for stats in stats_l]
        iter_momentum_norm = [stats["iter_momentum_norm"] for stats in stats_l]
        
        train_aux = {
            "param_norm": param_norm,
            "grad_norm": grad_norm,
            "param_norm_layer": param_norm_layer,
            "grad_norm_layer": grad_norm_layer,
            "g2w_ratio": g2w_ratio,
            "momentum_norm": momentum_norm,
            "train_loss_path": loss_path,
            "train_acc_path": acc_path,
            "iter_ndata": iter_ndata,
            "iter_loss": iter_loss,
            "iter_param_norm": iter_param_norm,
            "iter_grad_norm": iter_grad_norm,
            "iter_momentum_norm": iter_momentum_norm,
        }

        return loss, acc, train_aux

    # @TimeLog("dur_val_core", mode="add")
    @TimeLog("dur_total_core", mode="add")
    def val_1batch(self, inputs, labels):
        outputs, path_outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)
        preds, corr = self.eval_flow(outputs, labels)

        stats = {"batch_loss": loss.item() * len(inputs), "batch_corr": corr, "batch_ndata": len(inputs)}
        path_stats = self.path_stats(path_outputs, labels)

        return stats | path_stats

    def val_agg(self, stats_l, mode=None):
        total_ndata = sum(stats["batch_ndata"] for stats in stats_l)

        loss = sum(stats["batch_loss"] for stats in stats_l) / total_ndata
        acc = sum(stats["batch_corr"] for stats in stats_l) / total_ndata

        loss_path = [sum(x) / total_ndata for x in zip(*(s["batch_loss_path"] for s in stats_l))]
        acc_path = [sum(x) / total_ndata for x in zip(*(s["batch_corr_path"] for s in stats_l))]

        val_aux = {
            "val_loss_path": loss_path,
            "val_acc_path": acc_path
        }

        return loss, acc, val_aux
    
    def path_stats(self, path_outputs, labels):
        path_losses = [self.criterion(path_output, labels).detach().item() * len(labels) for path_output in path_outputs]
        path_corrs = [self.eval_flow(path_output, labels)[1] for path_output in path_outputs]
        
        path_stats = {"batch_loss_path": path_losses, "batch_corr_path": path_corrs}

        return path_stats
    