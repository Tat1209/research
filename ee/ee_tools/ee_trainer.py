import math
import sys
from pathlib import Path

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

    def agg_epoch(self, stats_l, mode=None):
        total_loss = sum(stats["batch_loss"] for stats in stats_l)
        total_corr = sum(stats["batch_corr"] for stats in stats_l)
        total_ndata = sum(stats["batch_ndata"] for stats in stats_l)

        total_loss_path = [sum(x) for x in zip(*(s["batch_loss_path"] for s in stats_l))]
        total_corr_path = [sum(x) for x in zip(*(s["batch_corr_path"] for s in stats_l))]
        
        loss = total_loss / total_ndata
        acc = total_corr / total_ndata
        loss_path = [path_loss / total_ndata for path_loss in total_loss_path]
        corr_path = [path_corr / total_ndata for path_corr in total_corr_path]
        
        return loss, acc, loss_path, corr_path
    
    @TimeLog("dur_train_core", mode="add")
    @TimeLog("dur_total_core", mode="add")
    def train_1batch(self, inputs, labels):
        # outputs = self.network(inputs)
        outputs, path_outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)
        preds, corr = self.eval_flow(outputs, labels)

        self.update_grad(loss)

        stats = {"batch_loss": loss.item() * len(inputs), "batch_corr": corr, "batch_ndata": len(inputs)}
        path_stats = self.path_stats(path_outputs, labels)

        return stats.copy() | path_stats.copy()

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
    