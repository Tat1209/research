import sys
from pathlib import Path

import torch
from torchvision import transforms

this_path = Path(__file__) if '__file__' in globals() else Path("<unknown>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

from datasets import Datasets
from run_manager import RunManager, RunsManager
from trainer import Trainer, MultiTrainer, MergeEnsemble
from modules import CrossEntropyLossT
from hook import HookManager
import utils

from models.resnet_ee import resnet18 as offnet
from models.resnet_git_ee_def import resnet18 as gitnet_def
from models.resnet_git_ee import resnet18 as gitnet
from models.resnet_git_ee_bn import resnet18 as gitnet_bn
from models.resnet_git_ee_test import resnet18 as gitnet_test

fetch_ds = Datasets(root=work_path / "assets/datasets/")

base_train_ds = fetch_ds("cifar10_train")
base_val_ds = fetch_ds("cifar10_val")

train_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), base_train_ds.normalizer()]
val_trans = [transforms.ToTensor(), base_train_ds.normalizer()]

train_ds = base_train_ds.transform(train_trans).in_ndata(5000)
val_ds = base_val_ds.transform(val_trans).in_ndata(5000)

train_dl = train_ds.loader()
val_dl = val_ds.loader()

epochs = 1
max_lr = 0.005
# exp_name = "exp_compare_unify_all"
exp_name = "exp_dbg"


for net in [gitnet]:
# for net in [gitnet_def, gitnet, gitnet_bn]:
    network = net(num_classes=train_ds.fetch_classes(), nb_fils=4, ee_groups=64)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
    scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    ee_trainer = Trainer(network, loss_func, optimizer, scheduler_t, device)

    run_mgr = RunManager(exc_path=this_path, exp_name=exp_name)
    run_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")
    run_mgr.log_param("epochs", epochs)
    run_mgr.log_param("ensemble_type", "EE")
    run_mgr.log_param("params", ee_trainer.count_params())
    run_mgr.log_text("model_torchinfo.txt", ee_trainer.torchinfo(dl=train_dl, batch_dim=None))

    for e in range(epochs):
        with HookManager(ee_trainer.network) as hm:
            hm.register_backward(module=ee_trainer.network.fc[3], name='final_layer', fn=lambda module, input, output: input[0].detach().cpu())
            train_loss, train_acc = ee_trainer.train_1epoch(train_dl)
            tsr_l = hm.get('final_layer')
        if e == 0:
            run_mgr.log_param("grad", ee_trainer.count_params_with_grad())
        val_loss, val_acc = ee_trainer.val_1epoch(val_dl)
        met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        ee_trainer.printmet(met_dict, e + 1, epochs, itv=epochs/5)
        
        run_mgr.log_metrics(met_dict, step=e + 1)
        run_mgr.log_metrics(ee_trainer.timeinfo(), step=e + 1)
        run_mgr.log_metric("grad_mean", ee_trainer.grad_mean(), step=e + 1)

        run_mgr.ref_stats(step=e + 1, itv=epochs/100, last_step=epochs)
        run_mgr.ref_results(step=e + 1, itv=epochs/100, last_step=epochs)

    trainers = []
    for i in range(64):
        network = net(num_classes=train_ds.fetch_classes(), nb_fils=4)
        # optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
        # scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        trainer = Trainer(network, loss_func, optimizer, scheduler_t, device)
        trainers.append(trainer)

    me_trainers = MergeEnsemble(trainers, device)
    me_trainers.loss_func = torch.nn.CrossEntropyLoss()
    me_trainers.optimizer = torch.optim.Adam([p for trainer in me_trainers.trainers for p in trainer.network.parameters()], lr=max_lr)
    me_trainers.scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

    run_mgr = RunManager(exc_path=this_path, exp_name=exp_name)
    run_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")
    run_mgr.log_param("epochs", epochs)
    run_mgr.log_param("ensemble_type", "ME")
    run_mgr.log_param("params", me_trainers.count_params())
    for e in range(epochs):
        train_loss, train_acc = me_trainers.train_1epoch(train_dl)
        if e == 0:
            run_mgr.log_param("grad", me_trainers.count_params_with_grad())
        val_loss_t, val_acc_t = me_trainers.val_1epoch(val_dl, incl_members=True)
        val_loss = val_loss_t[0]
        val_loss_em = val_loss_t[1]
        val_acc = val_acc_t[0]
        val_acc_em = val_acc_t[1]
        # val_loss, val_acc = me_trainers.val_1epoch(val_dl)
        met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        # met_dict = {"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "val_acc_em[0]": val_acc_em[0], "val_acc_em[1]": val_acc_em[1]}
        me_trainers.printmet(met_dict, e + 1, epochs, itv=epochs/5)
        run_mgr.log_metrics(met_dict, step=e + 1)
        run_mgr.log_metrics(me_trainers.timeinfo(), step=e + 1)
        run_mgr.log_metric("grad_mean", me_trainers.grad_mean(), step=e + 1)

        run_mgr.ref_stats(step=e + 1, itv=epochs/100, last_step=epochs)
        run_mgr.ref_results(step=e + 1, itv=epochs/100, last_step=epochs)


    sd_a = ee_trainer.network.state_dict()
    sd_b = me_trainers.trainers[0].network.state_dict()
    
    print(net.__name__)

    for key in sd_a.keys():
        tsr_a = sd_a[key].type(torch.float32)
        tsr_b = sd_b[key].type(torch.float32)
        
        try:
            tsr_a = tsr_a[:len(tsr_a) // 64]
        except (IndexError, TypeError):
            pass
        
        a = tsr_a.var()
        b = tsr_b.var()
        # a = tsr_a.abs().mean()
        # b = tsr_b.abs().mean()
        
        if a > 0.000001 and b > 0.000001 and a > 0.000001: 
            print(f"{key:40}: {a:8.6e}, {b:8.6e}, {b/a:8.4f}")
            # print(f"{key:50}: {a.var():8.6e}, {b.var():8.6e}, {b.var()/a.var():8.4f}")
            # print(f"{key:50}: {tsr_a.shape}, {tsr_b.shape}")
    




