import sys
from pathlib import Path

import torch
from torchvision import transforms

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "research"), None))
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))



from datasets import Datasets
from run_manager import RunManager, RunsManager
from trainer import Trainer, MultiTrainer, MergeEnsemble
from modules import CrossEntropyLossT
import utils

from models.resnet_ee import resnet18 as offnet
from models.resnet_git_ee_def import resnet18 as gitnet_def
from models.resnet_git_ee import resnet18 as gitnet
from models.resnet_git_ee_bn import resnet18 as gitnet_bn
from models.resnet_git_ee_bn_def import resnet18 as gitnet_bn_def

fetch_ds = Datasets(root=work_path / "assets/datasets/")

base_train_ds = fetch_ds("cifar10_train")
base_val_ds = fetch_ds("cifar10_val")

train_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), base_train_ds.normalizer()]
val_trans = [transforms.ToTensor(), base_train_ds.normalizer()]

train_ds = base_train_ds.transform(train_trans).in_ndata(5000)
val_ds = base_val_ds.transform(val_trans).in_ndata(5000)

train_dl = train_ds.loader()
val_dl = val_ds.loader()

epochs = 100
# max_lr = 0.2
max_lr = 0.005
# exp_name = "exp_compare_unify_all"
exp_name = "exp_dbg"


for i in range(20):
    # for net in [gitnet_bn_def]:
    for net in [gitnet_def, gitnet, gitnet_bn, gitnet_bn_def]:
        network = net(num_classes=train_ds.fetch_classes(), nb_fils=4, ee_groups=64)
        loss_func = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(network.parameters(), lr=max_lr)
        optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
        scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        trainer = Trainer(network, loss_func, optimizer, scheduler_t, device)

        run_mgr = RunManager(exc_path=__file__, exp_name=exp_name)
        run_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")
        run_mgr.log_param("epochs", epochs)
        run_mgr.log_param("ensemble_type", "EE")
        run_mgr.log_param("params", trainer.count_params())
        run_mgr.log_param("optimizer", "SGD")

        for e in range(epochs):
            train_loss, train_acc = trainer.train_1epoch(train_dl)
            if e == 0:
                run_mgr.log_param("grad", trainer.count_params_with_grad())
            val_loss, val_acc = trainer.val_1epoch(val_dl)
            met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
            trainer.printmet(met_dict, e + 1, epochs, itv=epochs/5)
            
            run_mgr.log_metrics(met_dict, step=e + 1)
            run_mgr.log_metrics(trainer.timeinfo(), step=e + 1)
            run_mgr.log_metric("grad_mean", trainer.grad_mean(), step=e + 1)

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

        etrainer = MergeEnsemble(trainers, device)
        etrainer.loss_func = torch.nn.CrossEntropyLoss()
        # etrainer.optimizer = torch.optim.SGD([p for trainer in etrainer.trainers for p in trainer.network.parameters()], lr=max_lr)
        etrainer.optimizer = torch.optim.Adam([p for trainer in etrainer.trainers for p in trainer.network.parameters()], lr=max_lr)
        etrainer.scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

        run_mgr = RunManager(exc_path=__file__, exp_name=exp_name)
        run_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")
        run_mgr.log_param("epochs", epochs)
        run_mgr.log_param("ensemble_type", "ME")
        run_mgr.log_param("params", etrainer.count_params())
        run_mgr.log_param("optimizer", "SGD")
        for e in range(epochs):
            train_loss, train_acc = etrainer.train_1epoch(train_dl)
            if e == 0:
                run_mgr.log_param("grad", etrainer.count_params_with_grad())
            val_loss_t, val_acc_t = etrainer.val_1epoch(val_dl, incl_members=True)
            val_loss = val_loss_t[0]
            val_loss_em = val_loss_t[1]
            val_acc = val_acc_t[0]
            val_acc_em = val_acc_t[1]
            # val_loss, val_acc = etrainer.val_1epoch(val_dl)
            met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
            # met_dict = {"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "val_acc_em[0]": val_acc_em[0], "val_acc_em[1]": val_acc_em[1]}
            etrainer.printmet(met_dict, e + 1, epochs, itv=epochs/5)
            run_mgr.log_metrics(met_dict, step=e + 1)
            run_mgr.log_metrics(etrainer.timeinfo(), step=e + 1)
            run_mgr.log_metric("grad_mean", trainer.grad_mean(), step=e + 1)

            run_mgr.ref_stats(step=e + 1, itv=epochs/100, last_step=epochs)
            run_mgr.ref_results(step=e + 1, itv=epochs/100, last_step=epochs)




