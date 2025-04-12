import sys

import torch

work_path = "/home/haselab/Documents/tat/Research/"
sys.path.append(f"{work_path}app/torch_libs/")

from datasets import Datasets, dl
from run_manager_old import RunManager, RunsManager
from trainer import Model, Ens, EEModel, MultiTrain
from trans import Trans
import utils

# from models.resnet_ee import resnet18 as net
# model_arc = "official_ee"
# from torchvision.models import resnet18 as net
# model_arc = "official"
from models.resnet_git_ee import resnet18 as net

model_arc = "gitresnet_ee"
# from torchvision.models import efficientnet_v2_s as net
# model_arc = "efficientnet_v2_s"


ds = Datasets(root=f"{work_path}assets/datasets/")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


for fi in [[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 8, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 16, 16, 16, 20, 20, 20, 24, 24, 28, 28, 32, 32, 40, 48, 64, 80, 96, 128]]:
    for di in [0.0075, 0.005, 0.0025]:
        # for fi in [[1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 128]]:
        #     for di in [1.0, 0.75, 0.5, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.0025]:
        # runs = [RunManager(exc_path=__file__, exp_name="exp_tmp") for _ in fi]
        runs = [RunManager(exc_path=__file__, exp_name="exp_fil_data") for _ in fi]
        runs_mgr = RunsManager(runs)

        runs_mgr.log_param("max_lr", max_lr := 0.005)  # Adam
        runs_mgr.log_param("epochs", epochs := int(100 / di))
        runs_mgr.log_param("batch_size", batch_size := 125)
        runs_mgr.log_param("fils", fils := fi)
        runs_mgr.log_param("ensembles", ensembles := 1)

        runs_mgr.log_param("ensemble_type", ensemble_type := ["easy", "merge", "pure"][0])
        runs_mgr.log_param("mixup", mixup := False)
        runs_mgr.log_param("train_trans", repr(train_trans := Trans.cf_git))
        runs_mgr.log_param("val_trans", repr(val_trans := Trans.cf_gen))

        train_loader = dl(ds("cifar10_train", train_trans, label_balance=True).in_range(di), batch_size, shuffle=True)
        val_loader = dl(ds("cifar10_val", val_trans), batch_size, shuffle=True)

        runs_mgr.log_param("num_data", len(train_loader.dataset))
        runs_mgr.log_param("iters/epoch", iters_per_epoch := len(train_loader))
        runs_mgr.log_param("dataset", train_loader.dataset.ds_name)
        runs_mgr.log_param("model_arc", model_arc)

        models = []
        for fils in fi:
            network = net(num_classes=10, nb_fils=fils, ee_groups=ensembles)
            loss_func = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
            scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

            model = EEModel(network, loss_func, optimizer=optimizer, scheduler_t=scheduler_t, device=device)
            models.append(model)
        mmodel = MultiTrain(models, device)

        runs_mgr.log_param("params", [model.count_params() for model in mmodel.models])
        hp_dict = {
            "loss_func": [repr(model.loss_func) for model in mmodel.models],
            "optimizer": [repr(model.optimizer) for model in mmodel.models],
            "scheduler": [utils.sched_repr(model.scheduler_t[0]) for model in mmodel.models],
        }
        runs_mgr.log_params(hp_dict)
        runs_mgr.log_text([repr(model.network) for model in mmodel.models], "model_layers.txt")
        runs_mgr.log_text([model.arc_check(dl=train_loader) for model in mmodel.models], "model_structure.txt")

        print(f"{di=}, {fi=}, {sum(fi)=}, {batch_size=}")

        for e in range(epochs):
            runs_mgr.log_metric("lr", [model.get_lr() for model in mmodel.models], step=e + 1)

            train_loss, train_acc = mmodel.train_1epoch(train_loader, mixup=mixup)
            val_loss, val_acc = mmodel.val_1epoch(val_loader)

            met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}

            runs_mgr.log_metrics(met_dict, step=e + 1)
            mmodel.printlog(met_dict, e + 1, epochs, itv=epochs / 4)
            runs_mgr.ref_stats(itv=5, step=e + 1, last_step=epochs)

        runs_mgr.ref_stats()
        runs_mgr.log_torch_save([model.get_sd() for model in mmodel.models], "state_dict.pt")
