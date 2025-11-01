import sys

import torch
import mlflow

sys.path.append("/home/haselab/Documents/tat/Research/app/torch_libs")
from datasets import Datasets, dl
from trainer import Model, Ens, EEModel
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

work_path = '/home/haselab/Documents/tat/Research/'
ds = Datasets(root=f"{work_path}assets/datasets/")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

mlflow.set_tracking_uri(f'{work_path}mlruns/')
mlflow.set_experiment("datas_fils_same_train")
# mlflow.set_experiment("tmp")


for fi in [1]:
# for fi in [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 128, 192, 256]:
    # for di in [0.002, 0.004, 0.006, 0.008, 0.01, 0.014, 0.02, 0.03, 0.05, 0.074, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]:
    # for di in [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]:
    for di in [0.0025]:
        with mlflow.start_run(run_name=f"{fi}") as run:

            mlflow.log_metric("max_lr",     max_lr := 0.005)   # Adam
            mlflow.log_metric("epochs",     epochs := int(100 / di))
            mlflow.log_metric("batch_size", batch_size := 125)
            mlflow.log_metric("fils",       fils := fi)
            mlflow.log_metric("ensembles",  ensembles := 1)

            mlflow.log_param("ensemble_type",   ensemble_type := ['easy', 'merge', 'pure'][0])
            mlflow.log_param("mixup",       mixup := False)
            mlflow.log_param("train_trans", repr(train_trans := Trans.cf_git))
            mlflow.log_param("val_trans",   repr(val_trans := Trans.cf_gen))

            train_loader = dl(ds("cifar10_train", train_trans, label_balance=True).in_range(di), batch_size, shuffle=True)
            val_loader = dl(ds("cifar10_val", val_trans), batch_size, shuffle=True)

            mlflow.log_metric("num_data", len(train_loader.dataset))
            mlflow.log_metric("iters/epoch", iters_per_epoch := len(train_loader))
            mlflow.log_param("dataset", train_loader.dataset.ds_name)
            mlflow.log_param("model_arc", model_arc)

            network = net(num_classes=10, nb_fils=fils, ee_groups=ensembles)
            loss_func = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
            scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

            model = EEModel(network, loss_func, optimizer=optimizer, scheduler_t=scheduler_t, device=device)

            mlflow.log_metric("params", model.count_params())
            hp_dict = {"loss_func":repr(model.loss_func), "optimizer":repr(model.optimizer), "scheduler":utils.sched_repr(model.scheduler_t[0])}
            mlflow.log_params(hp_dict)
            mlflow.log_text(repr(model.network), "model_layers.txt")
            mlflow.log_text(model.arc_check(dl=train_loader), "model_structure.txt")


            print(f"{di=}, {fi=}, {batch_size=}")

            for e in range(epochs):
                mlflow.log_metric("lr", model.get_lr(), step=e+1)

                train_loss, train_acc = model.train_1epoch(train_loader, mixup=mixup)
                val_loss, val_acc = model.val_1epoch(val_loader)

                met_dict = {"epoch":e+1, "train_loss":train_loss, "train_acc":train_acc, "val_loss":val_loss, "val_acc":val_acc}

                mlflow.log_metrics(met_dict, step=e+1)
                model.printlog(met_dict, e+1, epochs, itv=epochs/4)

            # model.mlflow_save_sd(mlflow)

