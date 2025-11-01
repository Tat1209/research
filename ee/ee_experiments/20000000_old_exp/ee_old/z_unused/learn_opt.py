import torch
import mlflow
import optuna

from datasets import Prep
from app.ee.trainer import Model, Ens
from trans import Trans

import utils

from models.resnet_git import resnet18 as net
model_arc = "gitresnet"

work_path = '/home/haselab/Documents/tat/Research/'

tr = Trans()
pr = Prep(root=f"{work_path}assets/datasets/", seed=0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

mlflow.set_tracking_uri(f'{work_path}mlruns/')
mlflow.set_experiment("scheduler_lr_gridsearch_v3")


def objective(trial):
    with mlflow.start_run(run_name=f"0") as run:

        mlflow.log_param("model_arc",   model_arc)
        mlflow.log_metric("max_lr",     max_lr := trial.suggest_categorical("max_lr", [10**(i / 10) for i in range(-20, 0, 2)]))
        mlflow.log_metric("epochs",     epochs := 100)
        mlflow.log_metric("batch_size", batch_size := trial.suggest_categorical("batch_size", [125, 250]))
        mlflow.log_metric("fils",       fils := 64)
        mlflow.log_metric("ensembles",  ensembles := 1)
        mlflow.log_metric("data_range", data_range := 1.0)
        mlflow.log_param("mixup",       mixup := False)

        mlflow.log_metric("iters_per_epoch", iters_per_epoch := len(pr.dl("cifar_train", tr.cf_gen, batch_size, in_range=data_range)))

        mlflow.log_param("train_trans", repr(trans := tr.cf_git))
        mlflow.log_param("mixup",   mixup := False)


        train_loader = pr.dl("cifar_train", trans, batch_size, in_range=data_range)
        val_loader = pr.dl("cifar_val", tr.cf_gen, batch_size)


        models = []
        for _ in range(ensembles):
            network = net(num_classes=100)
            loss_func = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
            sched_list = [
                (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch"),
                (torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=iters_per_epoch), "batch"),
            ]
            sched_index = trial.suggest_categorical("sched_tuple", [i for i in range(len(sched_list))])
            sched_tuple = sched_list[sched_index]


            model = Model(network, loss_func, optimizer=optimizer, sched_tuple=sched_tuple, device=device)
            models.append(model)
        ens = Ens(models)


        hp_dict = {"loss_func":repr(ens.models[0].loss_func), "optimizer":repr(ens.models[0].optimizer), "scheduler":utils.sched_repr(ens.models[0].sched_tuple[0])}
        mlflow.log_params(hp_dict)
        mlflow.log_metric("params", ens.count_params())
        mlflow.log_text(ens.models[0].arc_check(dl=train_loader), "model_structure.txt")

        for e in range(epochs):
            mlflow.log_metric("lr", ens.models[0].get_lr(), step=e+1)

            Loss, Acc = ens.me_train_1epoch(train_loader, mixup=mixup)
            vLoss, vAcc = ens.me_val_1epoch(val_loader)

            met_dict = {"epoch":e+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc}
            mlflow.log_metrics(met_dict, step=e+1)
            model.printlog(met_dict, e, epochs, itv=epochs/10)
            
    return vAcc

if __name__ == '__main__':
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name="Nyaaaaaaaaaaaa",
        direction='maximize',
        load_if_exists=True,
    )
    study.optimize(objective)


    
    
