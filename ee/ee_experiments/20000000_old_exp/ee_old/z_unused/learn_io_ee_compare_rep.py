import torch
import mlflow

from datasets import Prep
from app.ee.trainer import Model, Ens
from trans import Trans

import utils

import random
import numpy as np

# from models.myresnet import resnet18 as net
# model_arc = "myresnet"
# from torchvision.models import resnet18 as net
# model_arc = "official"
from models.resnet_git_ee import resnet18 as net
# from models.gitresnet_ee_only import resnet18 as net
model_arc = "gitresnet_ee"

work_path = '/home/haselab/Documents/tat/Research/'

pr = Prep(root=f"{work_path}assets/datasets/", seed=0)
# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

mlflow.set_tracking_uri(f'{work_path}mlruns/')
mlflow.set_experiment("tmp")
# mlflow.set_experiment("fil_ens")

fi = 16
epochs = 1

mod_1 = (0, 0, 0)
mod_2 = (9, 0, 0)

p_val = 0.05

with mlflow.start_run(run_name=f"{fi}") as run:
    mlflow.log_metric("max_lr",     max_lr := 0.1)
    mlflow.log_metric("epochs",     epochs)
    mlflow.log_metric("batch_size", batch_size := 125)
    mlflow.log_metric("fils",       fils := fi)
    mlflow.log_metric("ensembles",  ensembles := (64 // fi) ** 2)
    mlflow.log_param("ensemble_type",   ensemble_type := ['merge', 'pure'][0])

    print(f"{fi}fils, {ensembles}ensembles, {ensemble_type}")

    mlflow.log_param("data_range",  data_range := 1)
    mlflow.log_param("mixup",       mixup := False)
    mlflow.log_param("train_trans", repr(train_trans := Trans.cf_git))
    mlflow.log_param("val_trans", repr(val_train := Trans.cf_gen))

    utils.torch_fix_seed()
    train_loader = pr.dl("cifar_train", train_trans, batch_size, in_range=data_range)
    # val_loader = pr.dl("cifar_val", val_train, batch_size, in_range=(0, 1.0))
    val_loader = pr.dl("fix_rand", val_train, batch_size, in_range=(0, 1.0))

    mlflow.log_metric("iters_per_epoch", iters_per_epoch := len(train_loader))
    mlflow.log_param("model_arc", model_arc)

    utils.torch_fix_seed()
    models = []
    for _ in range(ensembles):
        network = net(num_classes=100, nb_fils=fils)
        for p in network.parameters(): p.data.fill_(p_val)
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
        sched_tuple = (scheduler, "epoch")

        model = Model(network, loss_func, optimizer=optimizer, sched_tuple=sched_tuple, device=device)
        models.append(model)
    ens2 = Ens(models, device=device)

    utils.torch_fix_seed()
    models = []
    for _ in range(1):
        network = net(num_classes=100, nb_fils=fils, ee_groups=ensembles)
        # for p in network.parameters(): p.data.fill_(p_val)
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
        sched_tuple = (scheduler, "epoch")

        model = Model(network, loss_func, optimizer=optimizer, sched_tuple=sched_tuple, device=device)
        models.append(model)
    ens1 = Ens(models, device=device)

    utils.copy_params_ee(ens2.models, ens1.models[0])
    
    p_net1 = utils.get_patial_net(ens2.models[0].network, mod_1, mod_2)


    # ens2.load_state_dict('tmp.pkl')

    hp_dict = {"loss_func":repr(ens2.models[0].loss_func), "optimizer":repr(ens2.models[0].optimizer), "scheduler":utils.sched_repr(ens2.models[0].sched_tuple[0])}
    mlflow.log_params(hp_dict)
    mlflow.log_metric("params", ens2.count_params())
    mlflow.log_text(repr(ens2.models[0].network), "model_layers.txt")
    mlflow.log_text(ens2.models[0].arc_check(dl=train_loader), "model_structure.txt")


    utils.torch_fix_seed()
    for e in range(epochs):
        mlflow.log_metric("lr", ens2.models[0].get_lr(), step=e+1)

        # Loss, Acc = ens2.me_train_1epoch(train_loader, mixup=mixup)
        vLoss, vAcc, vout2 = ens2.val_1epoch_tmp(val_loader)

        # met_dict = {"epoch":e+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc}
        met_dict = {"epoch":e+1, "vLoss":vLoss, "vAcc":vAcc}

        b_vLoss = vLoss

        mlflow.log_metrics(met_dict, step=e+1)
        model.printlog(met_dict, e, epochs, itv=1)


    # ens2.save_state_dict('tmp.pkl')


    utils.torch_fix_seed()

    mlflow.log_metric("max_lr",     max_lr := 0.1)
    mlflow.log_metric("epochs",     epochs)
    mlflow.log_metric("batch_size", batch_size := 125)
    mlflow.log_metric("fils",       fils := fi)
    mlflow.log_metric("ensembles",  ensembles := (64 // fi) ** 2)
    mlflow.log_param("ensemble_type",   ensemble_type := ['merge', 'pure'][0])

    print(f"{fi}fils, {ensembles}ensembles, {ensemble_type}")

    mlflow.log_param("data_range",  data_range := 1)
    mlflow.log_param("mixup",       mixup := False)
    mlflow.log_param("train_trans", repr(train_trans := Trans.cf_git))
    mlflow.log_param("val_trans", repr(val_train := Trans.cf_gen))

    train_loader = pr.dl("cifar_train", train_trans, batch_size, in_range=data_range)
    # val_loader = pr.dl("cifar_val", val_train, batch_size, in_range=(0, 1.0))
    val_loader = pr.dl("fix_rand", val_train, batch_size, in_range=(0, 1.0))

    mlflow.log_metric("iters_per_epoch", iters_per_epoch := len(train_loader))
    mlflow.log_param("model_arc", model_arc)



    p_net2 = utils.get_patial_net(ens1.models[0].network, mod_1, mod_2)

    # ens1.load_state_dict('tmp.pkl')

    hp_dict = {"loss_func":repr(ens1.models[0].loss_func), "optimizer":repr(ens1.models[0].optimizer), "scheduler":utils.sched_repr(ens1.models[0].sched_tuple[0])}
    mlflow.log_params(hp_dict)
    mlflow.log_metric("params", ens1.count_params())
    mlflow.log_text(repr(ens1.models[0].network), "model_layers.txt")
    mlflow.log_text(ens1.models[0].arc_check(dl=train_loader), "model_structure.txt")


    utils.torch_fix_seed()
    for e in range(epochs):
        mlflow.log_metric("lr", ens1.models[0].get_lr(), step=e+1)

        # Loss, Acc = ens1.me_train_1epoch(train_loader, mixup=mixup)
        vLoss, vAcc, vout1 = ens1.val_1epoch_tmp(val_loader)
        

        met_dict = {"epoch":e+1, "vLoss":vLoss, "vAcc":vAcc}
        a_vLoss = vLoss
        # met_dict = {"epoch":e+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc}
        mlflow.log_metrics(met_dict, step=e+1)
        model.printlog(met_dict, e, epochs, itv=1)


    # ens1.save_state_dict('tmp.pkl')

    ens2.mlflow_save_state_dict(mlflow)
                
                    
                        
# print(vout2 - vout1)
print(torch.sum(torch.abs(vout2 - vout1)) / torch.sum(torch.abs(vout1) + torch.abs(vout2)))
# print(b_vLoss - a_vLoss)



    # sha = (1, 3, 32, 32)

    # x = torch.arange(sha[2] * sha[3]).float()
    # x = x.view(sha[2], sha[3])

    # x = torch.stack([x for _ in range(sha[1])], dim=0).to(device)
    # # y = torch.cat([x for _ in range(ensembles)], dim=0).to(device)
    # y = x.clone().detach()

    # x = torch.stack([x for _ in range(sha[0])], dim=0).to(device)
    # y = torch.stack([y for _ in range(sha[0])], dim=0).to(device)

    # print(x.shape)
    # print(y.shape)

    # out1 = p_net1(y)
    # out2 = p_net2(x)
    
    # # out1 = out1[:,:out1.size(1)//ensembles,:,:]                                
    
    # print(out1)
    # print(out1.shape)
    # print(out2)
    # print(out2.shape)

# # print(vout2 - vout1)
# print(torch.sum(torch.abs(out2 - out1)) / torch.sum(torch.abs(out1) + torch.abs(out2)))
# # print(b_vLoss - a_vLoss)