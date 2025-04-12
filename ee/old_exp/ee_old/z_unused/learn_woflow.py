import torch
import polars as pl

from time import time

from datasets import Datasets
from app.ee.trainer import Model, Ens
from trans import Trans

from models.resnet_ee import resnet18 as net

working_dir = "/home/haselab/Documents/tat/"

tr = Trans()
pr = Datasets(root=f"{working_dir}assets/datasets/", seed=0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

save_path = f"{working_dir}app/ee/results_e/"

epochs = 200
learning_rate = 0.001  
batch_size = 400        

fils = 64
ensembles = 1
data_range = 1.0

iters = len(pr.dl("cifar_train", tr.cf_gen, batch_size, in_range=data_range))
models = []
for i in range(ensembles):
    network = net(nb_fils=fils, num_classes=100)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=iters)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)

    model = Model(network, loss_func, optimizer=optimizer, scheduler=scheduler, device=device)
    models.append(model)
    
ens = Ens(models)
for e in range(epochs):
    # Loss, Acc = ens.me_train_1epoch(pr.dl("cifar_train", tr.cf_raug(3), batch_size, in_range=data_range), mixup=False, sched_iter=False)
    Loss, Acc = ens.me_train_1epoch(pr.dl("cifar_train", tr.cf_crop, batch_size, in_range=data_range), mixup=False, sched_iter=False)
    vLoss, vAcc = ens.me_val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size))
    timestamp = time()

    met_dict = {"epoch":e+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc, "timestamp":timestamp}
    model.log_met(met_dict)
    model.printlog(met_dict, e, epochs, itv=1) # itv = epochs

    # 初回だけ保存
    if ni == 0: model.hist_to_csv(f"{save_path}{run_name}_1train.csv")
    
    # metrix保存 dfには学習後metrixが試行ごとに記録される
    # if ni == 0: df = model.get_last_met()
    # else: df = pl.concat([df, model.get_last_met()])
    # df = pl.concat([df, model.get_last_met()])
        
        
