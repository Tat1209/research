import torch
import mlflow

from datasets import Prep
from app.ee.trainer import Model
from trans import Trans

from models.resnet_git import resnet18 as net
model_arc = "gitresnet"

from torch.optim.lr_scheduler import _LRScheduler
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


work_path = '/home/haselab/Documents/tat/Research/'

tr = Trans()
pr = Prep(root=f"{work_path}assets/datasets/", seed=0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

mlflow.set_tracking_uri(f'{work_path}mlruns/')
mlflow.set_experiment("git_reap")

with mlflow.start_run(run_name=f"tmp_run") as run:
    mlflow.log_param("max_lr",      max_lr := 1e-1)
    mlflow.log_param("epochs",      epochs := 200)
    mlflow.log_param("batch_size",  batch_size := 128)
    mlflow.log_param("trans",       repr(trans := tr.cf_git))

    train_loader = pr.dl("cifar_train", trans, batch_size)
    val_loader = pr.dl("cifar_val", tr.cf_gen, batch_size)

    mlflow.log_param("iters_per_epoch", iters_per_epoch := len(pr.dl("cifar_train", tr.cf_gen, batch_size)))
    mlflow.log_param("model_arc", model_arc)

    mlflow.log_param("train_trans", repr(trans := tr.cf_git))

    network = net(num_classes=100)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)

    model = Model(network, loss_func, optimizer=optimizer, sched_tuple=None, device=device)


    hp_dict = {"loss_func":repr(model.loss_func), "optimizer":repr(model.optimizer), "params":model.count_params()}
    mlflow.log_params(hp_dict)
    mlflow.log_text(model.arc_check(dl=train_loader), "model_structure.txt")

    for e in range(epochs):
        if e == 0:
            scheduler = WarmUpLR(optimizer, iters_per_epoch * 1)
            model.sched_tuple = (scheduler, "batch")
        elif e == 1:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=1)
            model.sched_tuple = (scheduler, "epoch")

        mlflow.log_metric("lr", model.get_lr(), step=e+1)

        Loss, Acc = model.train_1epoch(train_loader, mlflow_obj=mlflow, log_batch_lr=True)
        vLoss, vAcc = model.val_1epoch(val_loader)

        met_dict = {"epoch":e+1, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc}
        mlflow.log_metrics(met_dict, step=e+1)
        model.printlog(met_dict, e, epochs, itv=epochs/10)

    
    
