import sys
import math
from pathlib import Path

import torch
from torchvision import transforms

this_path = Path(__file__) if '__file__' in globals() else Path("<unknown>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

from datasets import Datasets
from run_manager import RunManager, RunsManager
from trainer import Network, Networks, Trainer, MultiTrainer, MergeEnsemble
from modules import CrossEntropyLossT
import utils

from models.resnet_ee import resnet18 as resnet18_ee
from models.resnet_git_ee import resnet18 as resnet18_git_ee
from models.resnet_git_ee import resnet50 as resnet50_git_ee

src_text, src_name= utils.get_source(with_name=True)

fetch_ds = Datasets(root=work_path / "assets/datasets/")

# exp_name = "exp_opt"
exp_name = "exp_tmp"

net = resnet18_git_ee
base_epochs = 10
base_ndata = 5000
max_lr = 0.1
batch_size = 128
ndata = base_ndata
# base_epochs = 1000
# base_ndata = 10000
# max_lr_l = [0.005]
# batch_size = 128
# ndata_l = [10000, 5000, 3000, 2000, 1000]

train_ds_str = "cifar100_train"
val_ds_str = "cifar100_val"

fils = 4
ensembles = 64
# fil_ens_l = [(32, 1), (4, 64)]
# # fil_ens_l = [(32, 1), (16, 4), (8, 16), (4, 64)]
# fils_l, ensembles_l = map(list, zip(*fil_ens_l))


base_train_ds = fetch_ds(train_ds_str)
base_val_ds = fetch_ds(val_ds_str)

match train_ds_str:
    case "cifar10_train":
        train_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), base_train_ds.normalizer()]
        val_trans = [transforms.ToTensor(), base_train_ds.normalizer()]
    case "cifar100_train":
        train_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), base_train_ds.normalizer()]
        val_trans = [transforms.ToTensor(), base_train_ds.normalizer()]
    case "stl10_train":
        train_trans = [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 360)), base_train_ds.normalizer()]
        val_trans = [transforms.ToTensor(), base_train_ds.normalizer()]
    case _:
        pass

base_train_ds = base_train_ds.transform(train_trans)
base_val_ds = base_val_ds.transform(val_trans)

train_ds = base_train_ds.balance_label(seed=0).in_ndata(ndata)
val_ds = base_val_ds

run_mgr = RunManager(exc_path=__file__, exp_name=exp_name)
# runs_mgr = RunsManager([RunManager(exc_path=__file__, exp_name=exp_name) for _ in fil_ens_l])

run_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")

run_mgr.log_param("train_dataset", train_ds.ds_str)
run_mgr.log_param("val_dataset", val_ds.ds_str)
run_mgr.log_param("num_classes", num_classes := train_ds.fetch_classes())

run_mgr.log_param("train_trans", repr(train_trans))
run_mgr.log_param("val_trans", repr(val_trans))

run_mgr.log_param("train_num", len(train_ds))
run_mgr.log_param("val_num", len(val_ds))

run_mgr.log_param("epochs", epochs := int(base_epochs * base_ndata / ndata + 1e-7))
run_mgr.log_param("max_lr", max_lr)
run_mgr.log_param("batch_size", batch_size)

train_dl = train_ds.loader(batch_size, shuffle=True)
val_dl = val_ds.loader(batch_size, shuffle=True)

run_mgr.log_param("iters/epoch", iters_per_epoch := len(train_dl))
run_mgr.log_param("data_per_class", ndata / num_classes)
# run_mgr.log_param("base_fils", base_fils)
run_mgr.log_param("fils", fils)
run_mgr.log_param("ensembles", ensembles)
# run_mgr.log_param("fils", fils_l)
# run_mgr.log_param("ensembles", ensembles_l)
run_mgr.log_text(src_name, src_text)

trainers = []
models = []

network = Network(net(num_classes=num_classes, nb_fils=fils, ee_groups=ensembles))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
# optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trainer = Trainer(network, criterion, optimizer, scheduler_t, device)
models.append(network)
trainers.append(trainer)
    
network = Networks(merge_stat=True)
for _ in range(ensembles):
    network_mem = Network(net(num_classes=num_classes, nb_fils=fils))
    network.append(network_mem)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
# optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trainer = MergeEnsemble(network, criterion, optimizer, scheduler_t, device)
models.append(network)
trainers.append(trainer)
    
for network, trainer in zip(models, trainers):
    hp_dict = {
        "params": network.count_params(),
        "criterion": trainer.repr_criterion(),
        "optimizer": trainer.repr_optimizer(),
        "scheduler": trainer.repr_scheduler(),
    }

    run_mgr.log_params(hp_dict)
    run_mgr.log_text("model_repr.txt", trainer.networks.repr_network())
    run_mgr.log_text("model_torchinfo.txt", trainer.networks.torchinfo(dl=train_dl))

    print(f"{len(train_ds)=}")

    for e in range(epochs):
        lrs = trainer.get_lr()
        train_loss, train_acc = trainer.train_1epoch(train_dl)

        met_dict = {"epoch": e + 1, "lr": lrs, "train_loss": train_loss, "train_acc": train_acc}
        if utils.interval(step=e + 1, itv=epochs/100, last_step=epochs):
            val_loss, val_acc = trainer.val_1epoch(val_dl)
            met_dict.update({"val_loss": val_loss, "val_acc": val_acc})

        run_mgr.log_metrics(met_dict, step=e + 1)
        run_mgr.log_metrics(trainer.timeinfo(), step=e + 1)
        trainer.printmet(met_dict, e + 1, epochs, itv=epochs / 5)
        run_mgr.ref_stats(step=e + 1, itv=epochs/100, last_step=epochs)
        run_mgr.ref_results(step=e + 1, itv=epochs/100, last_step=epochs)

# run_mgr.log_torch_save(trainer.networks.get_sd(), "state_dict.pt")


