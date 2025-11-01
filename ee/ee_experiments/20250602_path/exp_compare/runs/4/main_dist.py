import sys
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms

this_path = Path(__file__) if '__file__' in globals() else Path("<unknown>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

ee_tools_path_p = work_path / Path("ee")
sys.path.append(str(ee_tools_path_p))

import utils
from datasets import DatasetFetcher
from run_manager import RunManager, RunsManager
from trainer import MultiTrainer, Network, Networks, Trainer

from ee_tools.ee_trainer import EETrainer
from ee_tools.models.resnet_ee import resnet18 as resnet18_ee
from ee_tools.models.resnet_git_ee import resnet18 as resnet18_git_ee
from ee_tools.models.resnet_git_ee import resnet50 as resnet50_git_ee

src_text, src_name= utils.get_source(with_name=True)

fetch_ds = DatasetFetcher(root=work_path / "assets/datasets/")

exp_name = "exp_compare"
# exp_name = "exp_sgd_lr"
# exp_name = "exp_tmp"

net = resnet18_git_ee

base_epochs = 200
base_ndata = 5000
ndata_l = [5000, 1000]
max_lr_l = [0.1, 0.005]
batch_size = 128

# base_epochs = 2
# base_ndata = 256
# ndata_l = [256]
# # max_lr_l = [0.1]
# max_lr_l = [0.1, 0.005]
# batch_size = 128


train_ds_str = "cifar100_train"
val_ds_str = "cifar100_val"

fil_ens_l = [(32, 1), (4, 64)]
# # fil_ens_l = [(32, 1), (16, 4), (8, 16), (4, 64)]
fils_l, ensembles_l = map(list, zip(*fil_ens_l))
base_fils_l = [round(a * b ** (1/2)) for a, b in fil_ens_l]

base_train_ds = fetch_ds(train_ds_str)
base_val_ds = fetch_ds(val_ds_str)

match train_ds_str:
    case "cifar10_train":
        train_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), base_train_ds.normalizer()]
        val_trans = [transforms.ToTensor(), base_train_ds.normalizer()]
    case "cifar100_train":
        train_trans = [transforms.ToImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
        val_trans = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
    case "stl10_train":
        train_trans = [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 360)), base_train_ds.normalizer()]
        val_trans = [transforms.ToTensor(), base_train_ds.normalizer()]
    case _:
        pass

base_train_ds = base_train_ds.transform(train_trans)
base_val_ds = base_val_ds.transform(val_trans)

for ndata in ndata_l:
    train_ds = base_train_ds.balance_label(seed=0).in_ndata(ndata)
    val_ds = base_val_ds

    runs_mgr = RunsManager([RunManager(exc_path=__file__, exp_name=exp_name) for _ in fil_ens_l for _ in max_lr_l])

    runs_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")

    runs_mgr.log_param("train_dataset", train_ds.ds_str)
    runs_mgr.log_param("val_dataset", val_ds.ds_str)
    runs_mgr.log_param("num_classes", num_classes := train_ds.fetch_classes())

    runs_mgr.log_param("train_trans", repr(train_trans))
    runs_mgr.log_param("val_trans", repr(val_trans))

    runs_mgr.log_param("train_ndata", len(train_ds))
    runs_mgr.log_param("val_ndata", len(val_ds))

    runs_mgr.log_param("epochs", epochs := int(base_epochs * base_ndata / ndata + 1e-7))
    runs_mgr.log_param("max_lr", max_lr_l * len(fil_ens_l))
    runs_mgr.log_param("batch_size", batch_size)

    train_dl = train_ds.loader(batch_size, shuffle=True)
    val_dl = val_ds.loader(batch_size, shuffle=True)

    runs_mgr.log_param("iters/epoch", len(train_dl))
    runs_mgr.log_param("iters", len(train_dl) * epochs)
    runs_mgr.log_param("target_steps", base_ndata * base_epochs)
    runs_mgr.log_param("ndata_per_class", ndata / num_classes)
    runs_mgr.log_param("fils", fils_l)
    runs_mgr.log_param("ensembles", ensembles_l)
    runs_mgr.log_param("base_fils", base_fils_l)
    runs_mgr.log_text(src_name, src_text)

    trainers = []
    for fils, ensembles in fil_ens_l:
        network = Network(net(num_classes=num_classes, nb_fils=fils, ee_groups=ensembles, merge_mode="both"))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(network.parameters(), lr=max_lr_l[0], momentum=0.9, weight_decay=5e-4, nesterov=True) # Hard coded for now
        # optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        trainer = EETrainer(network, criterion, optimizer, scheduler, device)
        trainers.append(trainer)

        network = Network(net(num_classes=num_classes, nb_fils=fils, ee_groups=ensembles, merge_mode="both"))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=max_lr_l[1]) # Hard coded for now
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        trainer = EETrainer(network, criterion, optimizer, scheduler, device)
        trainers.append(trainer)
        
    mtrainer = MultiTrainer(trainers)
        
    hp_dict = {
        "criterion": mtrainer.fmt_criterion(),
        "optimizer": mtrainer.fmt_optimizer(),
        "scheduler": mtrainer.fmt_scheduler(),
    }

    runs_mgr.log_params(hp_dict)
    runs_mgr.log_text("model_repr.txt", network.repr_network())
    runs_mgr.log_text("model_torchinfo.txt", network.torchinfo(dl=train_dl))

    for e in range(epochs):
        lrs = mtrainer.get_lr()
        train_loss, train_acc, train_path_loss, train_path_acc = mtrainer.train_1epoch(train_dl)

        met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc}
        met_dict |= {"train_path_loss": train_path_loss, "train_path_acc": train_path_acc}
        if utils.interval(step=e + 1, itv=epochs/100, last_step=epochs):
            val_loss, val_acc, val_path_loss, val_path_acc = mtrainer.val_1epoch(val_dl)
            met_dict.update({"val_loss": val_loss, "val_acc": val_acc, "val_path_loss": val_path_loss, "val_path_acc": val_path_acc})

        runs_mgr.log_metrics(met_dict, step=e + 1)
        runs_mgr.log_metrics(mtrainer.time_info(), step=e + 1)
        runs_mgr.log_metrics(mtrainer.time_stats(incl_fmt=False), step=e + 1)
        runs_mgr.log_metrics(mtrainer.time_stats_mt(incl_fmt=False), step=e + 1)

        mtrainer.printmet(met_dict, e + 1, epochs, itv=epochs / 5)
        runs_mgr.ref_stats(step=e + 1, itv=epochs/100, last_step=epochs)
        runs_mgr.ref_results(step=e + 1, itv=epochs/100, last_step=epochs)

        if utils.interval(step=e + 1, itv=epochs/10, last_step=epochs):
            runs_mgr.log_torch_save(mtrainer.networks.get_sd(), f"state_dict_{e + 1}.pt")

    # runs_mgr.log_torch_save(mtrainer.networks.get_sd(), "state_dict.pt")
