import sys
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms

this_path = Path(__file__) if '__file__' in globals() else Path("<undefined>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

ee_tools_path_p = work_path / Path("ee")
sys.path.append(str(ee_tools_path_p))

import utils
from datasets import fetch_handler
from run_manager import RunManager, RunsManager
from network import Network, Networks
from trainer import MultiTrainer, Trainer

from ee_tools.ee_trainer import EETrainer
from ee_tools.models.resnet_ee import resnet18 as resnet18_ee
from ee_tools.models.resnet_ee import resnet50 as resnet50_ee
from ee_tools.models.resnet_cifar_ee import resnet18 as resnet18_cifar_ee
from ee_tools.models.resnet_cifar_hase import resnet18 as resnet18_cifar_hase
from ee_tools.models.resnet_cifar_ee import resnet50 as resnet50_cifar_ee

src_text, src_name= utils.get_source(with_name=True)

# exp_name = "exp_hase"
# exp_name = "exp_sync_mod"
exp_name = "exp_tmp"

# net = resnet18_ee
net = resnet18_cifar_hase

base_epochs = 100
base_ndata = 50000

ndata = 50000
# wd = 5e-4
# wd_l = [5e-3, 5e-5, 5e-7]
wd_l = [5e-4]

# optim_lr = [("sgd", 0.1), ("adam", 0.1)]
# optim_lr = [("sgd", 0.1), ("adam", 5e-3)]
# optim_lr = [("sgd", 0.1)]
# optim_lr = [("adam", 0.1), ("adam", 5e-3)]
# optim_lr = [("sgd", 0.1)]
# optim_lr = [("adam", 5e-3), ("adam", 0.1), ("adamw", 5e-3), ("adamw", 0.1)]
optim_lr = [("adamw", 5e-3)]

# max_lr = 0.1
batch_size = 256
# batch_size = 128


# chs_ens_l = [(32, 1), (4, 64)]
# chs_ens_l = [(32, 1), (16, 4), (8, 16), (4, 64)]
# chs_ens_l = [(64, 1), (16, 16), (8, 64), (4, 256)] # base = 64
# chs_ens_l = [(64, 1), (2, 1024)] # base = 64
# chs_ens_l = [(64, 1), (32, 4)] # base = 64
chs_ens_l = [(64, 1), (32, 4), (16, 16)] # base = 64

chs_l, ensembles_l = map(list, zip(*chs_ens_l))
base_chs_l = [round(a * b ** (1/2)) for a, b in chs_ens_l]


ds_root = work_path / "assets/datasets/"
train_ds_str = "cifar100_train"
val_ds_str = "cifar100_val"

base_train_ds = fetch_handler(ds_root, train_ds_str)
base_val_ds = fetch_handler(ds_root, val_ds_str)

for optim_str, max_lr in optim_lr:
    for wd in wd_l:
        match train_ds_str:
            case "cifar10_train":
                train_trans = [transforms.ToImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
                val_trans = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
            case "cifar100_train":
                train_trans = [transforms.ToImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
                val_trans = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
            case "stl10_train":
                train_trans = [transforms.ToImage(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 360)), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
                val_trans = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
            case _:
                pass

        train_ds = base_train_ds.transform(train_trans)
        val_ds = base_val_ds.transform(val_trans)

        train_dl = train_ds.loader(batch_size, shuffle=True)
        val_dl = val_ds.loader(batch_size, shuffle=False)

        runs_mgr = RunsManager([RunManager(exec_path=__file__, exp_name=exp_name, exp_tpl="exp_tpl_ee") for _ in chs_ens_l])
        runs_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")
        runs_mgr.log_param("da", "with_rr")

        runs_mgr.log_param("train_dataset", train_ds.state.dataset_id)
        runs_mgr.log_param("val_dataset", val_ds.state.dataset_id)
        runs_mgr.log_param("num_classes", num_classes := train_ds.fetch_classes())

        runs_mgr.log_param("train_trans", repr(train_trans))
        runs_mgr.log_param("val_trans", repr(val_trans))

        runs_mgr.log_param("train_ndata", len(train_ds))
        runs_mgr.log_param("val_ndata", len(val_ds))

        runs_mgr.log_param("epochs", epochs := int(base_epochs * base_ndata / ndata + 1e-7))
        runs_mgr.log_param("max_lr", max_lr)
        runs_mgr.log_param("wd", wd)
        runs_mgr.log_param("batch_size", batch_size)

        runs_mgr.log_param("iters/epoch", len(train_dl))
        runs_mgr.log_param("iters", len(train_dl) * epochs)
        runs_mgr.log_param("target_steps", base_ndata * base_epochs)
        runs_mgr.log_param("ndata_per_class", len(train_ds) / num_classes)
        runs_mgr.log_param("channels", chs_l)
        runs_mgr.log_param("ensembles", ensembles_l)
        runs_mgr.log_param("base_chs", base_chs_l)
        runs_mgr.log_text(src_name, src_text)

        trainers = []
        for channels, ensembles in chs_ens_l:
            network = Network(net(num_classes=num_classes, channels=channels, ensembles=ensembles, merge_mode="both"))
            criterion = torch.nn.CrossEntropyLoss()
            if optim_str == "sgd":
                optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=wd, nesterov=True)
            elif optim_str == "adam":
                optimizer = torch.optim.Adam(network.parameters(), lr=max_lr, weight_decay=wd)
            elif optim_str == "adamw":
                optimizer = torch.optim.AdamW(network.parameters(), lr=max_lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            trainer = EETrainer(network, criterion, optimizer, scheduler, device)
            trainers.append(trainer)

        mtrainer = MultiTrainer(trainers)
        networks = mtrainer.networks
                
        hp_dict = {
            "num_params": mtrainer.networks.param_stat(lambda p: p.numel()),
            "criterion": mtrainer.fmt_criterion(),
            "optimizer": mtrainer.fmt_optimizer(),
            "scheduler": mtrainer.fmt_scheduler(),
        }

        runs_mgr.log_params(hp_dict)
        runs_mgr.log_text("model_repr.txt", networks.repr_network())
        runs_mgr.log_text("model_torchinfo.txt", networks.torchinfo(dl=train_dl))

        for e in range(epochs):
            lr = mtrainer.get_lr()

            train_loss, train_acc, train_aux = mtrainer.train_1epoch(train_dl)
            mets = {"epoch": e + 1, "lr": lr, "train_loss": train_loss, "train_acc": train_acc}

            if utils.interval(step=e + 1, itv=epochs/100, last_step=epochs):
                val_loss, val_acc, val_aux = mtrainer.val_1epoch(val_dl)
                mets |= {"val_loss": val_loss, "val_acc": val_acc}
            else:
                val_aux = {}
                mets |= {"val_loss": None, "val_acc": None}

            runs_mgr.log_metrics(mets, step=e + 1)
            runs_mgr.log_metrics(train_aux, step=e + 1)
            runs_mgr.log_metrics(val_aux, step=e + 1)
            # runs_mgr.log_metrics(mtrainer.time_info(), step=e + 1)
            runs_mgr.log_metrics(mtrainer.time_stats(incl_fmt=False), step=e + 1)
            runs_mgr.log_metrics(mtrainer.time_stats_mt(incl_fmt=False), step=e + 1)

            mtrainer.printmet(mets, e + 1, epochs, itv=epochs / 5)
            runs_mgr.ref_stats(step=e + 1, itv=epochs/100, last_step=epochs)
            runs_mgr.ref_results(step=e + 1, itv=epochs/100, last_step=epochs)

        runs_mgr.log_torch_save(mtrainer.networks.get_sd(), "state_dict.pt")

