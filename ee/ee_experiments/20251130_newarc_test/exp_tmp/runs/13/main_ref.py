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
from torchvision.models import resnet18
from ee_tools.ee_refiner import EERefiner

def main():
    src_text, src_name= utils.get_source(with_name=True)

    exp_name = "exp_tmp"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = resnet18_ee
    net = resnet18

    base_epochs = 100
    base_ndata = 10000
    ndata_epochs_l = [(ndata, base_epochs * base_ndata // ndata) for ndata in [10000]]
    # ndata_epochs_l = [(50000, 100)] + [(ndata, base_epochs * base_ndata // ndata) for ndata in [10000, 5000, 2000, 1000]]

    wd_l = [1e-6]
    # wd_l = [0, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2]

    optim_lr = [("sgd", 0.1), ("adamw", 5e-3)]
    # optim_lr = [("sgd", 0.1), ("adamw", 5e-3)]

    batch_size = 128

    div_l = [1, 4, 8]
    ens_l = [div**2 for div in div_l]
    ch_scale = [1 / div for div in div_l]

    ds_root = work_path / "assets/datasets/"
    train_ds_str = "cifar100_train"
    val_ds_str = "cifar100_val"

    base_train_ds = fetch_handler(ds_root, train_ds_str)
    base_val_ds = fetch_handler(ds_root, val_ds_str)
    
    for optim_str, max_lr in optim_lr:
        for ndata, epochs in ndata_epochs_l:
            for wd in wd_l:
                match train_ds_str:
                    # case "cifar10_train":
                        # train_trans = [transforms.ToImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
                        # val_trans = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
                    case "cifar100_train":
                        train_trans = [transforms.ToImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
                        val_trans = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
                    case "stl10_train":
                        train_trans = [transforms.ToImage(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 360)), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
                        val_trans = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
                    case _:
                        pass

                train_ds = base_train_ds.transform(train_trans).balance_class(seed=0).in_ndata(ndata)
                val_ds = base_val_ds.transform(val_trans)

                train_dl = train_ds.loader(batch_size, shuffle=True)
                val_dl = val_ds.loader(batch_size, shuffle=False)

                runs_mgr = RunsManager([RunManager(exec_path=__file__, exp_name=exp_name, exp_tpl="exp_tpl_ee") for _ in div_l])
                runs_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")

                runs_mgr.log_param("train_dataset", train_ds.state.dataset_id)
                runs_mgr.log_param("val_dataset", val_ds.state.dataset_id)
                runs_mgr.log_param("num_classes", num_classes := train_ds.fetch_classes())

                runs_mgr.log_param("train_trans", repr(train_trans))
                runs_mgr.log_param("val_trans", repr(val_trans))

                runs_mgr.log_param("train_ndata", len(train_ds))
                runs_mgr.log_param("val_ndata", len(val_ds))

                runs_mgr.log_param("epochs", epochs)
                runs_mgr.log_param("max_lr", max_lr)
                runs_mgr.log_param("wd", wd)
                runs_mgr.log_param("batch_size", batch_size)

                runs_mgr.log_param("iters/epoch", len(train_dl))
                runs_mgr.log_param("iters", len(train_dl) * epochs)
                runs_mgr.log_param("target_steps", base_ndata * base_epochs)
                runs_mgr.log_param("ndata_per_class", len(train_ds) / num_classes)
                runs_mgr.log_param("div", div_l)
                runs_mgr.log_param("ensembles", ens_l)
                runs_mgr.log_text(src_name, src_text)

                trainers = []
                for div in div_l:
                    # network = Network(resnet18_ee(num_classes=num_classes, channels=64 // div, ensembles=div**2, merge_mode="both").to(device))
                    network = Network(EERefiner(resnet18(num_classes=num_classes)).cifar_style().multi_narrow(div=div, agg="both").init_weights().build().to(device))
                    criterion = torch.nn.CrossEntropyLoss()
                    if optim_str == "sgd":
                        optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=wd, nesterov=True)
                    elif optim_str == "adam":
                        optimizer = torch.optim.Adam(network.parameters(), lr=max_lr, weight_decay=wd)
                    elif optim_str == "adamw":
                        optimizer = torch.optim.AdamW(network.parameters(), lr=max_lr, weight_decay=wd)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
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

if __name__ == "__main__":
    main()
