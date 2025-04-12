import sys
from pathlib import Path
import math

import torch
from torchvision import transforms

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets
from run_manager import RunManager, RunsManager
from trainer import Trainer, MultiTrainer
from modules import CrossEntropyLossT
import utils

# from models.resnet_ee import resnet18 as net
# from models.resnet_git_ee import resnet18 as net
from models.resnet_git_ee import resnet50 as net
# from models.gitresnet_ee import resnet18 as net

ds = Datasets(root=work_path / "assets/datasets/")

exp_name = "exp_tmp"
# exp_name = "exp_r50"

# epochs = 5
base_epochs = 3

max_lr = 0.0005
batch_size = 128

train_ds_str_l = ["cifar10_train"]
val_ds_str_l = ["cifar10_val"]

# train_ds_str_l = ["cifar10_train", "cifar100_train", "stl10_train"]
# val_ds_str_l = ["cifar10_val", "cifar100_val", "stl10_val"]

ndata_l = [10000]
# ndata_l = [10000, 5000, 2500, 1000]
fils_ll = [[64, 32]]
# fils_ll = [[64, 32, 16, 8, 4]]

base_fils = 64

src_text, src_name= utils.get_source(with_name=True)

for train_ds_str, val_ds_str in zip(train_ds_str_l, val_ds_str_l):

    base_train_ds = ds(train_ds_str)
    val_ds = ds(val_ds_str)
    
    match train_ds_str:
        case "cifar10_train":
            base_ndata = 10000
            train_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), base_train_ds.normalizer()]
            val_trans = [transforms.ToTensor(), base_train_ds.normalizer()]
        case "cifar100_train":
            base_ndata = 10000
            train_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), base_train_ds.normalizer()]
            val_trans = [transforms.ToTensor(), base_train_ds.normalizer()]
        case "stl10_train":
            base_ndata = 5000
            train_trans = [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 360)), base_train_ds.normalizer()]
            val_trans = [transforms.ToTensor(), base_train_ds.normalizer()]
        case _:
            pass

    base_train_ds = base_train_ds.transform(train_trans)
    val_ds = val_ds.transform(val_trans)

    for ndata in ndata_l:
        train_ds = base_train_ds.balance_label(seed=0).in_ndata(ndata)
        if ndata > len(train_ds):
            continue
        
        for fils_l in fils_ll:
            runs_mgr = RunsManager([RunManager(exc_path=__file__, exp_name=exp_name) for _ in fils_l])

            runs_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")

            runs_mgr.log_param("train_dataset", train_ds.ds_str)
            runs_mgr.log_param("val_dataset", val_ds.ds_str)
            runs_mgr.log_param("num_classes", num_classes := train_ds.fetch_classes())

            runs_mgr.log_param("train_trans", repr(train_trans))
            runs_mgr.log_param("val_trans", repr(val_trans))

            runs_mgr.log_param("train_num", len(train_ds))
            runs_mgr.log_param("val_num", len(val_ds))

            runs_mgr.log_param("epochs", epochs := int(base_epochs / ndata * base_ndata + 1e-7))
            runs_mgr.log_param("max_lr", max_lr)
            runs_mgr.log_param("batch_size", batch_size)

            train_dl = train_ds.loader(batch_size, shuffle=True)
            val_dl = val_ds.loader(batch_size, shuffle=True)

            runs_mgr.log_param("iters/epoch", iters_per_epoch := len(train_dl))
            runs_mgr.log_param("base_fils", base_fils)
            runs_mgr.log_param("fils", fils_l)
            runs_mgr.log_param("ensembles", ensembles_l := [int((base_fils / fils + 1e-7) ** 2) for fils in fils_l])
            runs_mgr.log_text(src_text, src_name)

            trainers = []
            for fils, ensembles in zip(fils_l, ensembles_l):
                network = net(num_classes=num_classes)
                # network = net(num_classes=num_classes, nb_fils=fils, ee_groups=ensembles)
                # network = net(num_classes=num_classes, nb_fils=fils, ee_groups=ensembles, merge_sum=True)
                # network = torch.nn.DataParallel(network, device_ids=[0, 1, 2])

                # loss_func = CrossEntropyLossT(T=1)
                loss_func = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
                scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")
                device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

                trainer = Trainer(network, loss_func, optimizer, scheduler_t, device)
                trainers.append(trainer)
            mtrainer = MultiTrainer(trainers, device)

            runs_mgr.log_param("params", mtrainer.count_params())
            hp_dict = {
                "loss_func": mtrainer.repr_loss_func(),
                "optimizer": mtrainer.repr_optimizer(),
                "scheduler": mtrainer.repr_scheduler(),
            }

            runs_mgr.log_params(hp_dict)
            runs_mgr.log_text(mtrainer.repr_network(), "model_layers.txt")
            runs_mgr.log_text(mtrainer.arc_check(dl=train_dl), "model_structure.txt")

            print(f"{base_fils=}, {fils_l}, {len(train_ds)=}")

            for e in range(epochs):
                lrs = mtrainer.get_lr()

                train_loss, train_acc = mtrainer.train_1epoch(train_dl)

                met_dict = {"epoch": e + 1, "lr": lrs, "train_loss": train_loss, "train_acc": train_acc}
                if utils.interval(step=e + 1, itv=epochs/100, last_step=epochs):

                    val_loss, val_acc = mtrainer.val_1epoch(val_dl)
                    met_dict.update({"val_loss": val_loss, "val_acc": val_acc})

                runs_mgr.log_metrics(met_dict, step=e + 1)
                runs_mgr.log_metrics(mtrainer.timeinfo(), step=e + 1)
                mtrainer.printmet(met_dict, e + 1, epochs, itv=epochs / 5)
                runs_mgr.ref_stats(step=e + 1, itv=5, last_step=epochs)

            runs_mgr.log_torch_save(mtrainer.get_sd(), "state_dict.pt")
            runs_mgr.ref_results()

