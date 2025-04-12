import sys
from pathlib import Path

import torch
from torchvision import transforms

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets, dl
from run_manager import RunManager, RunsManager, RunViewer
from trainer import Model, MyMultiTrain
from trans import Trans

# from models.resnet_1ch import resnet18 as net
from torchvision.models import resnet18 as net

ds = Datasets(root=work_path / "assets/datasets/")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

exp_name = "exp_pre1"
# exp_name = "exp_tmp"

epochs = 100
# lr = 0.005
batch_size = 128

train_ds_str_l = ["tiny-imagenet_train"]
val_ds_str_l = ["tiny-imagenet_val"]
num_classes_l = [200]

for train_ds_str, val_ds_str, num_classes in zip(train_ds_str_l, val_ds_str_l, num_classes_l):
    for max_lrs in [[0.00005, 0.0001, 0.0003, 0.0005]]:
    # for max_lrs in [[0.001, 0.003, 0.005, 0.01, 0.03]]:
        runs = [RunManager(exc_path=__file__, exp_name=exp_name) for _ in max_lrs]
        runs_mgr = RunsManager(runs)

        train_trans = [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 360))]
        val_trans = [transforms.ToTensor()]

        train_ds = ds(train_ds_str, train_trans)
        val_ds = ds(val_ds_str, val_trans)

        runs_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")

        runs_mgr.log_param("train_dataset", train_ds.ds_str)
        runs_mgr.log_param("val_dataset", val_ds.ds_str)
        runs_mgr.log_param("num_classes", num_classes)

        runs_mgr.log_param("train_trans", repr(train_trans))
        runs_mgr.log_param("val_trans", repr(val_trans))

        runs_mgr.log_param("train_num", len(train_ds))
        runs_mgr.log_param("val_num", len(val_ds))

        # runs_mgr.log_param("epochs", epochs := 3)
        runs_mgr.log_param("epochs", epochs)
        runs_mgr.log_param("max_lr", max_lrs)
        runs_mgr.log_param("batch_size", batch_size)

        train_dl = dl(train_ds, batch_size, shuffle=True)
        val_dl = dl(val_ds, batch_size, shuffle=True)

        runs_mgr.log_param("iters/epoch", iters_per_epoch := len(train_dl))

        models = []
        for i, max_lr in enumerate(max_lrs):
            network = net(num_classes=num_classes)
            loss_func = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
            scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

            model = Model(network, loss_func, optimizer, scheduler_t, device)
            models.append(model)
        mmodel = MyMultiTrain(models, device)

        runs_mgr.log_param("params", mmodel.count_params())
        hp_dict = {
            "loss_func": mmodel.repr_loss_func(),
            "optimizer": mmodel.repr_optimizer(),
            "scheduler": mmodel.repr_scheduler(),
        }

        runs_mgr.log_params(hp_dict)
        runs_mgr.log_text(mmodel.repr_network(), "model_layers.txt")
        runs_mgr.log_text(mmodel.arc_check(dl=train_dl), "model_structure.txt")

        print(f"{max_lrs=}, {batch_size=}")

        for e in range(epochs):
            runs_mgr.log_metric("lr", mmodel.get_lr(), step=e + 1)

            train_loss, train_acc = mmodel.train_1epoch(train_dl)
            val_loss, val_acc = mmodel.val_1epoch(val_dl)

            met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}

            runs_mgr.log_metrics(met_dict, step=e + 1)
            mmodel.printlog(met_dict, e + 1, epochs, itv=epochs / 4)
            runs_mgr.ref_stats(itv=5, step=e + 1, last_step=epochs)

        runs_mgr.log_torch_save(mmodel.get_sd(), "state_dict.pt")

        rv = RunViewer(exc_path=__file__, exp_name=exp_name)
        rv.ref_results()
