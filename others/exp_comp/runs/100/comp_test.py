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

from torchvision.models import resnet18
from torchvision.models import efficientnet_v2_s

src_text, src_name= utils.get_source(with_name=True)

fetch_ds = DatasetFetcher(root=work_path / "assets/datasets/")

# exp_name = "exp_sgd_long"
# exp_name = "exp_sgd_lr"
# exp_name = "exp_tmp"
exp_name = "exp_comp"

net = resnet18

base_epochs = 100
max_lr_l = [0.01, 0.003, 0.001, 0.0003, 0.0001]
batch_size = 128
train_ratio = 0.7

train_ds_str = "comp_train"
test_ds_str = "comp_test"

train_val_ds = fetch_ds("comp_train")
train_ds, val_ds = train_val_ds.shuffle(seed=0).split_ratio(train_ratio)
# train_ds, val_ds = train_val_ds.balance_label(seed=0).split_ratio(train_ratio)

normalize = transforms.Normalize(mean=[0.3123692572116852, 0.3123692572116852, 0.3123692572116852], std=[0.3177872598171234, 0.3177872598171234, 0.3177872598171234], inplace=True)

train_trans = [transforms.ToImage(), transforms.CenterCrop(85), transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees=30, fill=(0, 0, 0)), transforms.ToDtype(torch.float32, scale=True), normalize]
val_trans = [transforms.ToImage(), transforms.CenterCrop(85), transforms.ToDtype(torch.float32, scale=True), normalize]

train_ds = train_ds.transform(train_trans)
val_ds = val_ds.transform(val_trans)

runs_mgr = RunsManager([RunManager(exc_path=__file__, exp_name=exp_name) for _ in max_lr_l])

runs_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")

runs_mgr.log_param("train_dataset", train_ds.ds_str)
runs_mgr.log_param("val_dataset", val_ds.ds_str)
runs_mgr.log_param("num_classes", num_classes := train_ds.fetch_classes())

runs_mgr.log_param("train_trans", repr(train_trans))
runs_mgr.log_param("val_trans", repr(val_trans))

runs_mgr.log_param("train_ndata", len(train_ds))
runs_mgr.log_param("val_ndata", len(val_ds))

runs_mgr.log_param("epochs", epochs := base_epochs)
runs_mgr.log_param("max_lr", max_lr_l)
runs_mgr.log_param("batch_size", batch_size)

train_dl = train_ds.loader(batch_size, shuffle=True)
val_dl = val_ds.loader(batch_size, shuffle=True)

runs_mgr.log_param("iters/epoch", len(train_dl))
runs_mgr.log_param("iters", len(train_dl) * epochs)
runs_mgr.log_text(src_name, src_text)

trainers = []
for max_lr in max_lr_l:
    network = Network(net(weights="IMAGENET1K_V1")).tl_setup(num_classes=4, linear_layer="fc")
    # network = Network(net(weights="IMAGENET1K_V1")).tl_setup(num_classes=4, linear_layer="classifier.1")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
    scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = Trainer(network, criterion, optimizer, scheduler_t, device)
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
    if e == 20:
        mtrainer.networks.unfreeze()
    lrs = mtrainer.get_lr()
    train_loss, train_acc = mtrainer.train_1epoch(train_dl)

    met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc}
    if utils.interval(step=e + 1, itv=epochs/100, last_step=epochs):
        val_loss, val_acc = mtrainer.val_1epoch(val_dl)
        met_dict.update({"val_loss": val_loss, "val_acc": val_acc})
        
    
    runs_mgr.log_metric("trainable", mtrainer.networks.count_params(trainable=True), step=e + 1)

    runs_mgr.log_metrics(met_dict, step=e + 1)
    runs_mgr.log_metrics(mtrainer.time_info(), step=e + 1)
    runs_mgr.log_metrics(mtrainer.time_stats(incl_fmt=False), step=e + 1)
    runs_mgr.log_metrics(mtrainer.time_stats_mt(incl_fmt=False), step=e + 1)

    mtrainer.printmet(met_dict, e + 1, epochs, itv=epochs / 5)
    runs_mgr.ref_stats(step=e + 1, itv=epochs/100, last_step=epochs)
    runs_mgr.ref_results(step=e + 1, itv=epochs/100, last_step=epochs)

runs_mgr.log_torch_save(mtrainer.networks.get_sd(), "state_dict.pt")
