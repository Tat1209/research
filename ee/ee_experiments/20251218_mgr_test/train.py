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
from ee_tools.ee_refiner import EERefiner
from ee_tools.ee_trainer import EETrainer
from network import Network, Networks
from exp_manager import ExpManager
from torchvision import models
from trainer import MultiTrainer, Trainer

def run_mgr(cfg: dict):
    torch.set_num_threads(cfg["num_threads"]) 
    torch.set_num_interop_threads(cfg["num_interop_threads"])
    src_text, src_name = utils.get_source(path=this_path, with_name=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = cfg["exp_name"]
    net = getattr(models, cfg["model_str"])

    train_ds_str = cfg["train_ds_str"]
    val_ds_str = cfg["val_ds_str"]
    ndata = cfg["ndata"]

    epochs = cfg["epochs"]
    optim_str = cfg["optim_str"]
    max_lr = cfg["max_lr"]
    batch_size = cfg["batch_size"]
    wd = cfg["wd"]

    div = cfg["div"]
    ens = round(div ** 2)
    scale_ch = 1 / div

    ds_root = work_path / "assets/datasets/"

    base_train_ds = fetch_handler(ds_root, train_ds_str)
    base_val_ds = fetch_handler(ds_root, val_ds_str)
    
    match train_ds_str:
        case "cifar100_train":
            train_trans = [transforms.ToImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
            val_trans = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
        case "stl10_train":
            train_trans = [transforms.ToImage(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 360)), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
            val_trans = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), base_train_ds.normalizer()]
        case _:
            # default transforms or raise error
            train_trans = []
            val_trans = []

    train_ds = base_train_ds.transform(train_trans).balance_class(seed=0).in_ndata(ndata)
    val_ds = base_val_ds.transform(val_trans)

    train_dl = train_ds.loader(batch_size, shuffle=True, num_workers=cfg["num_workers"])
    val_dl = val_ds.loader(batch_size, shuffle=False, num_workers=cfg["num_workers"])

    exp = ExpManager(exp_path=this_path.parent / exp_name, exp_tpl="exp_tpl_ee")
    run_mgr = exp.create_run()

    run_mgr.log_param("model_arc", f"{net.__name__}")
    
    run_mgr.log_param("train_dataset", train_ds.state.dataset_id)
    run_mgr.log_param("val_dataset", val_ds.state.dataset_id)
    run_mgr.log_param("num_classes", num_classes := train_ds.fetch_classes())

    run_mgr.log_param("train_trans", repr(train_trans))
    run_mgr.log_param("val_trans", repr(val_trans))

    run_mgr.log_param("train_ndata", len(train_ds))
    run_mgr.log_param("val_ndata", len(val_ds))

    run_mgr.log_param("epochs", epochs)
    run_mgr.log_param("max_lr", max_lr)
    run_mgr.log_param("wd", wd)
    run_mgr.log_param("batch_size", batch_size)

    run_mgr.log_param("base_epochs", cfg.get("base_epochs", None))
    run_mgr.log_param("iters/epoch", len(train_dl))
    run_mgr.log_param("iters", len(train_dl) * epochs)
    run_mgr.log_param("processed_ndata", ndata * epochs)
    run_mgr.log_param("ndata_per_class", len(train_ds) / num_classes)
    run_mgr.log_param("div", div)
    run_mgr.log_param("ensembles", ens)
    run_mgr.log_param("scale_ch", scale_ch)
    run_mgr.log_text(src_name, src_text)

    # Network構築 (変更なし)
    if cfg.get("cifar_style"):
        network = Network(EERefiner(net(num_classes=num_classes)).cifar_style().multi_narrow(div=div, agg="mean", flex_ch=cfg["flex_ch"]).init_weights().build().to(device))
    else:
        network = Network(EERefiner(net(num_classes=num_classes)).multi_narrow(div=div, agg="mean", flex_ch=cfg["flex_ch"]).init_weights().build().to(device))
    
    if cfg["compile"]:
        network = torch.compile(network, mode="max-autotune")
    
    criterion = torch.nn.CrossEntropyLoss()
    
    if optim_str == "sgd":
        optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=wd, nesterov=True, fused=True)
    elif optim_str == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=max_lr, weight_decay=wd, fused=True)
    elif optim_str == "adamw":
        optimizer = torch.optim.AdamW(network.parameters(), lr=max_lr, weight_decay=wd, fused=True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
    trainer = Trainer(network, criterion, optimizer, scheduler, device, dtype=torch.bfloat16)
            
    hp_dict = {
        "num_params": network.param_stat(lambda p: p.numel()),
        "criterion": trainer.fmt_criterion(),
        "optimizer": trainer.fmt_optimizer(),
        "scheduler": trainer.fmt_scheduler(),
    }

    run_mgr.log_params(hp_dict)
    run_mgr.log_text("model_repr.txt", network.repr_network())
    run_mgr.log_text("model_torchinfo.txt", network.torchinfo(dl=train_dl))

    # Training Loop
    for e in range(epochs):
        lr = trainer.get_lr()

        train_loss, train_acc = trainer.train_1epoch(train_dl)
        mets = {"epoch": e + 1, "lr": lr, "train_loss": train_loss, "train_acc": train_acc}

        if utils.interval(step=e + 1, itv=epochs/100, last_step=epochs):
            val_loss, val_acc = trainer.val_1epoch(val_dl)
            mets |= {"val_loss": val_loss, "val_acc": val_acc}
        else:
            mets |= {"val_loss": None, "val_acc": None}

        run_mgr.log_metrics(mets, step=e + 1)
        run_mgr.log_metrics(trainer.time_stats(incl_fmt=True), step=e + 1)
        run_mgr.log_metric("progress", f"{(e + 1) / epochs * 100:.1f}%", step=e + 1)

        trainer.printmet(mets, e + 1, epochs, itv=epochs / 5)

        run_mgr.sync(step=e + 1, itv=epochs/100, last_step=epochs)
    try:
        exp.ref_results()
        print(f"Experiment completed. Results saved to {exp.exp_path}")
    except Exception as e:
        print(f"Final result aggregation failed: {e}")