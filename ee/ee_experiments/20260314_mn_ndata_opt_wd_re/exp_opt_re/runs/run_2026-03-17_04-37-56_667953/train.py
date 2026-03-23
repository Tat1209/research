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
from exp_manager import ExpManager
from network import Network
from torchvision import models
from trainer import Trainer
from transforms_util import GrayToRGB


def run(cfg: dict):
    torch.set_num_threads(cfg["num_threads"])
    torch.set_num_interop_threads(cfg["num_interop_threads"])
    src_text, src_name = utils.get_source(path=this_path, with_name=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = cfg["exp_name"]
    net = getattr(models, cfg["model_str"])

    train_ds_str = cfg["train_ds_str"]
    val_ds_str = cfg["val_ds_str"]
    ipc = cfg["ipc"]

    # epochs = cfg["epochs"]
    optim_str = cfg["optim_str"]
    max_lr = cfg["max_lr"]
    batch_size = cfg["batch_size"]
    wd = cfg["wd"]

    div = cfg["div"]
    ens = round(div ** 2)
    scale_ch = 1 / div

    ds_root = work_path / "assets/datasets/"

    base_train_ds = fetch_handler(ds_root, train_ds_str)

    # trainvalの場合は分割処理
    if "labeled" in train_ds_str:
        # クラス比率を維持して80%を学習、20%を検証に分割
        base_train_ds, base_val_ds = base_train_ds.split_ratio(0.8, stratify=True, seed=0)
    else:
        base_val_ds = fetch_handler(ds_root, val_ds_str)

    try:
        # ipcが"max"や"all"の場合は実行、数値指定でそれらと重複する場合はスキップする
        train_ds = base_train_ds.in_nshot(
            ipc=ipc,
            mode="strict", 
            seed=0, 
            notall=True,  # "all"と同じ結果になる数値指定はスキップ
            notmax=True   # "max"と同じ結果になる数値指定はスキップ
        )
            
    except ValueError as e:
        print(f"[SKIP] Experiment skipped: {e}")
        return

    val_ds = base_val_ds

    epochs = 200 if len(train_ds) >= 10000 else (200 * 10000) // len(train_ds)
    dataset_base = train_ds_str.rsplit('_', 1)[0] # "cifar10_train" -> "cifar10", "tiny-imagenet_train" -> "tiny-imagenet"

    match dataset_base:
        case "cifar10" | "cifar100":
            # RGB, Low-Res (32x32)
            train_trans = [
                transforms.ToImage(), 
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]
            val_trans = [
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]

        case "svhn":
            # RGB, Low-Res, No Flip (数字は反転不可)
            train_trans = [
                transforms.ToImage(), 
                transforms.RandomCrop(32, padding=4), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]
            val_trans = [
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]

        case "mnist":
            # Grayscale -> RGB(3ch), No Flip
            # MNISTは元が1chなので、Grayscale(3)で3ch化するのは正しい
            train_trans = [
                transforms.ToImage(), 
                transforms.Pad(2),  # 28x28 -> 32x32
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]
            val_trans = [
                transforms.ToImage(), 
                transforms.Pad(2), 
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]

        case "fashion-mnist":
            # Grayscale -> RGB(3ch), Flip OK
            train_trans = [
                transforms.ToImage(), 
                transforms.RandomCrop(32, padding=6), 
                transforms.RandomHorizontalFlip(), 
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]
            val_trans = [
                transforms.ToImage(), 
                transforms.Pad(2), 
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]

        case "stl10":
            train_trans = [
                transforms.ToImage(), 
                transforms.RandomCrop(96, padding=12), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]
            val_trans = [
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]

        case "tiny-imagenet":
            train_trans = [
                transforms.ToImage(), 
                transforms.RandomCrop(64, padding=8), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]
            val_trans = [
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]

        case "caltech101" | "cub200" | "flowers102" | "stanford-cars" | "oxford-pet":
            # Fine-Grained, High-Res
            train_trans = [
                transforms.ToImage(), 
                GrayToRGB(),
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]
            val_trans = [
                transforms.ToImage(), 
                GrayToRGB(),
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), 
                transforms.CenterCrop(224), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]

        case _:
            # Fallback (ImageNet Style)
            print(f"[WARNING] No specific transforms found for {dataset_base}. Using default ImageNet-style transforms.")
            train_trans = [
                transforms.ToImage(), 
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]
            val_trans = [
                transforms.ToImage(), 
                transforms.Resize(256), 
                transforms.CenterCrop(224), 
                transforms.ToDtype(torch.float32, scale=True), 
                train_ds.normalizer()
            ]

    train_ds = train_ds.transform(train_trans).balance_class(seed=0)
    val_ds = val_ds.transform(val_trans)
    
    train_dl = train_ds.loader(batch_size, shuffle=True, num_workers=cfg["num_workers"])
    val_dl = val_ds.loader(batch_size, shuffle=False, num_workers=cfg["num_workers"])

    exp = ExpManager(exp_path=this_path.parent / exp_name, exp_tpl="exp_tpl_ee")
    run_mgr = exp.create_run()

    # Log Parameters
    run_mgr.log_param("model_arc", f"{net.__name__}")
    run_mgr.log_param("train_dataset", train_ds.state.dataset_id)
    run_mgr.log_param("val_dataset", val_ds.state.dataset_id)
    run_mgr.log_param("num_classes", num_classes := train_ds.fetch_classes())
    
    run_mgr.log_param("train_trans", repr(train_trans))
    run_mgr.log_param("val_trans", repr(val_trans))
    run_mgr.log_param("train_ndata", len(train_ds))
    run_mgr.log_param("val_ndata", len(val_ds))
    ipc_info = train_ds.get_ipc()
    run_mgr.log_param("ipc", ipc_info["mean"])
    run_mgr.log_param("ipc_is_balanced", ipc_info["is_balanced"])
    run_mgr.log_param("ipc_info", ipc_info)
    # run_mgr.log_param("ipc", ipc)

    run_mgr.log_param("epochs", epochs)
    run_mgr.log_param("max_lr", max_lr)
    run_mgr.log_param("wd", wd)
    run_mgr.log_param("batch_size", batch_size)

    run_mgr.log_param("base_epochs", cfg.get("base_epochs", None))
    run_mgr.log_param("iters/epoch", len(train_dl))
    run_mgr.log_param("iters", len(train_dl) * epochs)
    run_mgr.log_param("processed_ndata", len(train_ds) * epochs)
    run_mgr.log_param("div", div)
    run_mgr.log_param("ensembles", ens)
    run_mgr.log_param("scale_ch", scale_ch)
    run_mgr.log_text(src_name, src_text)

    refiner = EERefiner(net(num_classes=num_classes)).adapt_structure(target=dataset_base).multi_narrow(div=div, agg="mean", flex_ch=cfg["flex_ch"]).init_weights()
    network = Network(refiner.build().to(device))
    # network = Network(EERefiner(net(num_classes=num_classes)).adapt_structure(target=dataset_base).multi_narrow(div=div, agg="mean", flex_ch=cfg["flex_ch"]).init_weights().build().to(device))
    
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
        "refiner": refiner.history(),
        "num_params": network.param_stat(lambda p: p.numel()),
        "criterion": trainer.fmt_criterion(),
        "optimizer": trainer.fmt_optimizer(),
        "scheduler": trainer.fmt_scheduler(),
    }

    run_mgr.log_params(hp_dict)
    run_mgr.log_text("model_repr.txt", network.repr_network())
    run_mgr.log_text("model_torchinfo.txt", network.torchinfo(dl=train_dl))

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
        
        run_mgr.sync(step=e + 1, itv=epochs/100, last_step=epochs)
        trainer.printmet(mets, e + 1, epochs, itv=epochs / 5)

    run_mgr.sync()
    torch.save(trainer.network.state_dict(), run_mgr.fpath("state_dict.pt"))
    exp.ref_results()
    
    print(f"Experiment completed. Results saved to {run_mgr.run_path}")
