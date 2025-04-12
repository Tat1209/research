import sys

import torch
import polars as pl

work_path = "/home/haselab/Documents/tat/Research/"
sys.path.append(f"{work_path}app/torch_libs/")

from datasets import Datasets, dl
from run_manager_old import RunManager, RunsManager
from trainer import Model, MultiTrain
from trans import Trans
import utils

from mymodel import MyModel

from torchvision.models import efficientnet_v2_s as net

from optim.lion import Lion

for lr in [1e-5]:
    for wi in [3e-3, 5e-3]:
        print(f"{lr=}")
        ds = Datasets(root=f"{work_path}assets/datasets/")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # exp_name = "exp_tmp"
        # exp_name = "exp_submit_single"
        # exp_name = "exp_ft_test"
        # exp_name = "exp_ft_gs"
        exp_name = "exp_ft_lion"

        run_mgr = RunManager(exc_path=__file__, exp_name=exp_name)

        run_mgr.log_param("max_lr", max_lr := lr)
        run_mgr.log_param("weight_decay", wi)
        run_mgr.log_param("epochs", epochs := 2000)
        run_mgr.log_param("batch_size", batch_size := 128)
        run_mgr.log_param("mixup", mixup := True)

        run_mgr.log_param("train_trans", repr(train_trans := Trans.as_da3))
        run_mgr.log_param("val_trans", repr(val_trans := Trans.as_gen))

        train_ds, val_ds = ds("ai-step_l").split(ratio=1.0, shuffle=True)
        # train_ds, val_ds = ds("ai-step_l").split(ratio=0.8, shuffle=True, balance_label=True)

        # run_mgr.log_param("u_num_label", u_num_label := 4000)
        # train_ds = train_ds.transform(train_trans).mult_label({0: int(u_num_label / 2), 1: int(u_num_label / 931), 2: int(u_num_label / 159), 4: int(u_num_label / 20), 5: int(u_num_label / 3112)}).shuffle()
        train_ds = train_ds.transform(train_trans)
        val_ds = val_ds.transform(val_trans)

        # u_num_label = 4000
        # labeled_ds = ds("ai-step_l").mult_label({0: int(u_num_label / 2), 1: int(u_num_label / 931), 2: int(u_num_label / 159), 4: int(u_num_label / 20), 5: int(u_num_label / 3112)}).shuffle()
        # val_range = (0.9, 1.0)
        # train_ds = labeled_ds.ex_range(val_range).transform(train_trans)
        # val_ds = labeled_ds.in_range(val_range).transform(val_trans)

        unlabeled_ds = ds("ai-step_ul")
        test_ds = unlabeled_ds

        train_loader = dl(train_ds, batch_size, shuffle=True)
        val_loader = dl(val_ds, batch_size=2500, shuffle=True)
        # test_loader = dl(test_ds, batch_size=2500, shuffle=False)
        test_loaders = [
            dl(test_ds.transform(val_trans), batch_size=2500, shuffle=False),
            dl(test_ds.transform(val_trans + [Trans.rotate90]), batch_size=2500, shuffle=False),
            dl(test_ds.transform(val_trans + [Trans.rotate180]), batch_size=2500, shuffle=False),
            dl(test_ds.transform(val_trans + [Trans.rotate270]), batch_size=2500, shuffle=False),
            dl(test_ds.transform(val_trans + [Trans.hflip]), batch_size=2500, shuffle=False),
            dl(test_ds.transform(val_trans + [Trans.rotate90, Trans.hflip]), batch_size=2500, shuffle=False),
            dl(test_ds.transform(val_trans + [Trans.rotate180, Trans.hflip]), batch_size=2500, shuffle=False),
            dl(test_ds.transform(val_trans + [Trans.rotate270, Trans.hflip]), batch_size=2500, shuffle=False),
        ]

        run_mgr.log_param("num_data", len(train_loader.dataset))
        run_mgr.log_param("iters/epoch", iters_per_epoch := len(train_loader))
        run_mgr.log_param("dataset", train_loader.dataset.ds_str)
        run_mgr.log_param("model_arc", net.__module__)

        network = net(num_classes=7)
        loss_func = torch.nn.CrossEntropyLoss(weight=train_ds.fetch_weight(base_classes=True).to(device))
        optimizer = Lion(network.parameters(), lr=max_lr, weight_decay=wi)
        # optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=wi)
        scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

        model = MyModel(network, loss_func, optimizer, scheduler_t, device)
        run_mgr.log_param("start_weight", path := "/home/haselab/Documents/tat/Research/app/ai-step2/exp_tl/1/state_dict.pt")
        model.load_sd(path)

        run_mgr.log_param("params", model.count_params())
        run_mgr.log_param("params_trainable", model.count_trainable_params())
        hp_dict = {
            "loss_func": model.repr_loss_func(),
            "optimizer": model.repr_optimizer(),
            "scheduler": model.repr_scheduler(),
        }
        run_mgr.log_params(hp_dict)
        run_mgr.log_text(model.repr_network(), "model_layers.txt")
        run_mgr.log_text(model.arc_check(dl=train_loader), "model_structure.txt")

        for e in range(epochs):
            run_mgr.log_metric("lr", model.get_lr(), step=e + 1)

            train_loss, train_acc, train_f1 = model.train_1epoch(train_loader, mixup=mixup)
            val_loss, val_acc, val_f1 = model.val_1epoch(val_loader)

            met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}

            run_mgr.log_metrics(met_dict, step=e + 1)
            model.printlog(met_dict, e + 1, epochs, itv=epochs / 5)
            run_mgr.ref_stats(itv=5, step=e + 1, last_step=epochs)

        run_mgr.log_torch_save(model.get_sd(), "state_dict.pt")

        outputs, labels = model.pred_tta(test_loaders)
        # outputs, labels = model.pred_1iter(test_loader)

        df_out = pl.DataFrame({"fname": labels, "pred": outputs.tolist()})
        run_mgr.log_csv(df_out, "output.csv", has_header=False)

        outputs, labels = model.pred_tta(test_loaders, categorize=False)
        run_mgr.log_torch_save((outputs, labels), "output_t.pt")

        run_mgr.write_stats()
