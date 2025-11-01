import sys
from pathlib import Path

import torch
from torchvision import transforms
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from spectral_metric.estimator import CumulativeGradientEstimator
from spectral_metric.visualize import make_graph

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets, dl
from run_manager import RunManager, RunsManager, RunViewer

from models.ae_csg import AutoEncoder as net
from trainer_ae import AETrainer, AEMultiTrain

ds = Datasets(root=work_path / "assets/datasets/")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

exp_name = "exp_csg_64"
# exp_name = "exp_csg_tmp"


epochs = 100
batch_size = 128

for max_lrs in [[0.001]]:
    runs = [RunManager(exc_path=__file__, exp_name=exp_name) for _ in max_lrs]
    runs_mgr = RunsManager(runs)

    train_trans = [transforms.ToTensor()]
    # train_trans = [transforms.Resize((64, 64)), transforms.ToTensor()]
    # val_trans = [transforms.ToTensor()]

    train_ds = ds("stl10_train", transform_l=train_trans)
    # train_ds = ds("cifar10_train", transform_l=train_trans).balance_label(seed=0).in_ratio(0.1)
    train_ds.fetch_ld(output=True)
    ae_dl = dl(train_ds, batch_size=batch_size)

    runs_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")

    runs_mgr.log_param("train_dataset", train_ds.ds_str)
    # runs_mgr.log_param("val_dataset", val_ds.ds_str)
    runs_mgr.log_param("num_classes", num_classes:=train_ds.fetch_classes())

    runs_mgr.log_param("train_trans", repr(train_trans))
    # runs_mgr.log_param("val_trans", repr(val_trans))

    runs_mgr.log_param("train_num", len(train_ds))
    # runs_mgr.log_param("val_num", len(val_ds))

    runs_mgr.log_param("epochs", epochs)
    runs_mgr.log_param("max_lr", max_lrs)
    runs_mgr.log_param("batch_size", batch_size)

    train_dl = dl(train_ds, batch_size, shuffle=True)
    # val_dl = dl(val_ds, batch_size, shuffle=True)

    runs_mgr.log_param("iters/epoch", iters_per_epoch := len(train_dl))

    models = []
    for max_lr in max_lrs:
        network = net(train_ds[0][0].shape, force=False)
        # loss_func = torch.nn.CrossEntropyLoss()
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=max_lr, eps=1e-07)
        scheduler_t = None
        # scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

        model = AETrainer(network, loss_func, optimizer, scheduler_t, device)
        models.append(model)
    mmodel = AEMultiTrain(models, device)

    runs_mgr.log_param("params", mmodel.count_params())
    hp_dict = {
            "loss_func": mmodel.repr_loss_func(),
            "optimizer": mmodel.repr_optimizer(),
            "scheduler": mmodel.repr_scheduler(),
            }

    runs_mgr.log_params(hp_dict)
    runs_mgr.log_text(mmodel.repr_network(), "model_layers.txt")
    runs_mgr.log_text(mmodel.arc_check(dl=train_dl), "model_structure.txt")

    for e in range(epochs):
        runs_mgr.log_metric("lr", mmodel.get_lr(), step=e + 1)

        train_loss = mmodel.train_1epoch(train_dl)

        met_dict = {"epoch": e + 1, "train_loss": train_loss}

        runs_mgr.log_metrics(met_dict, step=e + 1)
        mmodel.printlog(met_dict, e + 1, epochs, itv=epochs / 4)
        runs_mgr.ref_stats(itv=5, step=e + 1, last_step=epochs)

    runs_mgr.log_torch_save(mmodel.get_sd(), "ae_sd.pt")


    emb_dl = dl(train_ds, batch_size=128)
    out = mmodel.fetch_embs(emb_dl)

    for i, model_out in enumerate(out):

        embs, labels = model_out

        X = embs.view(embs.shape[0], -1).cpu().numpy()  # 画像をフラット化してnumpy配列に変換
        tsne = TSNE(n_components=3, random_state=0, verbose=0, n_jobs=2, n_iter=10000)
        X = tsne.fit_transform(X)
        y = labels.numpy()  # ラベルをnumpy配列に変換

        estimator = CumulativeGradientEstimator(M_sample=100, k_nearest=3)
        estimator.fit(data=X, target=y)

        print(estimator.csg)
        runs_mgr[i].log_param("csg", estimator.csg[0])

    runs_mgr.ref_stats()

    rv = RunViewer(exc_path=__file__, exp_name=exp_name)
    rv.ref_results()

