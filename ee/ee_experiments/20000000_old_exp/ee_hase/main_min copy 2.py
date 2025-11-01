import sys
from pathlib import Path
import math

import torch
from torchvision import transforms

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets
from trainer import Trainer, MultiTrainer
from trans import Trans

# from models.resnet_ee import resnet18 as net
from models.resnet_git_ee import resnet18 as net

ds = Datasets(root=work_path / "assets/datasets/")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

max_lr = 0.1
batch_size = 128

epochs = 200

fils = 64
ensembles = 1

train_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize(mean=[0.5070751905441284, 0.48654890060424805, 0.44091784954071045], std=[0.2673342823982239, 0.2564384639263153, 0.2761504650115967], inplace=True)]
val_trans = [transforms.ToTensor(), transforms.Normalize(mean=[0.5070751905441284, 0.48654890060424805, 0.44091784954071045], std=[0.2673342823982239, 0.2564384639263153, 0.2761504650115967], inplace=True)]

train_ds = ds("cifar100_train", train_trans) # 第1引数に対応するデータセットを取得 対応する文字列はDatasetsクラスを参照
val_ds = ds("cifar100_val", val_trans)
# train_ds = ds("stl10_train", train_trans)
# val_ds = ds("stl10_val", val_trans)

train_dl = train_ds.loader(batch_size=128)
val_dl = val_ds.loader(batch_size=128)

network = net(num_classes=train_ds.fetch_classes(), nb_fils=fils, ee_groups=ensembles)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

trainer = Trainer(network, loss_func, optimizer, scheduler_t, device)

# 訓練 lossやaccを出力するならループ内のコメントアウトを解除
for e in range(epochs):
    train_loss, train_acc = trainer.train_1epoch(train_dl)
    met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc}
    val_loss, val_acc = trainer.val_1epoch(val_dl)
    met_dict.update({"val_loss": val_loss, "val_acc": val_acc})

    trainer.printmet(met_dict, e + 1, epochs, itv=epochs / 4, timezone=0) # 仕様は実装コードに




# 単一loaderで複数のモデルを訓練するサンプル

# trainers = []
# fils_l = [64, 16]
# ensembles_l = [1, 16]

# for fils, ensembles in zip(fils_l, ensembles_l):
#     network = net(num_classes=train_ds.fetch_classes(), nb_fils=fils, ee_groups=ensembles)

#     loss_func = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
#     scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

#     trainer = Trainer(network, loss_func, optimizer, scheduler_t, device)
#     trainers.append(trainer)
# mtrainer = MultiTrainer(trainers, device)

# for e in range(epochs):
#     train_loss, train_acc = mtrainer.train_1epoch(train_dl) # それぞれ値が格納されたリストがreturnされる
#     met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc}

#     val_loss, val_acc = mtrainer.val_1epoch(val_dl)
#     met_dict.update({"val_loss": val_loss, "val_acc": val_acc})

#     mtrainer.printmet(met_dict, e + 1, epochs, itv=epochs / 4, anyval=True)
