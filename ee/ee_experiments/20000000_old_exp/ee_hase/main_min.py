import sys

import torch
from torchvision import transforms

torchlib_path = "/path/to/torch_libs/"  ###################################### 要変更
sys.path.append(torchlib_path)

from datasets import Datasets
from trainer import Trainer, MultiTrainer

# from models.resnet_ee import resnet18 as net  # 公式実装 ee_ver.
from models.resnet_git_ee import resnet18 as net  # github実装 ee_ver. (githubのurlはモジュールを参照)

# データセットが含まれるディレクトリのパスを指定
ds = Datasets(root="/path/to/datasets/") # この場合 /path/to/datasets/ 直下にデータセット Ex.)cifar-100-python/ があればよい  ###################################### 要変更

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

max_lr = 0.0005
batch_size = 128

epochs = 100

fils = 64
ensembles = 1

train_trans = [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 360))]
val_trans = [transforms.ToTensor()]

train_ds = ds("cifar100_train", train_trans) # 第1引数に対応するデータセットを取得 対応する文字列はDatasetsクラスを参照
val_ds = ds("cifar100_val", val_trans)
# train_ds = ds("stl10_train", train_trans)
# val_ds = ds("stl10_val", val_trans)

train_dl = train_ds.loader(batch_size=128)
val_dl = val_ds.loader(batch_size=128)

network = net(num_classes=train_ds.fetch_classes(), nb_fils=fils, ee_groups=ensembles)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

trainer = Trainer(network, loss_func, optimizer, scheduler_t, device)

# 訓練 lossやaccを出力するならループ内のコメントアウトを解除
for e in range(epochs):
    train_loss, train_acc = trainer.train_1epoch(train_dl)
    # met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc}
    val_loss, val_acc = trainer.val_1epoch(val_dl)
    # met_dict.update({"val_loss": val_loss, "val_acc": val_acc})

    # trainer.printmet(met_dict, e + 1, epochs, itv=epochs / 4, timezone=0) # 仕様は実装コードに




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
