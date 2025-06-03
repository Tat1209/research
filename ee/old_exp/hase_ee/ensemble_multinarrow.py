import torch
from torch import nn
import numpy as np
import wandb

import sys
import os

from utils.utils import load_config
from data.loader import get_dataloader, get_subset_dataloader

from models.modelutils import *
from models.ensemble_modelutils import *

def train(target_model, dname, runname, config, epochs, train_dl, test_dl, device):
    # WandBプロジェクトの作成とハイパーパラメータの設定
    run = wandb.init(
        project="EnsembleMultinarrow",
        group=dname,
        name=f"{runname}",
        config=config,
    )

    # 学習
    for epoch in range(epochs):
        # Training Phase 
        train_losses = target_model.train_one_epoch(train_dl, train=True, regularize=False, epoch=epoch)
        train_losses = {"train_"+k: v for k, v in train_losses.items()}

        #if epoch == epochs-1:  # 最終エポックのみValidationを実施
        with torch.no_grad():
            val_losses = target_model.train_one_epoch(test_dl, train=False, regularize=False, epoch=epoch)
        val_losses = {"val_"+k: v for k, v in val_losses.items()}
        train_losses.update(val_losses)
        
        wandb.log(train_losses)
        losres = " ".join([f"{k}_{v}" for k, v in train_losses.items()])
        print(f"Epoch: {epoch}, {losres}")

    # 選択したデバイスに関連付けられたメモリをクリア
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    run.finish()

def main():
    print(os.getcwd())
    num_gpus = torch.cuda.device_count()

    wandb.login()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = "../../datasets/"
    # YAML設定ファイルの読み込み
    config = load_config('configs/config_ensemble_mn.yaml')

    # ハイパーパラメータの取得
    optimizer_name = config['optimizer']
    max_lr = config['max_lr']
    scheduler_name = config['scheduler']
    batch_size = config['batch_size']
    epochs = config['epochs']
    config['num_gpus'] = num_gpus

    # コマンドライン引数の取得
    # 定義
    # 1. T: 温度パラメータ
    # 2. modelname: モデル名
    # 3. dataname: データセット名
    # 4. div: 分割数
    # 5. modeltype: comb, dist, lambda(0.5)
    # 6. pretrained: 学習済みモデルの使用有無
    # 7. num_subset: サブセットのサイズ
    T, modelname, dataname, div, = 1, "resnet50", "cifar100", 1
    modeltype, pretrained, num_subset = "comb", False, 5000
    scaling, crossing = False, True
    if sys.argv:
        T = float(sys.argv[1])
        modelname = sys.argv[2]
        dataname = sys.argv[3]
        div = int(sys.argv[4])
        modeltype = sys.argv[5]
        pretrained = sys.argv[6] == "True"
        num_subset = int(sys.argv[7])
    config['T'] = T
    config['modelname'] = modelname
    config['dataname'] = dataname
    config['div'] = div
    config['type'] = modeltype
    config['pretrained'] = pretrained
    config['num_subset'] = num_subset
    config['scaling'] = scaling
    config['crossing'] = crossing
    print(config)

    train_dl, test_dl = get_dataloader(dataname, root_dir, batch_size, num_gpus)
    if num_subset > 0:
        train_dl = get_subset_dataloader(train_dl, num_subset)

    # モデルの作成
    for_cifar_customize = False
    if (dataname == "cifar10") | (dataname == "cifar100"):
        num_classes = len(np.unique(train_dl.dataset.targets))  
        for_cifar_customize = True  
    else:
        num_classes = len(np.unique(train_dl.dataset.labels))

    single_model = create_model(modelname, num_classes, pretrained=pretrained, for_cifar_customize=for_cifar_customize)

    if div <= 1:
        model = EnsembleTrainer([single_model], lambda_=0.5, epsilon=0.0, type_=modeltype, 
                                type_regularizer="none", lr=0.1, T_max=epochs, 
                                device=device, total_epochs=epochs)
    else:
        modellist = []
        for i in range(div**2):
            modellist.append(create_model(modelname, num_classes, pretrained=False, for_cifar_customize=True, div=div))
        model = EnsembleTrainer(modellist, lambda_=0.5, epsilon=0.0, type_=modeltype, 
                                type_regularizer="none", lr=0.1, T_max=epochs,
                                device=device,total_epochs=epochs)
        transfer_weights(single_model, model, div=div, scale=False, cross=True)
    config['ensembles'] = div**2

    runname = f"{modeltype}_{div}"
    train(model, dataname, runname, config, epochs, train_dl, test_dl, device)

if __name__ == "__main__":
    main()