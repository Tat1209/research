import torch
import numpy as np
from time import time
import datetime
import torchvision
import pandas as pd


class Model:
    def __init__(self, pr, network, epochs, learning_rate, log_itv=10, fit_aug_ratio=None, mixup_alpha=None, pred_times=1, tta_aug_ratio=None): 
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

        self.pr = pr
        self.network = network.to(self.device)                                                        
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.log_itv=log_itv
        self.fit_aug_ratio=fit_aug_ratio
        self.mixup_alpha=mixup_alpha
        self.pred_times=pred_times
        self.tta_aug_ratio=tta_aug_ratio

        self.loss_func = torch.nn.CrossEntropyLoss()                                                          # 損失関数の設定
        self.optimizer = torch.optim.RAdam(self.network.parameters(), lr=self.learning_rate)             # 最適化手法の設定
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=1, eta_min=0.)


    def fit(self):

        hist = dict()
        hist["Epoch"] = [i+1 for i in range(self.epochs)]

        start = time()
        for epoch in range(self.epochs):
            if epoch != 0:
                stop = time()
                req_time = (stop-start) / epoch * self.epochs
                left = start + req_time - stop
                eta = (datetime.datetime.now() + datetime.timedelta(seconds=left) + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
                t_hour, t_min = divmod(left//60, 60)
                left = f"{int(t_hour):02d}:{int(t_min):02d}"

            # augmentationの場合分け augする場合は回転処理とmixupを両方行う
            if self.fit_aug_ratio is None: 
                if epoch == 0:
                    dl_train = self.pr.fetch_train(self.pr.tr.gen)
                    mixup = False
            else: 
                if epoch/self.epochs < self.fit_aug_ratio:
                    dl_train = self.pr.fetch_train(self.pr.tr.aug)
                    mixup = self.mixup_alpha is not None
                else:
                    dl_train = self.pr.fetch_train(self.pr.tr.gen)
                    mixup = False

            dl_val = self.pr.fetch_val(self.pr.tr.gen)

            summary = dict()

            stats_t = self.train_1epoch(dl_train, mixup=mixup)
            summary["Loss"], summary["Acc"] = stats_t["total_loss"], stats_t["total_corr"]

            stats_v = self.val_1epoch(dl_val)
            summary["vLoss"], summary["vAcc"] = stats_v["total_loss"], stats_v["total_corr"]

            for key, value in summary.items():
                if epoch == 0: hist[key] = [value]
                else: hist[key].append(value)

            disp_str = f'Epoch: {epoch+1:>4}/{self.epochs:>4}'      # 本当はsummary["Epoch"][epoch]とかがいいけどだるい
            for key, value in summary.items(): disp_str += f"    {key}: {value:<9.7f}"
            if epoch != 0: disp_str += f"    eta: {eta} (left: {left})"

            if (epoch+1) % self.log_itv == 0 or (epoch+1) == self.epochs: print(disp_str)
            else: print(disp_str, end="\r")

        return hist
        # histは、次の要素が入ったディクショナリ ... {"Epoch", "Loss", "Acc", "vLoss", "vAcc"}
    

    def train_1epoch(self, dl, mixup=False):
        self.network.train()  # モデルを訓練モードにする
        stats = {"total_loss":0., "total_corr":0.}

        for input_b, label_b in dl:
            input_b = input_b.to(self.device)
            label_b = label_b.to(self.device)

            if mixup:
                lmd = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                perm = torch.randperm(input_b.shape[0]).to(self.device)
                input2_b = input_b[perm, :]
                label2_b = label_b[perm]
                
                input_b = lmd * input_b  +  (1.0 - lmd) * input2_b
                output_b = self.network(input_b)
                loss_b = lmd * self.loss_func(output_b, label_b)  +  (1.0 - lmd) * self.loss_func(output_b, label2_b)
                stats["total_loss"] += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                stats["total_corr"] += (lmd * torch.sum(pred == label_b) + (1.0 - lmd) * torch.sum(pred == label2_b)).cpu().numpy()

            else: 
                output_b = self.network(input_b)
                loss_b = self.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
                stats["total_loss"] += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                stats["total_corr"] += torch.sum(pred == label_b.data).item()


            self.optimizer.zero_grad()              # optimizerを初期化 前バッチで計算した勾配の値を0に
            loss_b.backward()                         # 誤差逆伝播 勾配計算
            self.optimizer.step()                   # 重み更新して計算グラフを消す
        self.scheduler.step()

        # avg_loss = stats["total_loss"] / len(dl.dataset)
        # acc = stats["total_corr"] / len(dl.dataset)

        stats["total_loss"] /= len(dl.dataset)
        stats["total_corr"] /= len(dl.dataset)

        return stats
        # return avg_loss, acc


    def val_1epoch(self, dl):
        self.network.eval()  # モデルを評価モードにする
        stats = {"total_loss":0., "total_corr":0., "outputs":None, "labels":None}

        with torch.no_grad():
            for input_b, label_b in dl:
                input_b = input_b.to(self.device)
                label_b = label_b.to(self.device)

                output_b = self.network(input_b)
                loss_b = self.loss_func(output_b, label_b)  # 損失(出力とラベルとの誤差)の定義と計算 tensor(scalar, device, grad_fn)のタプルが返る
                stats["total_loss"] += loss_b.item()*len(input_b) # .item()で1つの値を持つtensorをfloatに
                _, pred = torch.max(output_b.detach(), dim=1)
                stats["total_corr"] += torch.sum(pred == label_b.data).item()

                output_b = output_b.detach().cpu().numpy()
                label_b = label_b.detach().cpu().numpy()
                if stats["outputs"] is None: stats["outputs"] = output_b
                else: stats["outputs"] = np.concatenate((stats["outputs"], output_b), axis=0)

        stats["total_loss"] /= len(dl.dataset)
        stats["total_corr"] /= len(dl.dataset)

        return stats


    def pred_1iter(self, dl, label=False):
        stats = {"outputs":None, "labels": None}
        with torch.no_grad():
            for input_b, label_b in dl:
                input_b = input_b.to(self.device)
                output_b = self.network(input_b)
                output_b = output_b.detach().cpu().numpy()

                if stats["outputs"] is None: stats["outputs"] = output_b
                else: stats["outputs"] = np.concatenate((stats["outputs"], output_b), axis=0)

                if label:
                    if stats["labels"] is None: stats["labels"] = label_b
                    else: stats["labels"] = np.concatenate((stats["labels"], label_b), axis=0)

        return stats


    def pred(self, categorize=True, val=False):
        summary = {"outputs":None, "labels":None}

        if val: fetch_t = self.pr.fetch_val
        else: fetch_t = self.pr.fetch_test
        label_flag = True

        if self.pred_times is None: self.pred_times = 1
        for i in range(self.pred_times):
            if self.tta_aug_ratio is not None:
                if i/self.pred_times < self.tta_aug_ratio: dl = fetch_t(self.pr.tr.aug)
                else: dl = fetch_t(self.pr.tr.flip_aug)
            else: dl = fetch_t(self.pr.tr.gen)

            stats_p = self.pred_1iter(dl, label=label_flag)

            # 出力に対してsoftmaxを適用 その後に平均をとる
            stats_p["outputs"] = torch.nn.functional.softmax(torch.from_numpy(stats_p["outputs"]), dim=1).numpy()
            if summary["outputs"] is None: summary["outputs"] = stats_p["outputs"]
            else: summary["outputs"] += stats_p["outputs"]
            
            if label_flag:
                summary["labels"] = stats_p["labels"]
                label_flag = False

        summary["outputs"] /= self.pred_times
        if categorize: summary["outputs"] = np.argmax(summary["outputs"], axis=1)
        return summary

        # summaryは、次の要素が入ったディクショナリ ... {"outputs", "labels"}


    def save_model(self, fname=None, fname_head="competition_model_"):
        if fname is None:
            date = datetime.datetime.now().strftime("%m%d_%H%M%S")
            if fname_head is None: fname = f"{date}.pth"
            else: fname = f"{fname_head}{date}.pth"
        torch.save(self.network, fname)
        
    
    def hist_to_csv(self, hist, fname=None, fname_head="competition_hist_"):
        if isinstance(hist, pd.DataFrame): df = hist
        else: df = pd.DataFrame(hist)

        if fname is None:
            date = datetime.datetime.now().strftime("%m%d_%H%M%S")
            if fname_head is None: fname = f"{date}.csv"
            else: fname = f"{fname_head}{date}.csv"
        df.to_csv(fname, index=False)
        
    
    def result_to_csv(self, result, fname=None, fname_head="competition_result_"):
        df = pd.DataFrame(result["outputs"], columns=range(result["outputs"].shape[1]))
        df["labels"] = result["labels"]

        if fname is None:
            date = datetime.datetime.now().strftime("%m%d_%H%M%S")
            if fname_head is None: fname = f"{date}.csv"
            else: fname = f"{fname_head}{date}.csv"
        df.to_csv(fname, index=False)
        

    def result_to_out(self, result, fname=None, fname_head="competition_result_"):
        df = pd.DataFrame()
        df["labels"] = result["labels"]
        df["labels"] = df["labels"].str[42:]
        df["outputs"] = result["outputs"]

        if fname is None:
            date = datetime.datetime.now().strftime("%m%d_%H%M%S")
            if fname_head is None: fname = f"{date}.csv"
            else: fname = f"{fname_head}{date}.csv"
        df.to_csv(fname, index=False, header=False)
        
    



