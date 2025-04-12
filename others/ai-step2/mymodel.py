import sys

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import seaborn

work_path = "/home/haselab/Documents/tat/Research/"
sys.path.append(f"{work_path}app/torch_libs/")

from trainer import Trainer, Model

from sklearn.metrics import confusion_matrix, precision_score

# from torchvision.transforms import v2


class MyModel(Model):
    def __init__(self, network, loss_func, optimizer, scheduler_t=None, device=None):
        super().__init__(network, loss_func, optimizer, scheduler_t, device)
        # self.cutmix = v2.CutMix(num_classes=7)
        # inputs, labels = self.cutmix(inputs, labels)

    def train_1epoch(self, dl, mixup=False):
        self.network.train()
        total_loss = 0.0
        total_corr = 0.0

        pred_list = None
        true_list = None

        for inputs, labels in dl:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if mixup:
                lmd = np.random.beta(0.2, 0.2)
                perm = torch.randperm(inputs.shape[0]).to(self.device)
                inputs2 = inputs[perm, :]
                labels2 = labels[perm]
                inputs = lmd * inputs + (1.0 - lmd) * inputs2

            output = self.network(inputs)

            if mixup:
                loss = lmd * self.loss_func(output, labels) + (1.0 - lmd) * self.loss_func(output, labels2)
            else:
                loss = self.loss_func(output, labels)

            _, pred = torch.max(output.detach(), dim=1)

            if mixup:
                corr = (lmd * torch.sum(pred == labels) + (1.0 - lmd) * torch.sum(pred == labels2)).item()
            else:
                corr = torch.sum(pred == labels).item()

            if pred_list is None:
                pred_list = pred
            else:
                pred_list = torch.cat([pred_list, pred])

            if true_list is None:
                true_list = labels
            else:
                true_list = torch.cat([true_list, labels])

            total_loss += loss.item() * len(inputs)
            total_corr += corr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler_t is not None and self.scheduler_t[1] == "batch":
                self.scheduler_t[0].step()
        if self.scheduler_t is not None and self.scheduler_t[1] == "epoch":
            self.scheduler_t[0].step()

        train_loss = total_loss / len(dl.dataset)
        train_acc = total_corr / len(dl.dataset)
        true_list = true_list.cpu()
        pred_list = pred_list.cpu()
        train_f1 = precision_score(y_true=true_list, y_pred=pred_list, average="macro", zero_division=0.0)

        return train_loss, train_acc, train_f1

    def val_1epoch(self, dl):
        if dl is None:
            return None, None, None

        self.network.eval()
        total_loss = 0.0
        total_corr = 0.0

        pred_list = None
        true_list = None

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                output = self.network(inputs)
                loss = self.loss_func(output, labels)

                _, pred = torch.max(output.detach(), dim=1)
                corr = torch.sum(pred == labels.data).item()

                total_loss += loss.item() * len(inputs)
                total_corr += corr

                if pred_list is None:
                    pred_list = pred
                else:
                    pred_list = torch.cat([pred_list, pred])

                if true_list is None:
                    true_list = labels
                else:
                    true_list = torch.cat([true_list, labels])

        val_loss = total_loss / len(dl.dataset)
        val_acc = total_corr / len(dl.dataset)
        true_list = true_list.cpu()
        pred_list = pred_list.cpu()

        val_f1 = precision_score(y_true=true_list, y_pred=pred_list, average="macro", zero_division=0.0)

        return val_loss, val_acc, val_f1

    def pred_1iter(self, dl, categorize=True):
        self.network.eval()
        total_output = None
        total_label = None

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                output = self.network(inputs)
                output = output.detach()
                output = torch.nn.Softmax(dim=1)(output)

                if categorize:
                    _, pred = torch.max(output, dim=1)
                    output = pred

                if total_output is None:
                    total_output = output
                else:
                    total_output = torch.cat((total_output, output), dim=0)

                if total_label is None:
                    total_label = labels
                else:
                    total_label.extend(labels)

            # sorted_pairs = sorted(zip(total_label, total_output))   # total_labelを基準にそろえるため，この順番でok
            # total_label, total_output = zip(*sorted_pairs)

        return total_output, total_label

    def pred_tta(self, dls, categorize=True):
        # dlsに含まれるloaderはすべてshuffle=Falseにしておくこと
        tta_output = None
        tta_label = None
        for dl in dls:
            total_output, total_label = self.pred_1iter(dl, categorize=False)
            if tta_output is None:
                tta_output = total_output
                tta_label = total_label
            else:
                tta_output += total_output
        tta_output /= len(dls)

        if categorize:
            _, pred = torch.max(tta_output, dim=1)
            tta_output = pred

        return tta_output, tta_label


class Ens(Trainer):
    def __init__(self, models=None, device=None):
        self.models = models
        super().__init__(device)

    def train_1epoch(self, dls, mixup=False):
        # 内包表記で一行で書けなくもない
        total_losses = []
        total_accs = []
        total_f1s = []

        for model, dl in zip(self.models, dls):
            train_loss, train_acc, train_f1 = model.train_1epoch(dl, mixup=mixup)
            total_losses.append(train_loss)
            total_accs.append(train_acc)
            total_f1s.append(train_f1)

        return total_losses, total_accs, total_f1s

    def val_1epoch(self, dl):
        if dl is None:
            return None, None, None

        for model in self.models:
            model.network.eval()

        total_loss = 0.0
        total_corr = 0.0

        pred_list = None
        true_list = None

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = [model.network(inputs) for model in self.models]
                losses = [self.models[m].loss_func(outputs[m], labels) for m in range(len(self.models))]
                output = torch.mean(torch.stack(outputs), dim=0)

                _, pred = torch.max(output.detach(), dim=1)
                corr = torch.sum(pred == labels.data).item()

                total_loss += sum([loss.item() for loss in losses]) * len(inputs)
                total_corr += corr

                if pred_list is None:
                    pred_list = pred
                else:
                    pred_list = torch.cat([pred_list, pred])

                if true_list is None:
                    true_list = labels
                else:
                    true_list = torch.cat([true_list, labels])

        val_loss = total_loss / len(dl.dataset)
        val_acc = total_corr / len(dl.dataset)
        true_list = true_list.cpu()
        pred_list = pred_list.cpu()

        val_f1 = precision_score(y_true=true_list, y_pred=pred_list, average="macro", zero_division=0.0)

        return val_loss, val_acc, val_f1

    def pred_1iter(self, dl, categorize=True):
        # dlはshuffle=Falseにしておくこと
        ens_output = None
        ens_label = None
        for model in self.models:
            total_output, total_label = model.pred_1iter(dl)
            if ens_output is None:
                ens_output = total_output
                ens_label = total_label
            else:
                ens_output += total_output
        ens_output /= len(self.models)

        if categorize:
            _, pred = torch.max(ens_output, dim=1)
            ens_output = pred

        return ens_output, ens_label

    def pred_tta(self, dls, categorize=True):
        # dlsに含まれるloaderはすべてshuffle=Falseにしておくこと
        tta_output = None
        tta_label = None
        for dl in dls:
            total_output, total_label = self.pred_1iter(dl, categorize=False)
            if tta_output is None:
                tta_output = total_output
                tta_label = total_label
            else:
                tta_output += total_output
        tta_output /= len(dls)

        if categorize:
            _, pred = torch.max(tta_output, dim=1)
            tta_output = pred

        return tta_output, tta_label

    def get_sd(self):
        return [model.get_sd() for model in self.models]

    def load_sd(self, sd_list):
        (self.models[i].load_sd(sd) for i, sd in enumerate(sd_list))

    def count_params(self):
        return sum(model.count_params() for model in self.models)

    def count_trainable_params(self):
        return sum(model.count_trainable_params() for model in self.models)

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return_l = []
            for i, model in enumerate(self.models):
                new_args = [arg[i] if isinstance(arg, list) and len(arg) == len(self.models) else arg for arg in args]
                new_kwargs = {k: v[i] if isinstance(v, list) and len(v) == len(self.runs) else v for k, v in kwargs.items()}
                return_l.append(getattr(model, attr)(*new_args, **new_kwargs))
            return return_l

        return wrapper
