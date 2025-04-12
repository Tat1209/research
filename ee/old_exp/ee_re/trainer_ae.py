import sys
from pathlib import Path

import torch

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from trainer import TrainerUtils, Trainer

class AETrainer(Trainer):
    def __init__(
        self,
        network=None,
        loss_func=None,
        optimizer=None,
        scheduler_t=None,
        device=None,
    ):
        super().__init__(network=network, loss_func=loss_func, optimizer=optimizer, scheduler_t=scheduler_t, device=device)


    def train_1epoch(self, dl):
        self.network.train()
        total_loss = 0.0
        # total_corr = 0.0

        for inputs, labels in dl:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.network(inputs)
            
            batch_size = len(inputs)
            inputs_flat = inputs.view(batch_size, -1)
            outputs_flat = outputs.view(batch_size, -1)
            
            loss = self.loss_func(inputs_flat, outputs_flat)

            # _, pred = torch.max(outputs.detach(), dim=1)
            # corr = torch.sum(pred == labels.data).item()

            total_loss += loss.item() * len(inputs)
            # total_corr += corr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler_t is not None and self.scheduler_t[1] == "batch":
                self.scheduler_t[0].step()
        if self.scheduler_t is not None and self.scheduler_t[1] == "epoch":
            self.scheduler_t[0].step()

        train_loss = total_loss / len(dl.dataset)
        # train_acc = total_corr / len(dl.dataset)

        return train_loss

    def val_1epoch(self, dl):
        if dl is None:
            return None, None

        self.network.eval()
        total_loss = 0.0
        # total_corr = 0.0

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.network(inputs)
                
                inputs_flat = inputs.view(128, -1)
                outputs_flat = outputs.view(128, -1)
                
                loss = self.loss_func(inputs_flat, outputs_flat)

                # _, pred = torch.max(outputs.detach(), dim=1)
                # corr = torch.sum(pred == labels.data).item()

                total_loss += loss.item() * len(inputs)
                # total_corr += corr

        val_loss = total_loss / len(dl.dataset)
        # val_acc = total_corr / len(dl.dataset)

        return val_loss

    def fetch_embs(self, dl):
        self.network.eval()
        total_outputs = None
        total_labels = None

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                outputs = self.network.encoder(inputs)
                outputs = outputs.detach()

                if total_outputs is None:
                    total_outputs = outputs
                else:
                    total_outputs = torch.cat((total_outputs, outputs), dim=0)

                if total_labels is None:
                    total_labels = labels
                else:
                    total_labels = torch.cat((total_labels, labels), dim=0)

        return total_outputs, total_labels


    # def pred_1iter(self, dl, categorize=True):
    #     self.network.eval()
    #     total_outputs = None
    #     total_labels = None

    #     with torch.no_grad():
    #         for inputs, labels in dl:
    #             inputs = inputs.to(self.device)
    #             output = self.network(inputs)
    #             output = output.detach()

    #             if categorize:
    #                 _, pred = torch.max(output, dim=1)
    #                 output = pred

    #             if total_outputs is None:
    #                 total_outputs = output
    #             else:
    #                 total_outputs = torch.cat((total_outputs, output), dim=0)

    #             if total_labels is None:
    #                 total_labels = labels
    #             else:
    #                 total_labels.extend(labels)

    #     return total_outputs, total_labels


class AEMultiTrain(TrainerUtils):
    def __init__(self, models=None, device=None):
        self.models = models
        super().__init__(device)

    def train_1epoch(self, dl):
        for model in self.models:
            model.network.train()
        total_losses = [0.0 for _ in self.models]
        # total_corrs = [0.0 for _ in self.models]

        for inputs, labels in dl:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            batch_size = len(inputs)
            outputs = [model.network(inputs).view(batch_size, -1) for model in self.models]

            inputs = inputs.view(batch_size, -1)
            # outputs = outputs.view(batch_size, -1)

            losses = [self.models[m].loss_func(inputs, outputs[m]) for m in range(len(self.models))]

            losses_add = [loss.item() * len(inputs) for loss in losses]

            total_losses = [a + b for a, b in zip(total_losses, losses_add)]

            for model in self.models:
                model.optimizer.zero_grad()

            for loss in losses:
                loss.backward()

            for model in self.models:
                model.optimizer.step()

            for model in self.models:
                if model.scheduler_t is not None and model.scheduler_t[1] == "batch":
                    model.scheduler_t[0].step()
        for model in self.models:
            if model.scheduler_t is not None and model.scheduler_t[1] == "epoch":
                model.scheduler_t[0].step()

        total_losses = [loss / (len(dl.dataset) * len(self.models)) for loss in total_losses]

        return total_losses
        # return list(zip(total_losses, total_accs))


    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return_l = []
            for i, model in enumerate(self.models):
                new_args = [arg[i] if isinstance(arg, list) and len(arg) == len(self.models) else arg for arg in args]
                new_kwargs = {k: v[i] if isinstance(v, list) and len(v) == len(self.runs) else v for k, v in kwargs.items()}
                return_l.append(getattr(model, attr)(*new_args, **new_kwargs))
            return return_l

        return wrapper