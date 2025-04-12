import sys
from pathlib import Path
import math

import torch
from torchvision import transforms

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from trainer import Trainer

class MyTrainer(Trainer):
    def __init__(
        self,
        network=None,
        loss_func=None,
        optimizer=None,
        scheduler_t=None,
        device=None,
    ):
        super().__init__(network, loss_func, optimizer, scheduler_t, device)
        
    def fetch_feat(self, dl, layer_list=None, flatten=False):
        self.network.eval()
        feature_maps = {}
        hooks = []

        def hook_fn(name):
            def hook(module, input, output):
                output = output.clone().to("cuda:0")
                if flatten:
                    output = output.view(len(output), -1)

                if feature_maps.get(name) is None:
                    feature_maps[name] = output
                else:
                    feature_maps[name] = torch.cat([feature_maps[name], output], dim=0)
                    
                # print(feature_maps[name].shape)

                # print(name)
                # print(output.shape)
            return hook

        def register_hooks(module, parent_name="", rec=True):
            for name, layer in module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                if layer_list is None  or  full_name in layer_list:
                    hooks.append(layer.register_forward_hook(hook_fn(full_name)))  # 全てのレイヤーにフックを登録

                if rec:
                    register_hooks(layer, full_name, rec=True)  # 再帰的に内部モジュールにもフックを登録

        # def register_hooks(module, parent_name=""):
        #     for name, layer in module.named_children():
        #         full_name = f"{parent_name}.{name}" if parent_name else name
        #         layer.register_forward_hook(hook_fn(full_name))  # 全てのレイヤーにフックを登録
        #         register_hooks(layer, full_name)  # 再帰的に内部モジュールにもフックを登録

        register_hooks(self.network) # モデル全体にフックを登録

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.network(inputs)
                outputs = outputs.detach()

                if "logit" in layer_list:
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                    if flatten:
                        outputs = outputs.view(len(outputs), -1)

                    if feature_maps.get("logit") is None:
                        feature_maps["logit"] = outputs
                        feature_maps["labels"] = labels
                    else:
                        feature_maps["logit"] = torch.cat([feature_maps["logit"], outputs], dim=0)
                        feature_maps["labels"] = torch.cat([feature_maps["labels"], labels], dim=0)

                # feat = feat.cpu()
                # labels = labels.cpu()

        for hook in hooks:
            hook.remove() # モデル全体にフックを登録
                
        for k, v in feature_maps.items():
            feature_maps[k] = v.cpu()

        return feature_maps