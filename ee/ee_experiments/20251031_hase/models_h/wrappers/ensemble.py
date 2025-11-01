import torch
import torch.nn as nn

from .base import initialize_weights
from ..easy_ensemble import GroupedLinearV2


class Ensembler(nn.Module):
    def __init__(self, backbones, T=1, need_fcs=False):
        super(Ensembler, self).__init__()
        self.backbones = nn.ModuleList(backbones)
        self.T = T
        self.need_fcs = need_fcs
        if self.need_fcs:
            in_features = sum([backbone.fc.in_features for backbone in backbones])
            self.fc = nn.Sequential(nn.BatchNorm1d(in_features, affine = False),
                                    nn.Linear(in_features, 200))
            initialize_weights(self.fc)
            for i in range(len(backbones)):
                backbones[i].fc = nn.Identity()
        
    def forward(self, x):
        x = [backbone(x) for backbone in self.backbones]
        if self.need_fcs:
            x = torch.cat([torch.flatten(i, 1) for i in x], dim=1)
            x = self.fc(x) / self.T
        else:
            x = torch.stack(x, dim=1)
            x = torch.sum(x, dim=1) / self.T
        return x
    
# ModelWrapperをアンサンブルする
class EnsembleWrapper(nn.Module):
    def __init__(self, models):
        super(EnsembleWrapper, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        logits_list = [model(x) for model in self.models]
        if len(logits_list) == 1:
            return logits_list[0]
        else:
            logits = torch.stack(logits_list, dim=1)
            return torch.mean(logits, dim=1)

    def get_features(self, x):
        return [model.get_features(x) for model in self.models]
    
    def get_logits(self, x):
        return [model.get_logits(x) for model in self.models]       
        
    def get_features_and_logits(self, x):
        temp = [model.get_features_and_logits(x) for model in self.models]
        features = [item[0] for item in temp]
        logits = [item[1] for item in temp]
        return features, logits
    
    def __len__(self):
        return len(self.models)


# 分類器だけアンサンブルするモデルラッパー
class HeadEnsembleWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_ensembles: int, num_classes: int, last_out_channels: int):
        super().__init__()
        self.model = model # 通常のbackboneモデル(fc層なし)
        self.num_classes = num_classes
        self.num_ensembles = num_ensembles
        self.last_out_channels = last_out_channels
        self.fc = GroupedLinearV2(
            in_features=last_out_channels,
            out_features=num_classes * num_ensembles,
            num_groups=num_ensembles,
            bias=True
        )
    
    def forward(self, x):
        # 入力をチャネル方向にN回繰り返す
        logits = self.get_logits(x)
        logits = torch.stack(logits, dim=1)  # N個の出力
        output = logits.mean(dim=1)  # 平均化        
        return output
        
    def get_features_and_logits(self, x):
        bs = x.size(0)
        features = self.model(x)
        logits = self.fc(features).view(bs, self.num_ensembles, self.num_classes)
        features = features.view(bs, self.num_ensembles, -1)
        # Listにして返す
        return [features[:,i,:] for i in range(self.num_ensembles)], \
               [logits[:,i,:] for i in range(self.num_ensembles)]
    
    def get_features(self, x):
        f, l = self.get_features_and_logits(x)
        return f
    
    def get_logits(self, x):
        f, l = self.get_features_and_logits(x)
        return l
    
    def __len__(self):
        """モデル数を返す"""
        return self.num_ensembles
    