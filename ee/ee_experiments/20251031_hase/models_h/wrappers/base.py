import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial

def _initialize_one(module: nn.Module, is_ee: bool = False, 
                    linear_nonlinearity: str = 'relu'):
    # Linear
    if isinstance(module, nn.Linear):
        # 活性化に応じて初期化を切替（ReLU 前提なら Kaiming）
        init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity=linear_nonlinearity)
        if module.bias is not None:
            init.constant_(module.bias, 0.0)

    # BatchNorm(1d/2d/3d すべて)
    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        if module.weight is not None:
            init.constant_(module.weight, 1.0)
        if module.bias is not None:
            init.constant_(module.bias, 0.0)

    # GroupNorm / LayerNorm（必要なら追加）
    elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
        if module.weight is not None:
            init.constant_(module.weight, 1.0)
        if module.bias is not None:
            init.constant_(module.bias, 0.0)

    # Conv(1d/2d/3d すべて)
    elif isinstance(module, nn.modules.conv._ConvNd):
        if is_ee and module.groups > 1:
            # 「グループごとに別乱数で」のこだわりがある場合のみ
            chunks = module.weight.chunk(module.groups, dim=0)
            for w in chunks:
                init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        else:
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0.0)

def initialize_weights(model: nn.Module, *, is_ee: bool = False, 
                       linear_nonlinearity: str = 'relu'):
    """
    モデル全体を再帰的に初期化するユーティリティ。
    例: initialize_weights(model, is_ee=True)
    """
    fn = partial(_initialize_one, is_ee=is_ee, linear_nonlinearity=linear_nonlinearity)
    model.apply(fn)

class ModelWrapper(nn.Module):
    def __init__(self, backbone, in_features, num_classes, T=1, type="None"):
        super(ModelWrapper, self).__init__()
        self.backbone = backbone
        self.type = type
        self.T = T
        if self.type == "BN":
            self.normalize = nn.BatchNorm1d(in_features, affine = False)
            initialize_weights(self.normalize)
        elif self.type == "LN":
            self.normalize = nn.GroupNorm(1, num_classes, affine=True)
            initialize_weights(self.normalize)
        self.fc = nn.Linear(in_features, num_classes)
        initialize_weights(self.fc)
        
    def forward(self, x):
        x = self.get_features(x)
        if self.type == "BN":
            x = self.normalize(x)
        x = self.fc(x)
        if self.type == "LN":
            x = self.normalize(x)
        x /= self.T
        return x

    def get_features(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_features_and_logits(self, x):
        x = self.get_features(x)
        features = x
        if self.type == "BN":
            x = self.normalize(x)
        x = self.fc(x)
        if self.type == "LN":
            x = self.normalize(x)
        x /= self.T
        return features, x