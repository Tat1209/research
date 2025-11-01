import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .base import initialize_weights


# Relaxed Softmax: Efficient Confidence Auto-Calibration for Safe Pedestrian Detection
# https://openreview.net/forum?id=S1lG7aTnqQ
class RelaxedSoftmax(nn.Module):
    def __init__(self, backbone, in_features, num_classes):
        super(RelaxedSoftmax, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(in_features, num_classes)
        self.fca = nn.Linear(in_features, 1)
        initialize_weights(self.fc)
        initialize_weights(self.fca)
        
    def forward(self, x):
        x = self.get_features(x)
        features = self.fc(x)
        alpha = F.softplus(self.fca(x)) + 1e-6
        return features * alpha

    def get_features(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_features_and_logits(self, x):
        x = self.get_features(x)
        features = self.fc(x)
        alpha = F.softplus(self.fca(x)) + 1e-6
        return features, features * alpha


# Curriculum Temperature for Knowledge Distillation（CTKD）
# https://dl.acm.org/doi/abs/10.1609/aaai.v37i2.25236
class CurriculumTemperature(nn.Module):
    def __init__(self, backbone, in_features, num_classes, use_ctkd=True):
        super(CurriculumTemperature, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(in_features, num_classes)
        initialize_weights(self.fc)
        self.use_ctkd = use_ctkd
        if use_ctkd:
            self.temp_module = GlobalTemperatureModule(init_value=1.0)
        
    def forward(self, x, lambda_val=1.0):
        f = self.get_features(x)
        logits = self.fc(f)
        if self.use_ctkd:
            batch_size = x.size(0)
            # 温度モジュールから温度値を取得
            temperature = self.temp_module(batch_size)  # shape: [batch_size, 1]
            # GRLを適用して温度パラメータの逆方向勾配更新を実現
            # temperature_adv = grad_reverse(temperature, lambda_val)
            temperature_adv = temperature
            # ロジットを温度で割ってスケールする（Softmax前のスケーリング）
            scaled_logits = logits / temperature_adv
            return scaled_logits
        else:
            return logits

    def get_features(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x


class GlobalTemperatureModule(nn.Module):
    def __init__(self, init_value=1.0):
        super(GlobalTemperatureModule, self).__init__()
        # 学習可能な1つのスカラーとして定義
        self.temp = nn.Parameter(torch.tensor(init_value))
        
    def forward(self, batch_size):
        # 全サンプルに対して同じ温度値を適用するようバッチサイズ分に展開
        temperature = self.temp.expand(batch_size, 1)
        return temperature


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        # 順伝播では入力をそのまま返す
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # 逆伝播時、勾配に逆符号（lambda_倍）を掛ける
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)