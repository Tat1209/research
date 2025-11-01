import copy
import math
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Function
import torch.nn.functional as F
from tqdm import tqdm

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

def create_model_ensembles(name="resnet50", num_classes=200, T=1, pretrained=False, div=1, ensembles=1, needs_fcs=False, **kwargs):
    weights = 'IMAGENET1K_V1' if pretrained else None
    if name == "resnet50":
        backbones = []
        for i in range(ensembles):
            m = models.resnet50(weights=weights)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            cnn_trans(m, div=div)
            initialize_weights(m)
            backbones.append(m)
    if name == "resnet18":
        backbones = []
        for i in range(ensembles):
            m = models.resnet18(weights=weights)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            cnn_trans(m, div=div)
            initialize_weights(m)
            backbones.append(m)

    return Ensembler(backbones, T=T, need_fcs=needs_fcs)

def create_model(name="resnet50", num_classes = 200, pretrained=False, for_cifar_customize=False, div=-1, **kwargs):
    weights = 'IMAGENET1K_V1' if pretrained else None

    if name == "resnet18":
        model = models.resnet18(weights = weights)
        if for_cifar_customize:
            # 最初の畳み込み層を変更 (カーネルサイズを3x3、ストライドを1に設定)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()  # 最初のプーリング層をスキップまたは削除
        backbone = nn.Sequential(*(list(model.children())[:-1]))
        in_features = model.fc.in_features
    if name == "resnet50":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=pretrained)
        backbone = nn.Sequential(*(list(model.children())[:-1]))
        in_features = model.fc.in_features
    elif name == "efficientnetb0":
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=pretrained)
        backbone = nn.Sequential(*(list(model.children())[:-1] + list(model.classifier.children())[:-2]))
        in_features = model.classifier.fc.in_features
    elif name == "efficientnetb5":
        model = models.efficientnet_b5(weights=weights)
        backbone = nn.Sequential(*(list(model.children())[:-1]))
        in_features = model.classifier[-1].in_features
    elif name == "vitb16" or name == "vitb16_woLN":
        model = models.vit_b_16(weights=weights)
        #backbone = nn.Sequential(*(list(model.children())[:-1]))
        in_features = model.heads.head.in_features
        model.heads = nn.Sequential()
        if name == "vitb16_woLN":
            model.encoder.ln = nn.Sequential()
        backbone = model
    elif name == "swint" or name == "swint_woLN":
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Sequential()
        if name == "swint_woLN":
            model.norm = nn.Sequential()
        backbone = model
    elif name == "convnextsmall" or name == "convnextsmall_woLN":
        model = models.convnext_small(weights=weights)
        #backbone = nn.Sequential(*(list(model.children())[:-1]))
        if name == "convnextsmall":
            backbone = nn.Sequential(*(list(model.children())[:-1]+list(model.classifier[:-1])))
        elif name == "convnextsmall_woLN":
            backbone = nn.Sequential(*(list(model.children())[:-1]+[model.classifier[1]]))
        in_features = model.classifier[2].in_features
    elif name == "regnety8gf":
        model = models.regnet_y_8gf(weights=weights)
        backbone = nn.Sequential(*(list(model.children())[:-1]))
        in_features = model.fc.in_features
    elif name == "regnety400mf":
        model = models.regnet_y_400mf(weights=None)
        backbone = nn.Sequential(*(list(model.children())[:-1]))
        in_features = model.fc.in_features
    elif name == "regnety800mf":
        model = models.regnet_y_800mf(weights=None)
        backbone = nn.Sequential(*(list(model.children())[:-1]))
        in_features = model.fc.in_features
    elif name == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        backbone = nn.Sequential(*((list(model.children())[:-1])+list(model.classifier[:-1])))
        in_features = model.classifier[1].in_features
    elif name == "shufflenetv2_x10":
        model = models.shufflenet_v2_x1_0(weights=None)
        backbone = nn.Sequential(*(list(model.children())[:-1]))
        in_features = model.fc.in_features

    type_ = kwargs.get('type', 'None')
    if type_ == "Relaxed":
        model = RelaxedSoftmax(backbone, in_features, num_classes)
    elif type_ == "Curriculum":
        model = CurriculumTemperature(backbone, in_features, num_classes, use_ctkd=True)
    else:
        model = ModelWrapper(backbone, in_features, num_classes, **kwargs)

    if div > 0:  # backboneを細くする処理
        model = cnn_trans2(model, div=div)

    if not pretrained:
        initialize_weights(model)
    
    return model

def initialize_weights(module):
    ''' initialize weights
    :param module:
    :return:
    '''
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.BatchNorm2d)):
            if m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            #nn.init.normal_(m.weight, 0, 0.01)
            init_range = 1.0 / math.sqrt(m.weight.shape[1])
            nn.init.uniform_(m.weight, a=-init_range, b=init_range)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def one_cycle(model, loader, device, optimizer, criterion, train=True, addinfo=False):
    if train:
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    preds, ans, all_features = [], [], []
    for inputs, labels in tqdm(loader):
    #for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()
        
        if addinfo:
            features, outputs = model.get_features_and_logits(inputs)
            if hasattr(model, 'normalize'):
                features = model.normalize(features)
            all_features.append(features.cpu().detach())
        else:
            outputs = model(inputs)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        preds += predicted.tolist()
        ans += labels.tolist()
    if addinfo:
        all_features = torch.cat(all_features, dim=0)
        #Ez, Ez2, Ezz = calcEzs(all_features)
        stats = feature_statistics(all_features, weight=model.fc.weight)
        return running_loss/len(loader.dataset), preds, ans, stats

    return running_loss/len(loader.dataset), preds, ans

def calcEzs(fx):
    N, M = fx.shape
    Ez = fx.mean(dim=0)  # [M]
    Ez_sq = (fx**2).mean(dim=0)  # [M]
    a = fx.t().matmul(fx) / N
    # 上三角成分を取り出す
    a = a - torch.diag(torch.diag(a))
    a = a.triu(1)  # 上三角行列の上三角成分を取り出す
    a = a[a != 0]  # ゼロ以外の要素を取り出す
    
    # それぞれの平均と標準偏差を返す．
    return (Ez.abs().mean(), Ez.mean(), Ez.std()), (Ez_sq.abs().mean(), Ez_sq.mean(), Ez_sq.std()), (a.abs().mean(), a.mean(), a.std())

from typing import Dict, Tuple, Optional
def feature_statistics(
        fx: torch.Tensor,
        weight: Optional[torch.Tensor] = None
) -> Dict[str, Tuple[float, float, float]]:
    """
    Collect statistics of pooled features that hold irrespective of
    BatchNorm insertion.  In particular we use the *raw* second-order
    moment  E[z_k z_l]  (k≠l) instead of the centred covariance so that
    BN あり／なし を直接比較できる。

    Parameters
    ----------
    fx : torch.Tensor               # shape [N, M]
        Mini-batch or full-set feature matrix (GAP 出力または BN 後特徴)
    weight : torch.Tensor | None    # shape [C, M]
        Fully-connected classifier weights; if passed, R_j statistics are
        computed.

    Returns
    -------
    stats : dict
        'Ez'   : ( mean|E[z_k]| ,  mean E[z_k] , std E[z_k] )
        'Ez2'  : ( mean|E[z_k^2]-1| , mean (E[z_k^2]-1) , std (E[z_k^2]-1) )
        'Ezz'  : ( mean|E[z_k z_l]| , mean E[z_k z_l] , std E[z_k z_l] )
        'R'    : ( mean|R_j| , mean R_j , std R_j )      # if weight is given

    Notes
    -----
    *   The diagonal term in R_j uses  E[z_k^2]  to remain valid even when
        BN の γ,β が動き  Var[z_k]≠1 となる場合。
    """
    N, M = fx.shape
    # ---------------- 1st & 2nd moments ----------------
    Ez  = fx.mean(dim=0)                        # E[z_k]
    Ez2 = (fx ** 2).mean(dim=0)                 # E[z_k^2]

    stats = {
        'Ez':  (Ez.abs().mean().item(),
                Ez.mean().item(),
                Ez.std().item()),

        'Ez2': ((Ez2 - 1).abs().mean().item(),
                (Ez2 - 1).mean().item(),
                (Ez2 - 1).std().item())
    }

    # ---------------- E[z_k z_l] (k≠l) ------------------
    # second-order raw moment matrix
    P = fx.T @ fx / N                           # [M, M], unbiased=False
    P.fill_diagonal_(0.0)                       # remove diagonal
    upper = P.triu(1)
    ezz_vals = upper[upper != 0]

    if ezz_vals.numel():
        stats['Ezz'] = (ezz_vals.abs().mean().item(),
                        ezz_vals.mean().item(),
                        ezz_vals.std().item())
    else:
        stats['Ezz'] = (0.0, 0.0, 0.0)          # M=1 保険

    # ---------------- R_j (optional) -------------------
    if weight is not None:
        W = weight.detach().cpu()               # [C, M]

        # diagonal part: Σ w_{jk}^2 E[z_k^2]
        diag = (W ** 2 @ Ez2)                   # [C]

        # off-diag part: Σ_{k≠l} w_k w_l E[zk zl]
        off_full = torch.einsum('cm,mn,cn->c', W, P, W)
        numer = off_full                        # P はすでに対角 0

        R = numer / diag.clamp(min=1e-12)
        stats['R'] = (R.abs().mean().item(),
                      R.mean().item(),
                      R.std().item())

    return stats

# 既存のモデルのフィルタ数を削減する関数（ResNet50で動作確認済み）
def cnn_trans(model, div=32):
    def halve_conv_filters(conv_layer, div):
        """Conv2d層のフィルタ数を半分にする関数"""
        if isinstance(conv_layer, nn.Conv2d):
            new_out_channels = conv_layer.out_channels // div
            if conv_layer.in_channels % div != 0 or conv_layer.in_channels == 3:
                new_in_channels = conv_layer.in_channels
            else:
                new_in_channels = conv_layer.in_channels // div
            new_conv = nn.Conv2d(new_in_channels, new_out_channels, 
                                kernel_size=conv_layer.kernel_size, 
                                stride=conv_layer.stride, 
                                padding=conv_layer.padding,
                                bias=conv_layer.bias is not None)
            # 重みのコピー
            new_conv.weight.data = conv_layer.weight[:new_out_channels, :new_in_channels, :, :].data.clone()
            if conv_layer.bias is not None:
                new_conv.bias.data = conv_layer.bias[:new_out_channels].data.clone()
            return new_conv
        return conv_layer
    
    def replace_batchnorm(bn_layer, div):
        """BatchNorm2d層を半分のチャネル数で新しく置き換える関数"""
        if isinstance(bn_layer, nn.BatchNorm2d):
            new_num_features = bn_layer.num_features // div
            new_bn = nn.BatchNorm2d(new_num_features)
            
            # 元のパラメータがある場合は新しい層にコピー
            new_bn.weight.data = bn_layer.weight[:new_num_features].data.clone()
            new_bn.bias.data = bn_layer.bias[:new_num_features].data.clone()
            new_bn.reset_running_stats()
            
            # 新しいレイヤーに置き換え
            return new_bn
        return bn_layer
    
    # モデル内の各層を探索し、フィルタ数を半分に変更
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Conv2d層の置き換え
            parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]] if '.' in name else model
            setattr(parent_module, name.split('.')[-1], halve_conv_filters(module, div))
        elif isinstance(module, nn.BatchNorm2d):
            # モデル内でモジュールを直接置き換え
            parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]] if '.' in name else model
            setattr(parent_module, name.split('.')[-1], replace_batchnorm(module, div))


    # FC層の出力特徴数も半分に調整
    if type(model.fc) == nn.Linear:
        model.fc = nn.Linear(model.fc.in_features // div, model.fc.out_features)


def cnn_trans2(model, div=32):
    def scale_conv_filters(conv_layer, div):
        if isinstance(conv_layer, nn.Conv2d):
            old_out = conv_layer.out_channels
            old_in  = conv_layer.in_channels
            # 出力チャネルは必ずスケール
            new_out = max(1, int(round(old_out / div)))
            # 入力チャネルは、div>=1（つまり縮小の場合）で割り切れないなら変更せず、
            # それ以外（拡大の場合など）はスケールする
            if div >= 1 and (old_in % div != 0):
                new_in = old_in
            elif (old_in == 3):
                new_in = old_in
            else:
                new_in = max(1, int(round(old_in / div)))
            
            # 新しい Conv2d 層を作成（kernel_size, stride, padding などはそのまま）
            new_conv = nn.Conv2d(new_in, new_out,
                                 kernel_size=conv_layer.kernel_size,
                                 stride=conv_layer.stride,
                                 padding=conv_layer.padding,
                                 dilation=conv_layer.dilation,
                                 groups=conv_layer.groups,
                                 bias=(conv_layer.bias is not None),
                                 padding_mode=conv_layer.padding_mode)
            
            # 重みのコピー（元の重みと新しい重みの形状が重なる部分だけコピー）
            with torch.no_grad():
                # コピーするチャネル数は「新旧それぞれのうち小さい方」
                min_out = min(old_out, new_out)
                min_in  = min(old_in, new_in)
                new_conv.weight.data[:min_out, :min_in, :, :] = \
                    conv_layer.weight.data[:min_out, :min_in, :, :].clone()
                if conv_layer.bias is not None:
                    min_bias = min(old_out, new_out)
                    new_conv.bias.data[:min_bias] = conv_layer.bias.data[:min_bias].clone()
            return new_conv
        return conv_layer

    def scale_batchnorm(bn_layer, div):
        if isinstance(bn_layer, nn.BatchNorm2d):
            old_features = bn_layer.num_features
            new_features = max(1, int(round(old_features / div)))
            new_bn = nn.BatchNorm2d(new_features)
            with torch.no_grad():
                min_features = min(old_features, new_features)
                new_bn.weight.data[:min_features] = bn_layer.weight.data[:min_features].clone()
                new_bn.bias.data[:min_features]   = bn_layer.bias.data[:min_features].clone()
                if hasattr(bn_layer, 'running_mean'):
                    new_bn.running_mean.data[:min_features] = bn_layer.running_mean.data[:min_features].clone()
                if hasattr(bn_layer, 'running_var'):
                    new_bn.running_var.data[:min_features] = bn_layer.running_var.data[:min_features].clone()
            return new_bn
        return bn_layer
    
    target_model = copy.deepcopy(model)  # モデルのコピーを作成

    # モデル内の各層の名前とモジュールの辞書を作成
    modules_dict = dict(target_model.named_modules())
    for name, module in target_model.named_modules():
        if isinstance(module, nn.Conv2d):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            parent_module = modules_dict[parent_name] if parent_name else target_model
            setattr(parent_module, name.split('.')[-1], scale_conv_filters(module, div))
        elif isinstance(module, nn.BatchNorm2d):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            parent_module = modules_dict[parent_name] if parent_name else target_model
            setattr(parent_module, name.split('.')[-1], scale_batchnorm(module, div))
    
    # FC層 (全結合層) の調整（存在する場合）
    if hasattr(target_model, 'fc') and isinstance(target_model.fc, nn.Linear):
        old_in = target_model.fc.in_features
        # 入力チャネルについては、div>=1 で割り切れなければ元の値を維持
        if div >= 1 and (old_in % div != 0):
            new_in = old_in
        else:
            new_in = max(1, int(round(old_in / div)))
        new_fc = nn.Linear(new_in, target_model.fc.out_features, bias=(target_model.fc.bias is not None))
        with torch.no_grad():
            min_in = min(old_in, new_in)
            new_fc.weight.data[:, :min_in] = target_model.fc.weight.data[:, :min_in].clone()
            if target_model.fc.bias is not None:
                new_fc.bias.data = target_model.fc.bias.data.clone()
        target_model.fc = new_fc

    return target_model

# パラメータ数をカウントする関数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)