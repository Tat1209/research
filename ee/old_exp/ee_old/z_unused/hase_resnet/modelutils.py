from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

def initialize_weights(module):
    ''' モジュールの重みを初期化する
    :param module:
    :return:
    '''
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
            if m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Single2Single(nn.Module):
    """ backbone to classifierのシンプルな接続モデル
    """
    def __init__(self, backbone, classifier, dropout=False, beforeT=1, afterT=1, repeat_input=1):
        super(Single2Single, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = None
        self.beforeT = beforeT
        self.afterT = afterT
        self.repeat_input = repeat_input
        if dropout:
            self.dropout = nn.Dropout(p=0.8)
    def forward(self, x):
        x = x.repeat(1, self.repeat_input, 1)
        x = self.backbone(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x /= self.beforeT
        x = self.classifier(x)
        x /= self.afterT
        return x

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        n, c, l = x.shape
        x = x.view(n, -1, self.groups, l)
        x = x.transpose_(1, 2).contiguous()
        x = x.view(n, c, l)
        return x

class ChannelShuffleHalf(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffleHalf, self).__init__()
        self.groups = groups

    def forward(self, x):
        n, c, l = x.shape
        x = x.view(n, self.groups, 2, -1, l)
        x1 = x[:,:,:1,:,:]
        idx = np.roll(np.arange(self.groups), 1)
        x2 = x[:,idx,1:,:,:]
        x = torch.cat([x1, x2], dim=2)
        #x = x.transpose_(1, 2).contiguous()
        x = x.view(n, c, l)
        return x

class Multi2Conc(nn.Module):
    """ マルチbackboneを結合し一つのClassifierで扱うモデル
    """
    def __init__(self, bs, classifier, beforeT=1, afterT=1):
        super(Multi2Conc, self).__init__()
        self.ensembles = len(bs)
        self.bs = nn.ModuleList(bs)
        self.classifier = classifier
        self.beforeT = beforeT
        self.afterT = afterT
    def forward(self, x):
        features = []
        for b in self.bs:
            out = b(x)
            features.append(out)
        output = torch.cat(features, dim=1)
        output /= self.beforeT
        output = self.classifier(output)
        output /= self.afterT
        return output

class Multi2Add(nn.Module):
    """ マルチbackboneからマルチclassifierの出力を加算して一つの出力を算出するモデル
    """
    def __init__(self, bs, cs, beforeT=1, afterT=1):
        super(Multi2Add, self).__init__()
        self.ensembles = len(bs)
        self.bs = nn.ModuleList(bs)
        self.cs = nn.ModuleList(cs)
        self.beforeT = beforeT
        self.afterT = afterT
    def forward(self, x):
        features = []
        for b, c in zip(self.bs, self.cs):
            out = b(x)
            out /= self.beforeT
            out = c(out)
            out /= self.afterT
            features.append(out)
        output = sum(features)
        return output
