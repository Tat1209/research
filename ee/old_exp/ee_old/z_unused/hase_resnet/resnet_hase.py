import torch
import torch.nn as nn
from . import modelutils

def conv3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3(inplanes, planes, stride, groups=groups)
        self.norm1 = nn.GroupNorm(groups, planes, affine=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3(planes, planes, groups=groups)
        self.norm2 = nn.GroupNorm(groups, planes, affine=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, nb_fils = 32, num_classes=6, groups=1, repeat_input=1):
        super(ResNet, self).__init__()

        self.inplanes = nb_fils
        self.groups = groups
        self.repeat_input = repeat_input
        self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, groups=self.groups)
        self.norm1 = nn.GroupNorm(groups, self.inplanes, affine=False)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nb_fils, layers[0], groups=self.groups)
        self.layer2 = self._make_layer(block, nb_fils*2, layers[1], stride=2, groups=self.groups)
        self.layer3 = self._make_layer(block, nb_fils*4, layers[2], stride=2, groups=self.groups)
        self.layer4 = self._make_layer(block, nb_fils*8, layers[3], stride=2, groups=self.groups)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nb_fils * 8 * block.expansion, num_classes)

        modelutils.initialize_weights(self)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * block.expansion, stride, groups=groups),
                nn.GroupNorm(groups, planes * block.expansion, affine=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.repeat(1, self.repeat_input, 1)
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x /= self.groups

        x = self.fc(x)

        return x


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)

class MMResNet(nn.Module):
    def __init__(self, ensembles, afterT=1, **kwargs):
        super(MMResNet, self).__init__()
        self.ensembles = ensembles
        self.ms = nn.ModuleList()
        for i in range(ensembles):
            self.ms.append(_resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs))
        self.afterT = afterT

    def forward(self, x):
        features = []
        for m in self.ms:
            out = m(x)
            out /= self.afterT
            features.append(out)
        output = sum(features)
        return output
