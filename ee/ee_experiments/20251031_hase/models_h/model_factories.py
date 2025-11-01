import copy
import math
import torch
import torch.nn as nn
from torchvision import models

from .wrappers.base import ModelWrapper, initialize_weights
from .wrappers.temperature import RelaxedSoftmax, CurriculumTemperature
from .wrappers.ensemble import EnsembleWrapper, HeadEnsembleWrapper
from .easy_ensemble import EasyEnsembleConverterV2


def create_model_ensembles(name="resnet50", num_classes=200, T=1, pretrained=False, div=1, ensembles=1, for_cifar_customize=False, is_ee=False, is_he=False, **kwargs):
    weights = 'IMAGENET1K_V1' if pretrained else None
    wrapped_models = []

    print("check:   --- > ", name, num_classes, T, pretrained, div, ensembles, for_cifar_customize, is_ee)
    
    if is_he:
        in_features = -1
        if name == "resnet50":
            backbone = models.resnet50(weights=weights)
            in_features = 2048
        elif name == "resnet18":
            backbone = models.resnet18(weights=weights)
            in_features = 512
        elif name == "resnet34":
            backbone = models.resnet34(weights=weights)
            in_features = 512
        elif name == "resnet101":
            backbone = models.resnet101(weights=weights)
            in_features = 2048
        elif name == "resnet152":
            backbone = models.resnet152(weights=weights)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported model name: {name}")
        backbone.fc = nn.Identity()
        if for_cifar_customize:
            backbone = cifar_resnet_customize(backbone) 

        return HeadEnsembleWrapper(backbone, ensembles, num_classes, in_features)
    
    if is_ee:
        if name == "resnet50":
            backbone = models.resnet50(weights=weights)
        elif name == "resnet18":
            backbone = models.resnet18(weights=weights)
        backbone.fc = torch.nn.Identity()
        if for_cifar_customize:
            backbone = cifar_resnet_customize(backbone) 
        # Easy Ensembleに変換
        converter = EasyEnsembleConverterV2(num_ensembles=ensembles, num_classes=num_classes, use_gn=False, scale=False, cross=False)
        ee_model = converter.convert_model(backbone, input_channels=3)
        # 現状初期化して利用
        initialize_weights(ee_model, is_ee=is_ee)
        return ee_model  # Return EasyEnsembleConverterV2 instance for EasyEnsembleTrainer
    else:
        if name == "resnet50":
            for i in range(ensembles):
                backbone = models.resnet50(weights=weights)
                backbone.fc = nn.Identity()
                in_features = 2048
                
                if div > 1:
                    cnn_trans(backbone, div=div)
                    in_features = in_features // div
                
                model = ModelWrapper(backbone, in_features, num_classes, T=T, **kwargs)
                if not pretrained:
                    initialize_weights(model)
                wrapped_models.append(model)
                
        elif name == "resnet18":
            for i in range(ensembles):
                backbone = models.resnet18(weights=weights)
                backbone.fc = nn.Identity()
                in_features = 512
                
                if div > 1:
                    cnn_trans(backbone, div=div)
                    in_features = in_features // div
                
                model = ModelWrapper(backbone, in_features, num_classes, T=T, **kwargs)
                if not pretrained:
                    initialize_weights(model)
                wrapped_models.append(model)
        elif name == "resnet34":
            for i in range(ensembles):
                backbone = models.resnet34(weights=weights)
                backbone.fc = nn.Identity()
                in_features = 512
                
                if div > 1:
                    cnn_trans(backbone, div=div)
                    in_features = in_features // div
                
                model = ModelWrapper(backbone, in_features, num_classes, T=T, **kwargs)
                if not pretrained:
                    initialize_weights(model)
                wrapped_models.append(model)
                
        elif name == "resnet101":
            for i in range(ensembles):
                backbone = models.resnet101(weights=weights)
                backbone.fc = nn.Identity()
                in_features = 2048
                
                if div > 1:
                    cnn_trans(backbone, div=div)
                    in_features = in_features // div
                
                model = ModelWrapper(backbone, in_features, num_classes, T=T, **kwargs)
                if not pretrained:
                    initialize_weights(model)
                wrapped_models.append(model)
                
        elif name == "resnet152":
            for i in range(ensembles):
                backbone = models.resnet152(weights=weights)
                backbone.fc = nn.Identity()
                in_features = 2048
                
                if div > 1:
                    cnn_trans(backbone, div=div)
                    in_features = in_features // div
                
                model = ModelWrapper(backbone, in_features, num_classes, T=T, **kwargs)
                if not pretrained:
                    initialize_weights(model)
                wrapped_models.append(model)
        else:
            raise ValueError(f"Unsupported model name: {name}")

        if for_cifar_customize:
            wrapped_models = [cifar_resnet_customize(model) for model in wrapped_models]
            
        model = EnsembleWrapper(wrapped_models)
        return model  # Return ModelWrapper instances for EnsembleTrainer


def create_model(name="resnet50", num_classes = 200, pretrained=False, for_cifar_customize=False, div=-1, **kwargs):
    weights = 'IMAGENET1K_V1' if pretrained else None

    if name == "resnet18":
        model = models.resnet18(weights = weights)
        backbone = model
        backbone.fc = nn.Identity()
        in_features = 512
    elif name == "resnet34":
        model = models.resnet34(weights = weights)
        backbone = model
        backbone.fc = nn.Identity()
        in_features = 512
    elif name == "resnet50":
        model = models.resnet50(weights = weights)
        backbone = model
        backbone.fc = nn.Identity()
        in_features = 2048
    elif name == "resnet101":
        model = models.resnet101(weights = weights)
        backbone = model
        backbone.fc = nn.Identity()
        in_features = 2048
    elif name == "resnet152":
        model = models.resnet152(weights = weights)
        backbone = model
        backbone.fc = nn.Identity()
        in_features = 2048
    else:
        raise ValueError(f"Unsupported model name: {name}")

    if for_cifar_customize:
        backbone = cifar_resnet_customize(backbone)

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


def cifar_resnet_customize(model):
    """Customize ResNet for CIFAR datasets"""
    if isinstance(model, ModelWrapper):
        # Replace the first conv layer to handle 32x32 input
        model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the max pooling layer
        model.backbone.maxpool = nn.Identity()
    else:
        # Replace the first conv layer to handle 32x32 input
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the max pooling layer
        model.maxpool = nn.Identity()

    return model


def cnn_trans(model, div=2):
    """Transform CNN by reducing filter numbers"""
    def halve_conv_filters(conv_layer, div):
        if isinstance(conv_layer, nn.Conv2d):
            old_out = conv_layer.out_channels
            old_in = conv_layer.in_channels
            new_out = max(1, old_out // div)
            new_in = max(1, old_in // div) if old_in != 3 else old_in  # Don't change RGB input
            
            new_conv = nn.Conv2d(new_in, new_out,
                                 kernel_size=conv_layer.kernel_size,
                                 stride=conv_layer.stride,
                                 padding=conv_layer.padding,
                                 dilation=conv_layer.dilation,
                                 groups=conv_layer.groups,
                                 bias=(conv_layer.bias is not None),
                                 padding_mode=conv_layer.padding_mode)
            
            with torch.no_grad():
                min_out = min(old_out, new_out)
                min_in = min(old_in, new_in)
                new_conv.weight.data[:min_out, :min_in, :, :] = \
                    conv_layer.weight.data[:min_out, :min_in, :, :].clone()
                if conv_layer.bias is not None:
                    new_conv.bias.data[:min_out] = conv_layer.bias.data[:min_out].clone()
            return new_conv
        return conv_layer
    
    def replace_batchnorm(bn_layer, div):
        if isinstance(bn_layer, nn.BatchNorm2d):
            old_features = bn_layer.num_features
            new_features = max(1, old_features // div)
            new_bn = nn.BatchNorm2d(new_features)
            with torch.no_grad():
                min_features = min(old_features, new_features)
                new_bn.weight.data[:min_features] = bn_layer.weight.data[:min_features].clone()
                new_bn.bias.data[:min_features] = bn_layer.bias.data[:min_features].clone()
                if hasattr(bn_layer, 'running_mean'):
                    new_bn.running_mean.data[:min_features] = bn_layer.running_mean.data[:min_features].clone()
                if hasattr(bn_layer, 'running_var'):
                    new_bn.running_var.data[:min_features] = bn_layer.running_var.data[:min_features].clone()
            new_bn.reset_running_stats()
            return new_bn
        return bn_layer
    
    # Apply transformations to all modules
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]] if '.' in name else model
            setattr(parent_module, name.split('.')[-1], halve_conv_filters(module, div))
        elif isinstance(module, nn.BatchNorm2d):
            parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]] if '.' in name else model
            setattr(parent_module, name.split('.')[-1], replace_batchnorm(module, div))

    # Adjust FC layer
    if type(model.fc) == nn.Linear:
        model.fc = nn.Linear(model.fc.in_features // div, model.fc.out_features)


def cnn_trans2(model, div=32):
    """More sophisticated CNN transformation with scaling"""
    def scale_conv_filters(conv_layer, div):
        if isinstance(conv_layer, nn.Conv2d):
            old_out = conv_layer.out_channels
            old_in  = conv_layer.in_channels
            new_out = max(1, int(round(old_out / div)))
            
            if div >= 1 and (old_in % div != 0):
                new_in = old_in
            elif (old_in == 3):
                new_in = old_in
            else:
                new_in = max(1, int(round(old_in / div)))
            
            new_conv = nn.Conv2d(new_in, new_out,
                                 kernel_size=conv_layer.kernel_size,
                                 stride=conv_layer.stride,
                                 padding=conv_layer.padding,
                                 dilation=conv_layer.dilation,
                                 groups=conv_layer.groups,
                                 bias=(conv_layer.bias is not None),
                                 padding_mode=conv_layer.padding_mode)
            
            with torch.no_grad():
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
    
    target_model = copy.deepcopy(model)
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
    
    # Adjust FC layer
    if hasattr(target_model, 'fc') and isinstance(target_model.fc, nn.Linear):
        old_in = target_model.fc.in_features
        if div >= 1 and (old_in % div != 0):
            new_in = old_in
        else:
            new_in = max(1, int(round(old_in / div)))
        
        old_out = target_model.fc.out_features
        new_fc = nn.Linear(new_in, old_out)
        
        with torch.no_grad():
            min_in = min(old_in, new_in)
            new_fc.weight.data[:, :min_in] = target_model.fc.weight.data[:, :min_in].clone()
            new_fc.bias.data[:] = target_model.fc.bias.data[:].clone()
        
        target_model.fc = new_fc
    
    return target_model