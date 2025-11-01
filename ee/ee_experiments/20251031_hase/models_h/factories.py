from typing import Any, Optional
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

from .architectures.resnet import ResNet
from .architectures.blocks import BasicBlock, Bottleneck
from .weights.configurations import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights,
    ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights, ResNeXt101_64X4D_Weights,
    Wide_ResNet50_2_Weights, Wide_ResNet101_2_Weights
)


def _resnet(
        block,
        layers,
        weights,
        progress: bool,
        **kwargs: Any,
        ) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__."""
    weights = ResNet18_Weights.verify(weights)
    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet34_Weights.IMAGENET1K_V1))
def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__."""
    weights = ResNet34_Weights.verify(weights)
    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__."""
    weights = ResNet50_Weights.verify(weights)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet101_Weights.IMAGENET1K_V1))
def resnet101(*, weights: Optional[ResNet101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__."""
    weights = ResNet101_Weights.verify(weights)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet152_Weights.IMAGENET1K_V1))
def resnet152(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__."""
    weights = ResNet152_Weights.verify(weights)
    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNeXt50_32X4D_Weights.IMAGENET1K_V1))
def resnext50_32x4d(*, weights: Optional[ResNeXt50_32X4D_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNeXt-50 32x4d model from `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_."""
    weights = ResNeXt50_32X4D_Weights.verify(weights)
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNeXt101_32X8D_Weights.IMAGENET1K_V1))
def resnext101_32x8d(*, weights: Optional[ResNeXt101_32X8D_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNeXt-101 32x8d model from `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_."""
    weights = ResNeXt101_32X8D_Weights.verify(weights)
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNeXt101_64X4D_Weights.IMAGENET1K_V1))
def resnext101_64x4d(*, weights: Optional[ResNeXt101_64X4D_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNeXt-101 64x4d model from `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_."""
    weights = ResNeXt101_64X4D_Weights.verify(weights)
    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", Wide_ResNet50_2_Weights.IMAGENET1K_V1))
def wide_resnet50_2(*, weights: Optional[Wide_ResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """Wide ResNet-50-2 model from `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_."""
    weights = Wide_ResNet50_2_Weights.verify(weights)
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", Wide_ResNet101_2_Weights.IMAGENET1K_V1))
def wide_resnet101_2(*, weights: Optional[Wide_ResNet101_2_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """Wide ResNet-101-2 model from `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_."""
    weights = Wide_ResNet101_2_Weights.verify(weights)
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)