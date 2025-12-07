import math
import sys
from pathlib import Path
from typing import Any, List, Literal, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter, UninitializedParameter

this_path = Path(__file__) if '__file__' in globals() else Path("<unknown>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

from network import Refiner

class EERefiner(Refiner):
    def __init__(self, model: nn.Module, _steps: List = None):
        super().__init__(model, _steps)
        
    def multi_narrow(self, div: int | None = 1, agg: Literal["mean", "sum", "none", "both"] = "both", arch: Literal["auto", "resnet", "mobilenet", "efficientnet", "convnext", "regnet"] = "auto", flex_ch=False, flex_mode: Literal["round", "floor", "cail"] = "round", grouped_linear_impl: Literal["einsum", "conv"] = "einsum") -> nn.Module:
        """
        分割数(div)からアンサンブル数(div^2)と幅(1/div)を自動決定し、パラメータ数を維持して分割

        Args:
            div (int): 分割係数。Ens数=div^2, 幅=1/div となる (例: div=2 -> 4モデル, 幅0.5)
            agg (Literal): 集約方法 ("mean", "sum", "none", "both")
                "mean": 各パスの出力を平均化して返す
                "sum": 各パスの出力を合計して返す
                "none": 各パスの出力をリストで返す
                "both": 平均化した出力と各パスの出力リストをタプルで返す
            arch (Literal): ネットワークアーキテクチャの種類 ("auto", "resnet", "mobilenet", "efficientnet", "convnext", "regnet")
            flex_ch (bool): チャンネルが割り切れない場合に調整を行うか
            flex_mode (Literal): flex_ch有効時の端数処理 ("round", "floor", "ceil")
            grouped_linear_impl (Literal): グループ化Linear層の実装方法 ("einsum", "conv")

        Returns:
            nn.Module: アンサンブル化されたモデル

        Examples:
            >>> # 基本: 幅1/2, 4モデル, 重み初期化あり
            >>> EERefiner(model.resnet18(num_classes=100)).multi_narrow(div=2, agg="mean").init_weights().build()
            >>> # 柔軟なチャンネル調整: 幅1/3で割り切れないch数を調整
            >>> EERefiner(model.convnext(num_classes=100)).multi_narrow(div=3, flex_ch=True).init_weights().build()
        """
        ensembles = round(div ** 2)
        ch_scale = 1 / div
        return self.easy_ensemble(ensembles=ensembles, ch_scale=ch_scale, agg=agg, arch=arch, flex_ch=flex_ch, flex_mode=flex_mode, grouped_linear_impl=grouped_linear_impl)
        
    def easy_ensemble(self, ensembles: int = 1, ch_scale: int | float = 1, agg: Literal["mean", "sum", "none", "both"] = "mean", arch: Literal["auto", "resnet", "mobilenet", "efficientnet", "convnext", "regnet"] = "auto", flex_ch=False, flex_mode: Literal["round", "floor", "ceil"] = "round", grouped_linear_impl: Literal["einsum", "conv"] = "einsum") -> nn.Module:
        """
        指定した数(ensembles)と幅(ch_scale)でモデルを分割・アンサンブル化します。

        Args:
            ensembles (int): サブモデルの数。
            ch_scale (float): 各モデルの幅倍率 (0.0 < scale <= 1.0)。
            agg (Literal): 集約方法 ("mean", "sum", "none", "both")
                "mean": 各パスの出力を平均化して返す
                "sum": 各パスの出力を合計して返す
                "none": 各パスの出力をリストで返す
                "both": 平均化した出力と各パスの出力リストをタプルで返す
            flex_ch (bool): チャンネル数調整の可否。端数が出る倍率指定時に有効。
            flex_mode (Literal): flex_ch有効時の端数処理 ("round", "floor", "ceil")
            grouped_linear_impl (Literal): グループ化Linear層の実装方法 ("einsum", "conv")

        Returns:
            nn.Module: アンサンブル化されたモデル

        Examples:
            >>> # 手動設定: 3モデル, 幅0.33倍, 端数調整あり, 重み初期化
            >>> EERefiner(model.mobilenet_v2(num_classes=100)).easy_ensemble(ensembles=3, ch_scale=0.33, flex_ch=True).init_weights().build()
        """
        return self.ee_convert(ensembles, ch_scale, arch, flex_ch, flex_mode, grouped_linear_impl).ee_wrapper(ensembles, agg)
    
    def ee_convert(self, ensembles: int, ch_scale: int | float, arch, flex_ch: bool, flex_mode: Literal["round", "floor", "ceil"], grouped_linear_impl: Literal["einsum", "conv"]) -> nn.Module:
        def _ee_convert(model, ensembles=1, ch_scale=1, arch=arch, flex_ch=flex_ch, flex_mode=flex_mode, grouped_linear_impl=grouped_linear_impl):
            if ensembles == 1 and ch_scale == 1:
                return model

            first_layer_processed = False

            def align_channels(target_channels: float, divisor: int) -> int:
                c = round(target_channels)
                
                if not flex_ch:
                    return c

                if c < divisor:
                    return divisor
                if flex_mode == 'floor':
                    return (c // divisor) * divisor
                elif flex_mode == 'ceil':
                    if c % divisor == 0:
                        return c
                    return ((c // divisor) + 1) * divisor
                elif flex_mode == 'round':
                    return round(c / divisor) * divisor
                else:
                    raise ValueError(f"Invalid flex_mode: {flex_mode}")

            def policy_classifier(name, module: nn.Module):
                target_names = {'fc', 'classifier', 'head'}

                def convert_fc_linear(module):
                    raw_in = module.in_features * ensembles * ch_scale
                    raw_out = module.out_features * ensembles # ch_scaleなし

                    if grouped_linear_impl == "einsum":
                        GroupedLinearClass = GroupedLinear
                    elif grouped_linear_impl == "conv":
                        GroupedLinearClass = GroupedLinearConv1d

                    new_module = GroupedLinearClass(
                        in_features=align_channels(raw_in, divisor=ensembles),
                        out_features=align_channels(raw_out, divisor=ensembles),
                        bias=module.bias is not None,
                        groups=ensembles,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                    return new_module
                
                if name in target_names:
                    if isinstance(module, nn.Linear): # fc自身がLinear層の場合
                        return convert_fc_linear(module)

                    def policy_fc_linear(sub_name, sub_module):
                        if isinstance(sub_module, nn.Linear):
                            return convert_fc_linear(sub_module)

                    self.apply_policies(module, [policy_fc_linear, policy_conv, policy_batchnorm, policy_layernorm, policy_linear]) # fcの中も走査が必要

                    return module # 再帰適用後のブロックを返す これがないとfc内に対してもがほかのpolicyを走査してしまう．

            def policy_conv(name, module: nn.Module):
                nonlocal first_layer_processed
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    is_depthwise = module.in_channels == module.out_channels and module.groups == module.in_channels

                    if is_depthwise:
                        align_divisor = ensembles
                    else:
                        align_divisor = module.groups * ensembles

                    if first_layer_processed:
                        raw_in = module.in_channels * ensembles * ch_scale
                    else:
                        raw_in = module.in_channels * ensembles
                        first_layer_processed = True
                    
                    new_in_channels = align_channels(raw_in, divisor=align_divisor)
                    new_groups = new_in_channels if is_depthwise else align_divisor

                    raw_out = module.out_channels * ensembles * ch_scale
                    new_out_channels = align_channels(raw_out, divisor=new_groups)

                    new_module = module.__class__(
                        in_channels=new_in_channels,
                        out_channels=new_out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        groups=new_groups,
                        bias=module.bias is not None,
                        padding_mode=module.padding_mode,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )

                    return new_module

            def policy_batchnorm(name, module: nn.Module):
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    raw_features = module.num_features * ensembles * ch_scale
                    new_module = module.__class__(
                        num_features=align_channels(raw_features, divisor=ensembles),
                        eps=module.eps,
                        momentum=module.momentum,
                        affine=module.affine,
                        track_running_stats=module.track_running_stats,
                        device=module.weight.device if module.affine else None,
                        dtype=module.weight.dtype if module.affine else None,
                    )
                    return new_module

            def policy_linear(name, module: nn.Module):
                if isinstance(module, nn.Linear):
                    raw_in = module.in_features * ensembles * ch_scale
                    raw_out = module.out_features * ensembles * ch_scale
                    
                    if grouped_linear_impl == "einsum":
                        GroupedLinearClass = GroupedLinear
                    elif grouped_linear_impl == "conv":
                        GroupedLinearClass = GroupedLinearConv1d

                    new_module = GroupedLinearClass(
                        in_features=align_channels(raw_in, divisor=ensembles),
                        out_features=align_channels(raw_out, divisor=ensembles),
                        bias=module.bias is not None,
                        groups=ensembles,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                    return new_module

            def policy_layernorm(name, module: nn.Module) -> nn.Module | None:
                if isinstance(module, nn.LayerNorm):
                    if type(module) is nn.LayerNorm: # ケースA: 純粋な nn.LayerNorm ConvNeXtのBlock内など、NHWC (Channel Last) でデータが来る場所
                        TargetNormClass = GroupedLayerNorm
                    else:
                        TargetNormClass = GroupedLayerNorm2d # ケースB: LayerNorm2d (特定のカスタムクラス) Stem/Downsampleなど、NCHW (Channel First) でデータが来る場所

                    if isinstance(module.normalized_shape, (int, float)):
                        old_channels = int(module.normalized_shape)
                    else:
                        old_channels = int(module.normalized_shape[0])

                    raw_channels = old_channels * ensembles * ch_scale
                    new_channels = align_channels(raw_channels, divisor=ensembles)

                    return TargetNormClass(
                        num_groups=ensembles,
                        num_channels=new_channels,
                        eps=module.eps,
                    )

            def policy_layer_scale_param(name, module: nn.Module):
                target_param_names = ['layer_scale', 'gamma']

                for param_name in target_param_names:
                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        
                        if isinstance(param, nn.Parameter) and param.dim() >= 1:
                            old_channels = param.shape[0]
                            
                            raw_new = old_channels * ensembles * ch_scale
                            new_channels = align_channels(raw_new, divisor=ensembles)

                            if old_channels != new_channels:
                                with torch.no_grad():
                                    repeat_factor = (new_channels // old_channels) + 1
                                    repeats = [repeat_factor] + [1] * (param.dim() - 1) # dim=3 (C,1,1), factor=2 の場合 [2, 1, 1]
                                    
                                    new_data = param.data.repeat(*repeats)[:new_channels]

                                setattr(module, param_name, nn.Parameter(new_data))

            self.apply_policies(model, [policy_classifier, policy_conv, policy_batchnorm, policy_layernorm, policy_linear])

            if arch == "auto":
                arch = self.get_arch()
                
            if arch in ("convnext"):
                self.apply_policies(model, [policy_layer_scale_param])
            
        return self.pipe(_ee_convert, ensembles=ensembles, ch_scale=ch_scale, arch=arch, flex_ch=flex_ch)
    
    def ee_wrapper(self, ensembles: int, agg: Literal["mean", "sum", "none", "both"]) -> nn.Module:
        return self.pipe(EEWrapper, ensembles=ensembles, agg=agg)

    @property
    def _linear_types(self):
        return (nn.Linear, GroupedLinear, GroupedLinearConv1d)

    def _apply_init(self, m: nn.Module, init_func: Callable[[Tensor], None], **kwargs):
        ee_init = kwargs.get('ee_init', False)
        is_grouped = hasattr(m, 'groups') and m.groups > 1
        
        if ee_init and is_grouped:
            for w in m.weight.chunk(m.groups, dim=0):
                init_func(w)
        else:
            init_func(m.weight)

        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def init_weights(self, arch: Literal["auto", "resnet", "regnet", "mobilenet", "efficientnet", "convnext"] = "auto", ee_init: bool = True) -> "Refiner":
        return super().init_weights(arch=arch, ee_init=ee_init)

class EEWrapper(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        ensembles: int,
        agg: Literal["mean", "sum", "none", "both"],
    ):
        super().__init__()
        
        self.model = model
        self.ensembles = ensembles
        self.agg = agg
        
        self.repeater = RepeatData(n=self.ensembles)
        self.chunk_merge = ChunkMerge(chunks=self.ensembles, agg=self.agg)

    def forward(self, x: torch.Tensor):
        x = self.repeater(x)
        x = self.model(x)
        x = self.chunk_merge(x)
        return x

class RepeatData(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        # self.data_dim = data_dim
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self.n})'

    def forward(self, input):
        num_dims = input.dim()
        # eff_dim = self.data_dim + 1 # batch次元を考慮
        # rep_pattern = [self.n if i == eff_dim else 1 for i in range(num_dims)]
        rep_pattern = [self.n if i == 1 else 1 for i in range(num_dims)] # [1, n, 1, ..., 1]

        return input.repeat(rep_pattern)
        
class ChunkMerge(nn.Module):
    def __init__(self, chunks, agg):
        super().__init__()
        
        assert isinstance(chunks, int) and chunks > 0, f'chunks must be positive int, got {chunks}'
        assert agg in ("mean", "sum", "none", "both"), f"Unsupported agg: {agg}"
        self.chunks = chunks
        self.agg = agg

        def __repr__(self) -> str:
            return f'{self.__class__.__name__}(chunks={self.chunks}, agg={self.agg})'
        
    def forward(self, input: torch.Tensor):
        eff_dim = 1  # チャネル次元 (N, C, H, W) の C に対応
        channels = input.size(eff_dim)

        if channels % self.chunks != 0:
            raise ValueError(f"Channel size {channels} is not divisible by chunks {self.chunks}. This ChunkMerge requires exact divisibility.")

        new_shape = list(input.shape) # input.shape = (N, C, H, W) -> view (N, chunks, channels_per_chunk, H, W)
        channels_per_chunk = channels // self.chunks
        new_shape[eff_dim:eff_dim+1] = [self.chunks, channels_per_chunk] # スライス代入
        reshaped = input.contiguous().view(*new_shape) # reshaped shape: (N, chunks, channels_per_chunk, ...)

        if self.agg == "mean":
            return reshaped.mean(dim=1)
        elif self.agg == "sum":
            return reshaped.sum(dim=1)
        elif self.agg == "none":
            chunks = reshaped.unbind(dim=1)
            return chunks
        elif self.agg == "both":
            chunks = reshaped.unbind(dim=1)
            agg = reshaped.mean(dim=1)
            return agg, chunks
        else:
            raise ValueError(f"Unsupported agg: {self.agg}")
 
class GroupedLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "groups"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        groups: int = 1,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if groups < 1:
            raise ValueError("groups must be >= 1")
        self.in_features = in_features if in_features is not None else 0
        self.out_features = out_features
        self.groups = groups
        if in_features is None:
            self.weight = UninitializedParameter()
            if bias:
                self.bias = UninitializedParameter()
            else:
                self.register_parameter("bias", None)
        else:
            if in_features % groups != 0 or out_features % groups != 0:
                raise ValueError("in_features and out_features must be divisible by groups")
            self.in_per_group = in_features // groups
            self.out_per_group = out_features // groups
            self.weight = Parameter(torch.empty((groups, self.out_per_group, self.in_per_group), **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()

    def _initialize_parameters(self, in_features: int, *, device: Any = None, dtype: Any = None) -> None:
        if in_features % self.groups != 0 or self.out_features % self.groups != 0:
            raise ValueError("in_features and out_features must be divisible by groups")
        self.in_features = in_features
        self.in_per_group = in_features // self.groups
        self.out_per_group = self.out_features // self.groups
        factory_kwargs = {"device": device, "dtype": dtype}
        kw = {k: v for k, v in factory_kwargs.items() if v is not None}
        self.weight = Parameter(torch.empty((self.groups, self.out_per_group, self.in_per_group), **kw))
        if hasattr(self, "bias") and isinstance(self.bias, UninitializedParameter):
            self.bias = Parameter(torch.empty(self.out_features, **kw))
        elif not hasattr(self, "bias"):
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if isinstance(self.weight, UninitializedParameter):
            return
        for g in range(self.groups):
            init.kaiming_uniform_(self.weight[g], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_per_group
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if isinstance(self.weight, UninitializedParameter):
            self._initialize_parameters(input.shape[-1], device=input.device, dtype=input.dtype)

        if input.shape[-1] != self.in_features:
            raise RuntimeError(f"last dim of input must be {self.in_features}, got {input.shape[-1]}")

        if self.groups == 1:
            w = self.weight.view(self.out_features, self.in_features)
            return F.linear(input, w, self.bias)

        leading_shape = input.shape[:-1]
        batch_flat = 1
        for s in leading_shape:
            batch_flat *= s
        x = input.reshape(batch_flat, self.in_features)
        x = x.view(batch_flat, self.groups, self.in_per_group)
        out = torch.einsum("bgi,goi->bgo", x, self.weight)
        out = out.reshape(batch_flat, self.out_features)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        out = out.view(*leading_shape, self.out_features)
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, groups={self.groups}, bias={self.bias is not None}"
       
class GroupedLinearConv1d(nn.Module):
    """グループ化されたLinear層"""
    
    def __init__(self, in_features: int, out_features: int, 
                 groups: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            groups=groups,
            bias=bias
        )
    
    def forward(self, x):
        if x.dim() != 3:
            batch_size = x.size(0)
            x = x.view(batch_size, -1, 1)
        x = self.conv(x)
        x = x.view(x.size(0), self.out_features)
        return x
    
class _PermutedGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.movedim(-1, 1)
        x = self.gn(x)
        x = x.movedim(1, -1)
        return x

class GroupedLayerNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.num_channels)
        x = self.gn(x)
        x = x.view(original_shape)
        return x
    
class GroupedLayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 1, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.layer = GroupedLayerNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1) 
        x = self.layer(x)
        x = x.permute(0, 3, 1, 2)
        
        return x
