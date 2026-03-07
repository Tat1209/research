import math
import sys
from pathlib import Path
from typing import Any, Callable, List, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter, UninitializedParameter
from torchvision.ops import Conv2dNormActivation

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
        """
        ensembles = round(div ** 2)
        ch_scale = 1 / div
        return self.easy_ensemble(ensembles=ensembles, ch_scale=ch_scale, agg=agg, arch=arch, flex_ch=flex_ch, flex_mode=flex_mode, grouped_linear_impl=grouped_linear_impl)
        
    def easy_ensemble(self, ensembles: int = 1, ch_scale: int | float = 1, agg: Literal["mean", "sum", "none", "both"] = "mean", arch: Literal["auto", "resnet", "mobilenet", "efficientnet", "convnext", "regnet"] = "auto", flex_ch=False, flex_mode: Literal["round", "floor", "ceil"] = "round", grouped_linear_impl: Literal["einsum", "conv"] = "einsum") -> nn.Module:
        """
        指定した数(ensembles)と幅(ch_scale)でモデルを分割・アンサンブル化します。
        """
        return self.ee_convert(ensembles, ch_scale, arch, flex_ch, flex_mode, grouped_linear_impl).ee_wrapper(ensembles, agg)
    
    def ee_convert(self, ensembles: int, ch_scale: int | float, arch, flex_ch: bool, flex_mode: Literal["round", "floor", "ceil"], grouped_linear_impl: Literal["einsum", "conv"]) -> nn.Module:

        def _ee_convert(model, ensembles=ensembles, ch_scale=ch_scale, arch=arch, flex_ch=flex_ch, flex_mode=flex_mode, grouped_linear_impl=grouped_linear_impl):
            if ensembles == 1 and ch_scale == 1:
                return model

            first_layer_processed = False
            _arch = arch
            if _arch == "auto":
                _arch = self.get_arch()

            # --- Helpers ---
            def align_channels(target_channels: float, divisor: int) -> int:
                c = round(target_channels)
                if not flex_ch: return c
                if c < divisor: return divisor
                
                if flex_mode == 'floor': return (c // divisor) * divisor
                elif flex_mode == 'ceil':
                    if c % divisor == 0: return c
                    return ((c // divisor) + 1) * divisor
                elif flex_mode == 'round': return round(c / divisor) * divisor
                else: raise ValueError(f"Invalid flex_mode: {flex_mode}")

            def rebuild_conv(module, in_channels=None, out_channels=None, groups=None):
                return module.__class__(
                    in_channels=in_channels if in_channels is not None else module.in_channels,
                    out_channels=out_channels if out_channels is not None else module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=groups if groups is not None else module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )

            def rebuild_bn(module, num_features):
                return module.__class__(
                    num_features=num_features,
                    eps=module.eps,
                    momentum=module.momentum,
                    affine=module.affine,
                    track_running_stats=module.track_running_stats,
                    device=module.weight.device if module.affine else None,
                    dtype=module.weight.dtype if module.affine else None,
                )
            
            def rebuild_linear(module, in_features=None, out_features=None, groups=None):
                if grouped_linear_impl == "einsum": GroupedLinearClass = GroupedLinear
                elif grouped_linear_impl == "conv": GroupedLinearClass = GroupedLinearConv1d
                
                # 通常のLinearかGroupedLinearか判定して適切に構築
                is_grouped = isinstance(module, (GroupedLinear, GroupedLinearConv1d)) or (groups is not None and groups > 1)
                
                if is_grouped:
                    return GroupedLinearClass(
                        in_features=in_features if in_features is not None else module.in_features,
                        out_features=out_features if out_features is not None else module.out_features,
                        bias=module.bias is not None,
                        groups=groups if groups is not None else getattr(module, 'groups', ensembles),
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                else:
                    # 分割なしの単純なLinear再構築用(今回は主にGrouped化に使用するため上側を通る)
                    return nn.Linear(
                        in_features=in_features if in_features is not None else module.in_features,
                        out_features=out_features if out_features is not None else module.out_features,
                        bias=module.bias is not None,
                        device=module.weight.device,
                        dtype=module.weight.dtype
                    )

            def rebuild_layernorm(module, num_channels, num_groups=None):
                if type(module) is nn.LayerNorm: TargetNormClass = GroupedLayerNorm
                else: TargetNormClass = GroupedLayerNorm2d
                
                return TargetNormClass(
                    num_groups=num_groups if num_groups is not None else ensembles,
                    num_channels=num_channels,
                    eps=module.eps,
                )

            def policy_classifier(name, module: nn.Module):
                target_names = {'fc', 'classifier', 'head'}
                
                if name in target_names:
                    # 内部のLinearだけを変換するためのサブ関数
                    def convert_fc_internal(m):
                        if isinstance(m, nn.Linear):
                            raw_in = m.in_features * ensembles * ch_scale
                            raw_out = m.out_features * ensembles
                            # LinearはGroups=Ensemblesにするが、出力層(FC)なのでch_scaleは入力にのみ掛かる
                            # (出力はクラス数xアンサンブル数)
                            return rebuild_linear(m, 
                                                  in_features=align_channels(raw_in, divisor=ensembles), 
                                                  out_features=align_channels(raw_out, divisor=ensembles),
                                                  groups=ensembles)
                        return None 

                    # Sequentialの場合: Linear変換だけでなく、Norm変換(policy_layernorm)も適用する
                    if isinstance(module, nn.Sequential):
                         # 修正: リストに policy_layernorm を追加
                         self.apply_policies(module, [lambda n, m: convert_fc_internal(m), policy_layernorm])
                         return module
                    elif isinstance(module, nn.Linear):
                        return convert_fc_internal(module)
                return None
            def policy_conv(name, module: nn.Module):
                nonlocal first_layer_processed
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    is_depthwise = module.in_channels == module.out_channels and module.groups == module.in_channels
                    if is_depthwise: align_divisor = ensembles
                    else: align_divisor = module.groups * ensembles

                    if first_layer_processed:
                        raw_in = module.in_channels * ensembles * ch_scale
                    else:
                        raw_in = module.in_channels * ensembles
                        first_layer_processed = True
                    
                    new_in_channels = align_channels(raw_in, divisor=align_divisor)
                    new_groups = new_in_channels if is_depthwise else align_divisor
                    
                    raw_out = module.out_channels * ensembles * ch_scale
                    new_out_channels = align_channels(raw_out, divisor=new_groups)

                    return rebuild_conv(module, in_channels=new_in_channels, out_channels=new_out_channels, groups=new_groups)

            def policy_batchnorm(name, module: nn.Module):
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    raw_features = module.num_features * ensembles * ch_scale
                    new_features = align_channels(raw_features, divisor=ensembles)
                    return rebuild_bn(module, num_features=new_features)

            def policy_layernorm(name, module: nn.Module) -> nn.Module | None:
                if isinstance(module, nn.LayerNorm):
                    if isinstance(module.normalized_shape, (int, float)):
                        old_channels = int(module.normalized_shape)
                    else:
                        old_channels = int(module.normalized_shape[0])
                    raw_channels = old_channels * ensembles * ch_scale
                    new_channels = align_channels(raw_channels, divisor=ensembles)
                    return rebuild_layernorm(module, num_channels=new_channels)
            
            def policy_linear(name, module: nn.Module):
                if isinstance(module, nn.Linear):
                    raw_in = module.in_features * ensembles * ch_scale
                    raw_out = module.out_features * ensembles * ch_scale
                    return rebuild_linear(module, 
                                          in_features=align_channels(raw_in, divisor=ensembles), 
                                          out_features=align_channels(raw_out, divisor=ensembles),
                                          groups=ensembles)

            def policy_cna(name, module: nn.Module):
                if isinstance(module, Conv2dNormActivation):
                    # 標準的な変換を適用
                    self.apply_policies(module, [policy_conv, policy_batchnorm, policy_layernorm])
                    
                    # 不整合の修正 (Conv -> Norm の接続)
                    last_channels = None
                    for i, layer in enumerate(module):
                        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                            last_channels = layer.out_channels
                        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                            if last_channels is not None and layer.num_features != last_channels:
                                module[i] = rebuild_bn(layer, num_features=last_channels)
                        elif isinstance(layer, nn.LayerNorm) or (hasattr(layer, 'normalized_shape') and hasattr(layer, 'eps')): # LayerNorm系
                             # GroupedLayerNormへの置換等はpolicy_layernormで行われている前提
                             # ここではチャンネル数の不整合だけを簡易チェックしてもよいが、CNBlock内のLNは特殊なので後述のブロックポリシーで扱う
                             pass
                    return module
                return None

            # --- RegNet Specific Policy ---
            def policy_bottleneck_transform(name, module: nn.Module):
                if type(module).__name__ == "BottleneckTransform":
                    self.apply_policies(module, [policy_cna, policy_conv, policy_batchnorm, policy_layernorm])
                    try:
                        conv_a = module[0][0]
                        conv_b = module[1][0]
                        if conv_a.out_channels != conv_b.in_channels:
                            target_mid = max(conv_a.out_channels, conv_b.in_channels)
                            module[0][0] = rebuild_conv(conv_a, out_channels=target_mid)
                            module[0][1] = rebuild_bn(module[0][1], num_features=target_mid)
                            module[1][0] = rebuild_conv(conv_b, in_channels=target_mid, out_channels=target_mid)
                            module[1][1] = rebuild_bn(module[1][1], num_features=target_mid)
                            
                            # SE Block Correction
                            if len(module) > 3: 
                                se_module = module[2]
                                if hasattr(se_module, 'fc1') and isinstance(se_module.fc1, nn.Conv2d):
                                    if se_module.fc1.in_channels != target_mid:
                                        se_module.fc1 = rebuild_conv(se_module.fc1, in_channels=target_mid)
                                if hasattr(se_module, 'fc2') and isinstance(se_module.fc2, nn.Conv2d):
                                    if se_module.fc2.out_channels != target_mid:
                                        se_module.fc2 = rebuild_conv(se_module.fc2, out_channels=target_mid)
                            
                            # Conv C Correction
                            conv_c = module[-1][0]
                            if conv_c.in_channels != target_mid:
                                module[-1][0] = rebuild_conv(conv_c, in_channels=target_mid)
                    except (IndexError, AttributeError): pass
                    return module
                return None

            # --- ConvNeXt Specific Policy ---
            def policy_cnblock(name, module: nn.Module):
                """ConvNeXt Block (CNBlock) の内部整合性を保つポリシー"""
                if hasattr(module, "block") and isinstance(module.block, nn.Sequential):
                    # 1. まず構成要素を個別に変換 (再帰的に適用)
                    # ここで Conv, LayerNorm, Linear が拡張される
                    self.apply_policies(module, [policy_conv, policy_layernorm, policy_linear])

                    # 2. Layer Scale Parameter (gamma) の再調整 [優先実行・try外へ移動]
                    # これを try ブロックに入れると、他の箇所の属性エラーでスキップされる恐れがあるため分離
                    if hasattr(module, "layer_scale"):
                        # policy_layer_scale_param を直接呼び出して処理
                        policy_layer_scale_param(name, module)

                    # 3. ブロック内部の次元不整合を修正 (Repair処理)
                    # 構造: [0:DWConv, 1:Permute, 2:LayerNorm, 3:Linear(Exp), 4:GELU, 5:Linear(Proj), 6:Permute]
                    try:
                        dw_conv = module.block[0]
                        norm = module.block[2]
                        pw_linear_exp = module.block[3]
                        pw_linear_proj = module.block[5]

                        # 基準となる次元 (DW Convの出力)
                        # Convが拡張されていれば、ここが正しい次元 (例: 768) になっているはず
                        base_dim = dw_conv.out_channels

                        # Check 1: LayerNormの入力次元
                        norm_dim = norm.normalized_shape if isinstance(norm.normalized_shape, (int, float)) else norm.normalized_shape[0]
                        if hasattr(norm, "num_channels"): norm_dim = norm.num_channels # GroupedLayerNorm対応

                        if norm_dim != base_dim:
                            module.block[2] = rebuild_layernorm(norm, num_channels=base_dim)

                        # Check 2: Expansion Linearの入力次元
                        # GroupedLinearの場合も考慮して getattr で安全に取得
                        exp_in_feat = getattr(pw_linear_exp, "in_features", None)
                        if exp_in_feat is not None and exp_in_feat != base_dim:
                            module.block[3] = rebuild_linear(pw_linear_exp, in_features=base_dim)

                        # Check 3: Expansion (Linear1出力) と Projection (Linear2入力) の整合性
                        exp_out_feat = getattr(pw_linear_exp, "out_features", None)
                        proj_in_feat = getattr(pw_linear_proj, "in_features", None)
                        
                        if exp_out_feat is not None and proj_in_feat is not None:
                            if proj_in_feat != exp_out_feat:
                                # Expansion側に合わせる
                                module.block[5] = rebuild_linear(pw_linear_proj, in_features=exp_out_feat)

                    except (IndexError, AttributeError) as e:
                        # 構造が想定と違う場合や属性がない場合はRepairをスキップするが
                        # 上記の layer_scale 修正は完了しているのでエラーは回避できるはず
                        pass
                    
                    # 自身を返して探索終了 (再帰適用済みのため)
                    return module
                return None

            def policy_layer_scale_param(name, module: nn.Module):
                # CNBlock以外にある単独のLayerScale用パラメータへの対応
                target_param_names = ['layer_scale', 'gamma']
                for param_name in target_param_names:
                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        if isinstance(param, nn.Parameter) and param.dim() >= 1:
                            if isinstance(module, (nn.Conv2d, nn.Linear)): 
                                # Conv/Linear自体の変換ポリシーでch数は変わっているはずなので、それに合わせる
                                target_dim = module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
                            elif hasattr(module, "normalized_shape"): # LayerNorm
                                target_dim = module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape
                            else:
                                # 親モジュールの情報がない場合、パラメータ自体から推測して拡張
                                old_channels = param.shape[0]
                                target_dim = align_channels(old_channels * ensembles * ch_scale, divisor=ensembles)

                            if param.shape[0] != target_dim:
                                with torch.no_grad():
                                    repeat_factor = (target_dim // param.shape[0]) + 1
                                    repeats = [repeat_factor] + [1] * (param.dim() - 1)
                                    new_data = param.data.repeat(*repeats)[:target_dim]
                                setattr(module, param_name, nn.Parameter(new_data))


            # --- Execution ---
            policies = [policy_classifier, policy_cna, policy_conv, policy_batchnorm, policy_layernorm, policy_linear]
            
            # アーキテクチャ固有の強力なポリシーを優先度高く挿入
            if _arch == "regnet":
                policies.insert(1, policy_bottleneck_transform)
            elif _arch == "convnext":
                policies.insert(1, policy_cnblock) # CNBlockを一括処理
                policies.append(policy_layer_scale_param) # 念のため個別パラメータも拾う

            self.apply_policies(model, policies)
            
            return model

        return self.pipe(_ee_convert, ensembles=ensembles, ch_scale=ch_scale, arch=arch, flex_ch=flex_ch, flex_mode=flex_mode, grouped_linear_impl=grouped_linear_impl)

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
