import math
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter, UninitializedParameter


class EEConvert(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        ensembles: int,
        width: int | float = 1,
        agg: Literal["mean", "sum", "none", "both"] = "mean", # 指定したくないけどwidth=1のときにnoneにできないので仕方なく
    ):
        super().__init__()
        
        self.ensembles = ensembles
        self.width = width
        self.agg = agg
        self.model = model
        
        self._first_layer_processed = False
        self._convert_layers(self.model)

        self.repeater = RepeatData(n=self.ensembles)
        self.chunk_merge = ChunkMerge(chunks=self.ensembles, agg=self.agg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.repeater(x)
        x = self.model(x)
        x = self.chunk_merge(x)
        return x

    def _convert_layers(self, module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if self._first_layer_processed:
                    new_in_channels = int(child.in_channels * self.ensembles * self.width)
                else:
                    new_in_channels = child.in_channels * self.ensembles
                    self._first_layer_processed = True

                new_child = child.__class__(
                    in_channels=new_in_channels,
                    out_channels=int(child.out_channels * self.ensembles * self.width),
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups * self.ensembles,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                setattr(module, name, new_child)

            elif isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                new_child = child.__class__(
                    num_features=int(child.num_features * self.ensembles * self.width),
                    eps=child.eps,
                    momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats,
                    device=child.weight.device if child.affine else None,
                    dtype=child.weight.dtype if child.affine else None,
                )
                setattr(module, name, new_child)

            elif isinstance(child, nn.Linear):
                new_child = GroupedLinear(
                    in_features=int(child.in_features * self.ensembles * self.width),
                    out_features=int(child.out_features * self.ensembles),      # ここだけ特殊 分類器じゃない場合は判定の上分岐が必要
                    bias=child.bias is not None,
                    groups=self.ensembles,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                setattr(module, name, new_child)
                
            else:
                self._convert_layers(child)

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
        
class ChunkMerge(nn.Module):
    def __init__(self, chunks, agg: Literal["mean", "sum", "none", "both"]):
        super().__init__()
        
        assert isinstance(chunks, int) and chunks > 0, f'chunks must be positive int, got {chunks}'
        assert agg in ("mean", "sum", "none", "both"), f"Unsupported agg: {agg}"
        self.chunks = chunks
        self.agg = agg

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
