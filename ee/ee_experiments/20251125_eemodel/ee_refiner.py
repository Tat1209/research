import copy
import fnmatch
import math
import warnings
from functools import partial
from typing import Any, Callable, List, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter, UninitializedParameter


class EEWrapper(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        ensembles: int,
        agg: Literal["mean", "sum", "none", "both"],
    ):
        super().__init__()
        
        self.ensembles = ensembles
        self.agg = agg
        self.model = model
        
        self.repeater = RepeatData(n=self.ensembles)
        self.chunk_merge = ChunkMerge(chunks=self.ensembles, agg=self.agg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
 
class PermutedGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.movedim(-1, 1)
        x = self.gn(x)
        x = x.movedim(1, -1)
        return x

class LazyPipeline:
    def __init__(self, seed_obj: Any, _steps: List[dict] = None):
        self._seed_obj = seed_obj
        self._steps = _steps or []

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.history()}>"

    def history(self) -> str:
        step_strs = []
        for step in self._steps:
            args = ", ".join(
                [str(a) for a in step['args']] + 
                [f"{k}={v}" for k, v in step['kwargs'].items()]
            )
            step_strs.append(f"{step['func'].__name__}({args})")
        
        base_name = getattr(self._seed_obj, '__name__', 'Base')
        if not isinstance(base_name, str):
            base_name = 'Base'
            
        return ".".join([base_name] + step_strs)

    def pipe(self, func: Callable, *args, **kwargs) -> 'LazyPipeline':
        step_info = {'func': func, 'args': args, 'kwargs': kwargs}
        return self.__class__(self._seed_obj, self._steps + [step_info])

    def _execute(self, target_obj: Any) -> Any:
        current_obj = target_obj
        for step in self._steps:
            new_result = step['func'](current_obj, *step['args'], **step['kwargs'])
            if new_result is not None:
                current_obj = new_result
        return current_obj

class Refiner(LazyPipeline):
    def __init__(self, model: nn.Module, _steps: List = None):
        super().__init__(model, _steps)
    
    @property
    def model(self) -> nn.Module:
        return self._seed_obj
    
    @property
    def _linear_types(self):
        return (nn.Linear,)

    def build(self, _inplace=False, dbg=False) -> nn.Module:
        # inplaceは未実装．forwardの変換が必要で，新たにnn.Moduleを返す場合，in-placeはできないっぽい．
        target_model = self.model if _inplace else self._check_and_copy()
        model = self._execute(target_model)
        
        if dbg:
            print(self.history())
        
        if _inplace:
            return
        else:
            return model

    def _check_and_copy(self) -> nn.Module:
        """GPU上のモデルをコピーする際のリスク管理"""
        has_gpu_tensor = any(p.is_cuda for p in self.model.parameters()) or \
                         any(b.is_cuda for b in self.model.buffers())

        if has_gpu_tensor:
            warnings.warn(
                "Warning: Deepcopying a model located on GPU. "
                "This doubles VRAM usage temporarily.", 
                UserWarning
            )
        
        return copy.deepcopy(self.model)
                
    def apply_policies(self, module, policies, full_path=False, _prefix=""):
        """
        モジュールを再帰的に走査し、policies (関数またはそのリスト) を適用します。
        
        policy_fn(name, child) の戻り値による挙動:
        - None (暗黙含む): ヒットせず。子モジュールへの再帰探索を継続します。
        - module (自分自身): 自身に置き換え (=置換せず)、探索を終了します。
        - new_module (別のもの): new_module に置換し、探索を終了します。
        
        Args:
            full_path (bool): 判定に用いる名前の種別を指定します。
                - False: 自身の `name` (例: "conv1") で判定します。
                - True: ルートからの `full_path` (例: "layer1.conv1") で判定します。
            _prefix (str, optional): 再帰用の内部変数です。指定不要です。

        Example:
            def policy(name, module):
                if isinstance(module, nn.Linear):
                    return nn.Identity()  # 置換して探索終了
                if name == "frozen_block":
                    return module         # 自身に置き換え (=置換せず)、探索終了
                # else return             # (or Implicit None) -> 探索を継続

            self.apply_policies(model, [policy_linear])                  # name には "conv1" 等が渡る
            self.apply_policies(model, [policy_linear], full_path=True)  # name には "backbone.layer1.conv1" 等が渡る
        """
        if not isinstance(policies, (list, tuple)):
            policies = [policies]

        for name, child in module.named_children():
            if full_path:
                # _prefix は常に文字列なので直接判定可能
                identifier = f"{_prefix}.{name}" if _prefix else name
                next_prefix = identifier
            else:
                identifier = name
                next_prefix = ""

            for policy_fn in policies:
                new_child = policy_fn(identifier, child)

                if new_child is not None:
                    if new_child is not child:
                        setattr(module, name, new_child)
                    break
            else:
                self.apply_policies(child, policies, full_path=full_path, _prefix=next_prefix)
        
    def _match_layername(self, name: str, targets: str | List[str]) -> bool:
        if isinstance(targets, str):
            targets = [targets]

        return any(fnmatch.fnmatch(name, t) or fnmatch.fnmatch(name, f"*model.{t}") for t in targets) # model.付きも念のため判定

    def remove_downsample(self, target_layer: List[str] | str) -> "Refiner":
        def _remove_downsample(model, target_layer=target_layer):
            def policy_fn(name, module: nn.Module) -> nn.Module | None:
                if self._match_layername(name, target_layer):
                    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                        return module.__class__(
                            in_channels=module.in_channels,
                            out_channels=module.out_channels,
                            kernel_size=module.kernel_size,
                            stride=1,
                            padding=module.padding,
                            dilation=module.dilation,
                            groups=module.groups,
                            bias=module.bias is not None,
                            padding_mode=module.padding_mode,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                        )
                    elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
                        return nn.Identity()

            self.apply_policies(model, policy_fn, full_path=True)

        return self.pipe(_remove_downsample, target_layer=target_layer)

    def kernel_size_adjust(self, target_layer: List[str] | str, to_size: int, padding: int | None = None) -> "Refiner":
        if padding is None:
            padding = (to_size - 1) // 2

        def _kernel_size_adjust(model, target_layer=target_layer, to_size=to_size, padding=padding):
            def policy_fn(name, module: nn.Module) -> nn.Module | None:
                if self._match_layername(name, target_layer):
                    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                        return module.__class__(
                            in_channels=module.in_channels,
                            out_channels=module.out_channels,
                            kernel_size=to_size,
                            stride=module.stride,
                            padding=padding,
                            dilation=module.dilation,
                            groups=module.groups,
                            bias=module.bias is not None,
                            padding_mode=module.padding_mode,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                        )
            self.apply_policies(model, policy_fn, full_path=True)

        return self.pipe(_kernel_size_adjust, target_layer=target_layer)

    def cifar_style(self, arch: Literal["auto", "resnet", "regnet", "mobilenet", "efficientnet", "convnext"] = "auto") -> "Refiner":
        # 必ず最初に適用すること．full_pathがの完全一致で判定するため，ラップされてからだとmatchしない．（一応，_match_layernameにmodel.も同時に判定する．）
        # CIFAR画像を入力した際，4x4の特徴マップがGAPに入るように調整する
        if arch == "auto":
            arch = self.get_arch()
            
        if arch in ("resnet"):
            return self.remove_downsample(["conv1", "maxpool"]).kernel_size_adjust("conv1", to_size=3, padding=1)

        if arch in ("regnet"):
            return self.remove_downsample(["stem.0", "trunk_output.block1.block1-0.*"])

        if arch in ("mobilenet"):
            return self.remove_downsample(["features.0.0", "features.2.conv.0.0"])

        if arch in ("efficientnet"):
            return self.remove_downsample(["features.0.0", "features.2.0.block.1.0"])

        if arch in ("convnext"):
            return self.remove_downsample(["features.0.0"]).kernel_size_adjust("features.0.0", to_size=3, padding=1)
        return self

    def _apply_init(self, m: nn.Module, init_func: Callable[[Tensor], None], **kwargs):
        init_func(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def init_weights(self, arch: Literal["auto", "resnet", "regnet", "mobilenet", "efficientnet", "convnext"] = "auto", **kwargs) -> "Refiner":
        def _init_weights(model, arch=arch, **kwargs):
            if arch == "auto":
                arch = self.get_arch()

            if arch == "resnet":
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        fn = partial(nn.init.kaiming_normal_, mode="fan_out", nonlinearity="relu")
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

            elif arch == "regnet":
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        def regnet_init(w):
                            fan_out = w.size(0) * w.size(2) * w.size(3)
                            nn.init.normal_(w, mean=0.0, std=math.sqrt(2.0 / fan_out))
                        self._apply_init(m, regnet_init, **kwargs)
                    elif isinstance(m, self._linear_types):
                        fn = partial(nn.init.normal_, mean=0.0, std=0.01)
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

            elif arch == "mobilenet":
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        fn = partial(nn.init.kaiming_normal_, mode="fan_out")
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, self._linear_types):
                        fn = partial(nn.init.normal_, mean=0.0, std=0.01)
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

            elif arch == "efficientnet":
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        fn = partial(nn.init.kaiming_normal_, mode="fan_out")
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, self._linear_types):
                        init_range = 1.0 / math.sqrt(m.out_features)
                        fn = partial(nn.init.uniform_, a=-init_range, b=init_range)
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

            elif arch == "convnext":
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, *self._linear_types)):
                        fn = partial(nn.init.trunc_normal_, std=0.02)
                        self._apply_init(m, fn, **kwargs)

            return model

        return self.pipe(_init_weights, arch=arch, **kwargs)

    def get_arch(self) -> str:
        model_name = self.model.__class__.__name__.lower()
        if "resnet" in model_name:
            return "resnet"
        elif "mobilenet" in model_name:
            return "mobilenet"
        elif "convnext" in model_name:
            return "convnext"
        elif "efficientnet" in model_name:
            return "efficientnet"
        elif "regnet" in model_name:
            return "regnet"
        else:
            warnings.warn(
                    f"Unknown model architecture detected: '{model_name}'. "
                    "Returned 'unknown'. Check if logic update is needed.",
                    category=UserWarning,
                    stacklevel=2
                )
            return "unknown"

class EERefiner(Refiner):
    def __init__(self, model: nn.Module, _steps: List = None):
        super().__init__(model, _steps)
        
    def multi_narrow(self, div: int | None = 1, agg: Literal["mean", "sum", "none", "both"] = "both", arch: Literal["auto", "resnet", "mobilenet", "efficientnet", "convnext", "regnet"] = "auto", flex_ch=False, flex_mode: Literal["round", "floor", "cail"] = "round", grouped_linear_impl: Literal["einsum", "conv"] = "einsum") -> nn.Module:
        """
        分割数(div)からアンサンブル数(div^2)と幅(1/div)を自動決定し、パラメータ数を維持して分割します。

        Args:
            div (int): 分割係数。Ens数=div^2, 幅=1/div となる (例: div=2 -> 4モデル, 幅0.5)。
            agg (Literal): 集約方法 ("mean", "sum", "none", "both")。
            flex_ch (bool): チャンネルが割り切れない場合に調整を行うか。
            flex_mode (Literal): flex_ch有効時の端数処理 ("round", "floor", "ceil")。

        Returns:
            nn.Module: アンサンブル化されたモデル。

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
            agg (Literal): 集約方法。
            flex_ch (bool): チャンネル数調整の可否。端数が出る倍率指定時に有効。

        Returns:
            nn.Module: アンサンブル化されたモデル。

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
                        TargetNormClass = PermutedGroupNorm
                    else:
                        TargetNormClass = nn.GroupNorm # ケースB: LayerNorm2d (特定のカスタムクラス) Stem/Downsampleなど、NCHW (Channel First) でデータが来る場所

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
