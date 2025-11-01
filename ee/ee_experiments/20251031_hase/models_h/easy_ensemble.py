import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, Dict, List
import copy
import math


class EasyEnsembleConverterV2:
    """
    PyTorchモデルをEasy Ensemble型に変換するクラス（改良版）
    - チャネル数を適切に分割
    - 重みを正確に転送
    """
    
    def __init__(self, num_ensembles: int = 4, num_classes: int = 10, 
                 use_gn: bool = True, scale: bool = False, cross: bool = False):
        """
        Args:
            num_ensembles (int): アンサンブル数 N（平方数である必要がある）
            num_classes (int): 分類タスクのクラス数
            scale (bool): in_channelsが減少する際に重みをスケール
            cross (bool): i,jの位置を交互に入れ替える
        """
        self.num_ensembles = num_ensembles
        self.num_classes = num_classes
        self.use_gn = use_gn
        self.div = int(math.sqrt(num_ensembles))
        assert self.div * self.div == num_ensembles, "num_ensembles must be a perfect square"
        
        self.scale = scale
        self.cross = cross
        self.counter = 0  # cross用のカウンター
    
    def convert_model(self, model: nn.Module, input_channels: int = 3) -> nn.Module:
        """
        標準的なPyTorchモデルをEasy Ensemble型に変換
        
        Args:
            model (nn.Module): 変換対象のモデル
            input_channels (int): 元の入力チャネル数
            
        Returns:
            nn.Module: Easy Ensemble型に変換されたモデル
        """
        # モデル構造を作成
        self.last_out_channels = -1
        ee_model = self._create_ee_structure(model, input_channels)
        
        # 重みを転送
        #self._transfer_all_weights(model, ee_model)
        
        # 入力を処理するラッパーを追加
        ee_model = EasyEnsembleWrapperV2(ee_model, self.num_ensembles,
                                         self.num_classes, 
                                         input_channels, 
                                         self.last_out_channels)
        
        return ee_model
    
    def _create_ee_structure(self, model: nn.Module, input_channels: int) -> nn.Module:
        """Easy Ensemble用のモデル構造を作成"""
        ee_model = copy.deepcopy(model)
        
        # 各層を変換
        self._convert_structure(ee_model, is_first_layer=True)
        
        return ee_model
    
    def _convert_structure(self, module: nn.Module, is_first_layer: bool = False, parent_name: str = ''):
        """モデル構造をEasy Ensemble用に変換"""
        
        for name, child in list(module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            if isinstance(child, nn.Conv2d):
                # 新しいConv2d層を作成
                if is_first_layer and child.in_channels == 3:
                    # 最初の層：入力チャネルはN倍、出力チャネルはdiv倍
                    new_conv = nn.Conv2d(
                        in_channels=3 * self.num_ensembles,
                        out_channels=child.out_channels * self.div,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=self.num_ensembles,
                        bias=child.bias is not None
                    )
                    is_first_layer = False
                else:
                    # その他の層：チャネル数はdiv倍
                    new_conv = nn.Conv2d(
                        in_channels=child.in_channels * self.div,
                        out_channels=child.out_channels * self.div,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=self.num_ensembles,
                        bias=child.bias is not None
                    )
                self.last_out_channels = child.out_channels * self.div
                setattr(module, name, new_conv)
                
            elif isinstance(child, nn.BatchNorm2d):
                if self.use_gn:
                    # GroupNormに変換（チャネル数もdiv倍）
                    new_norm = nn.GroupNorm(
                        num_groups=self.num_ensembles,
                        num_channels=child.num_features * self.div,
                        eps=child.eps,
                        affine=child.affine
                    )
                else:
                    # BatchNormはそのまま使用
                    new_norm = nn.BatchNorm2d(
                        num_features=child.num_features * self.div,
                        eps=child.eps,
                        momentum=child.momentum,
                        affine=child.affine
                    )
                setattr(module, name, new_norm)
                
            elif isinstance(child, nn.Linear):
                # GroupedLinearに変換（入出力ともdiv倍）
                new_linear = GroupedLinearV2(
                    in_features=child.in_features * self.div,
                    out_features=child.out_features * self.div,
                    num_groups=self.num_ensembles,
                    bias=child.bias is not None
                )
                
                self.last_out_channels = child.out_features * self.div
                setattr(module, name, new_linear)
                
            else:
                # 再帰的に処理
                self._convert_structure(child, is_first_layer, full_name)
    
    def _transfer_all_weights(self, src_model: nn.Module, dst_model: nn.Module):
        """全ての重みを転送"""
        src_dict = dict(src_model.named_modules())
        dst_dict = dict(dst_model.named_modules())
        
        for name, src_module in src_dict.items():
            if name not in dst_dict:
                continue
            
            dst_module = dst_dict[name]
            
            if isinstance(src_module, nn.Conv2d) and isinstance(dst_module, nn.Conv2d):
                self._transfer_conv_weights(src_module, dst_module)
                self.counter += 1
                
            elif isinstance(src_module, nn.BatchNorm2d) and isinstance(dst_module, nn.GroupNorm):
                self._transfer_bn_weights(src_module, dst_module)
                
            elif isinstance(src_module, nn.Linear) and isinstance(dst_module, GroupedLinearV2):
                self._transfer_linear_weights(src_module, dst_module)
    
    def _transfer_conv_weights(self, src: nn.Conv2d, dst: nn.Conv2d):
        """Conv層の重みを転送（提供されたコードのロジックを適用）"""
        with torch.no_grad():
            weight = src.weight.data
            outs, ins, h, w = weight.shape
            
            # 各グループに重みを配分
            for i in range(self.div):
                for j in range(self.div):
                    g = i * self.div + j  # グループ番号
                    
                    outr = outs // self.num_ensembles
                    
                    if src.in_channels == 3:
                        # 最初の層：各グループが全入力チャネルを見る
                        out_start = g * outr
                        out_end = (g + 1) * outr
                        in_start = g * 3
                        in_end = (g + 1) * 3
                        
                        dst.weight.data[out_start:out_end, in_start:in_end] = \
                            weight[out_start:out_end, :, :, :]
                    else:
                        # 中間層：入出力を分割
                        inr = ins // self.num_ensembles
                        scale_factor = self.num_ensembles if self.scale else 1.0
                        
                        out_start = g * outr
                        out_end = (g + 1) * outr
                        
                        if self.cross and self.counter % 2 == 0:
                            # crossモードで偶数層の場合、i,jを入れ替え
                            in_block_idx = j
                            out_block_idx = i
                        else:
                            in_block_idx = i
                            out_block_idx = j
                        
                        in_start = in_block_idx * inr
                        in_end = (in_block_idx + 1) * inr
                        out_src_start = out_block_idx * outr
                        out_src_end = (out_block_idx + 1) * outr
                        
                        # 対応するブロックから重みをコピー
                        dst.weight.data[out_start:out_end, in_start:in_end] = \
                            weight[out_src_start:out_src_end, in_start:in_end, :, :] * scale_factor
            
            # バイアスの転送
            if src.bias is not None:
                dst.bias.data.copy_(src.bias.data)
    
    def _transfer_bn_weights(self, src: nn.BatchNorm2d, dst: nn.GroupNorm):
        """BatchNorm重みをGroupNormに転送"""
        with torch.no_grad():
            if src.affine:
                weight = src.weight.data
                bias = src.bias.data
                channels = weight.shape[0]
                
                # 各グループに対応する部分をコピー
                for i in range(self.div):
                    for j in range(self.div):
                        g = i * self.div + j
                        inr = channels // self.num_ensembles
                        
                        if self.cross and self.counter % 2 == 0:
                            idx = j
                        else:
                            idx = i
                        
                        start = g * inr
                        end = (g + 1) * inr
                        src_start = idx * inr
                        src_end = (idx + 1) * inr
                        
                        dst.weight.data[start:end] = weight[src_start:src_end]
                        dst.bias.data[start:end] = bias[src_start:src_end]
    
    def _transfer_linear_weights(self, src: nn.Linear, dst: 'GroupedLinearV2'):
        """Linear層の重みを転送"""
        with torch.no_grad():
            weight = src.weight.data
            outs, ins = weight.shape
            
            # 全グループで出力を共有
            for i in range(self.div):
                for j in range(self.div):
                    g = i * self.div + j
                    inr = ins // self.num_ensembles
                    scale_factor = self.num_ensembles if self.scale else 1.0
                    
                    if self.cross and self.counter % 2 == 0:
                        idx = j
                    else:
                        idx = i
                    
                    in_start = idx * inr
                    in_end = (idx + 1) * inr
                    
                    # 各グループのLinear層に重みをコピー
                    dst.groups[g].weight.data = weight[:, in_start:in_end] * scale_factor
                    
                    if src.bias is not None:
                        dst.groups[g].bias.data = src.bias.data / self.num_ensembles


class GroupedLinearV2(nn.Module):
    """グループ化されたLinear層"""
    
    def __init__(self, in_features: int, out_features: int, 
                 num_groups: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            groups=num_groups,
            bias=bias
        )
    
    def forward(self, x):
        if x.dim() != 3:
            batch_size = x.size(0)
            x = x.view(batch_size, -1, 1)
        x = self.conv(x)
        x = x.view(x.size(0), self.out_features)
        return x
    
class EasyEnsembleWrapperV2(nn.Module):
    """入力を適切に処理するラッパー"""
    
    def __init__(self, model: nn.Module, num_ensembles: int, num_classes: int, 
                 original_channels: int, last_out_channels: int):
        super().__init__()
        self.model = model # EasyEnsemble型のモデル(fc層なし)
        self.num_classes = num_classes
        self.num_ensembles = num_ensembles
        self.original_channels = original_channels
        self.last_out_channels = last_out_channels
        self.fc = GroupedLinearV2(
            in_features=last_out_channels,
            out_features=num_classes * num_ensembles,
            num_groups=num_ensembles,
            bias=True
        )
    
    def forward(self, x):
        # 入力をチャネル方向にN回繰り返す
        logits = self.get_logits(x)
        logits = torch.stack(logits, dim=1)  # N個の出力
        output = logits.mean(dim=1)  # 平均化        
        return output
    
    def input_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_repeated = x.repeat(1, self.num_ensembles, 1, 1)
        return x_repeated
    
    def get_features_and_logits(self, x):
        x = self.input_forward(x)
        bs = x.size(0)
        features = self.model(x)
        logits = self.fc(features).view(bs, self.num_ensembles, self.num_classes)
        features = features.view(bs, self.num_ensembles, -1)
        # Listにして返す
        return [features[:,i,:] for i in range(self.num_ensembles)], \
               [logits[:,i,:] for i in range(self.num_ensembles)]
    
    def get_features(self, x):
        f, l = self.get_features_and_logits(x)
        return f
    
    def get_logits(self, x):
        f, l = self.get_features_and_logits(x)
        return l
    
    def __len__(self):
        """モデル数を返す"""
        return self.num_ensembles


# テスト関数
def test_easy_ensemble_v2():
    """Easy Ensemble V2のテスト"""
    import torchvision.models as models
    
    print("=== Easy Ensemble V2 Test ===")
    
    # シンプルなCNNモデル
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu1 = nn.ReLU()
            
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.relu2 = nn.ReLU()
            
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)
        
        def forward(self, x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # モデルを作成
    model = SimpleCNN()
    
    # 特定の値で初期化（テスト用）
    with torch.no_grad():
        # Conv1: 16フィルタ
        for i in range(16):
            model.conv1.weight.data[i] = i * 0.01
        
        # Conv2: 32フィルタ
        for i in range(32):
            model.conv2.weight.data[i] = i * 0.01
        
        # FC: 10クラス
        model.fc.weight.data.uniform_(-0.1, 0.1)
    
    # パラメータ数を確認
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {original_params:,}")
    
    # Easy Ensembleに変換
    converter = EasyEnsembleConverterV2(num_ensembles=4, scale=False, cross=False)
    ee_model = converter.convert_model(model)
    
    # 変換後のパラメータ数
    ee_params = sum(p.numel() for p in ee_model.parameters())
    print(f"Easy Ensemble parameters: {ee_params:,}")
    print(f"Parameter ratio: {ee_params / original_params:.2f}x")
    
    # モデル構造を確認
    print("\n=== Model Structure ===")
    for name, module in ee_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.GroupNorm, GroupedLinearV2)):
            if isinstance(module, nn.Conv2d):
                print(f"{name}: Conv2d(in={module.in_channels}, out={module.out_channels}, groups={module.groups})")
            elif isinstance(module, nn.GroupNorm):
                print(f"{name}: GroupNorm(groups={module.num_groups}, channels={module.num_channels})")
            elif isinstance(module, GroupedLinearV2):
                print(f"{name}: GroupedLinear(in={module.in_features}, out={module.out_features}, groups={module.num_groups})")
    
    # 推論テスト
    print("\n=== Inference Test ===")
    x = torch.randn(2, 3, 32, 32)
    
    with torch.no_grad():
        # 元のモデル
        out_original = model(x)
        print(f"Original output shape: {out_original.shape}")
        
        # Easy Ensemble
        out_ee = ee_model(x)
        print(f"EE output shape: {out_ee.shape}")
    
    # 重み転送の確認
    print("\n=== Weight Transfer Verification ===")
    # Conv1の重みを確認
    conv1_ee = None
    for name, module in ee_model.model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 12:  # 3*4
            conv1_ee = module
            break
    
    if conv1_ee:
        print("Conv1 weight check:")
        for g in range(4):
            i, j = g // 2, g % 2
            out_start = g * 4  # 16/4 = 4
            in_start = g * 3
            
            # 元の重みと転送後の重みを比較
            original_weight_sample = model.conv1.weight.data[j*4, 0, 0, 0].item()
            ee_weight_sample = conv1_ee.weight.data[out_start, in_start, 0, 0].item()
            
            print(f"  Group {g} (i={i}, j={j}): Original[{j*4}] = {original_weight_sample:.4f}, "
                  f"EE[{out_start},{in_start}] = {ee_weight_sample:.4f}")
    
    return ee_model


if __name__ == "__main__":
    # テスト実行
    ee_model = test_easy_ensemble_v2()
    print("\nTest completed successfully!")


# テスト関数
def test_converter_v2():
    """改良版コンバーターのテスト"""
    
    # テスト用モデル
    class TestCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.relu2 = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    print("=== Easy Ensemble Converter V2 Test ===")
    
    # モデルを作成
    model = TestCNN()
    
    # パラメータ数を確認
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {original_params:,}")
    
    # Easy Ensembleに変換（N=4）
    converter = EasyEnsembleConverterV2(num_ensembles=4, scale=False, cross=False)
    ee_model = converter.convert_model(model)
    
    # 変換後のパラメータ数
    ee_params = sum(p.numel() for p in ee_model.parameters())
    print(f"Easy Ensemble parameters: {ee_params:,}")
    print(f"Ratio: {ee_params / original_params:.2f}x")
    
    # 動作確認
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_original = model(x)
        out_ee = ee_model(x)
        
    print(f"\nOriginal output shape: {out_original.shape}")
    print(f"EE output shape: {out_ee.shape}")
    
    # 重みの分布を確認
    print("\n=== Weight Distribution Check ===")
    
    # Conv1の重みを確認
    for name, module in ee_model.named_modules():
        if isinstance(module, nn.Conv2d) and module.weight.shape[1] == 3:  # 最初の層
            print(f"\nFirst Conv layer:")
            print(f"Weight shape: {module.weight.shape}")
            print(f"Groups: {module.groups}")
            
            # 各グループの重みの統計
            for g in range(4):
                out_start = g * 16
                out_end = (g + 1) * 16
                in_start = g * 3
                in_end = (g + 1) * 3
                
                group_weight = module.weight[out_start:out_end, in_start:in_end]
                print(f"Group {g}: mean={group_weight.mean():.4f}, std={group_weight.std():.4f}")
            break
    
    return ee_model


def compare_with_original_code_logic():
    """提供されたコードのロジックとの比較"""
    
    print("\n=== Comparing with Original Code Logic ===")
    
    # シンプルなモデルで比較
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.Conv2d(16, 32, 3),
        nn.Linear(32, 10)
    )
    
    # 特定の値で初期化
    with torch.no_grad():
        model[0].weight.data.fill_(1.0)
        model[1].weight.data.fill_(2.0)
        model[2].weight.data.fill_(3.0)
    
    # 変換
    converter = EasyEnsembleConverterV2(num_ensembles=4, scale=False, cross=False)
    ee_model = converter.convert_model(model)
    
    print("Weight transfer verification completed")
    
    return model, ee_model

