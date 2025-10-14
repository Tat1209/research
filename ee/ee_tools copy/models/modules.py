from typing import Literal

import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'{self.__class__.__name__}{self.shape}'

    def forward(self, input):
        batch_size = input.size(0)
        target = (batch_size, *self.shape)
        out = input.view(target)
        return out


class CopyConcat(nn.Module):
    def __init__(self, n, dim):
        super().__init__()
        # dimは、(C, H, W) に対する処理を想定
        self.n = n
        self.dim = dim

    def __repr__(self):
        return f'{self.__class__.__name__}({self.n}, {self.dim})'

    def forward(self, input):
        # inputにはバッチ(B, C, H, W)が入る。バッチ次元を考慮するため、dimに1たしてる
        eff_dim = self.dim + 1
        out = torch.cat([input for _ in range(self.n)], dim=eff_dim)
        # out = torch.cat([input.clone() for _ in range(self.n)], dim=eff_dim) # for safe
        return out


class SplitMerge(nn.Module):
    def __init__(
        self,
        chunks: int,
        dim: int,
        mode: Literal["mean", "sum", "none", "both"] = "mean"
    ):
        super().__init__()
        assert chunks > 0, f'chunks must be positive int, got {chunks}'
        self.chunks = chunks
        self.dim = dim
        self.mode = mode

    def __repr__(self):
        return f'{self.__class__.__name__}({self.chunks}, mode="{self.mode}")'

    def forward(self, input: torch.Tensor):
        eff_dim = self.dim + 1
        chunks = torch.chunk(input, self.chunks, dim=eff_dim)
        merged = torch.stack(chunks, dim=eff_dim)

        if self.mode == "sum":
            return merged.sum(dim=eff_dim)
        elif self.mode == "mean":
            return merged.mean(dim=eff_dim)
        elif self.mode == "both":
            return merged.mean(dim=eff_dim), chunks
        elif self.mode == "none":
            return chunks
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")