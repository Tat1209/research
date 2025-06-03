import torch

def forward(input, chunks, dim, use_sum):
    eff_dim = dim + 1
    chunks = torch.chunk(input, chunks, dim=eff_dim)
    merged = torch.stack(chunks, dim=eff_dim)
    if use_sum:
        out = merged.sum(dim=eff_dim)
    else:
        out = merged.mean(dim=eff_dim)
    return out


def forward2(input, chunks, dim, use_sum):
    # inputにはバッチ(B, C, H, W)が入る。バッチ次元を考慮するため、dimに1たしてる
    eff_dim = dim + 1
    x = input.view(input.shape[0], chunks, -1)
    
    if use_sum:
        x = torch.sum(x, dim=eff_dim)
    else:
        x = torch.mean(x, dim=eff_dim)
    return x

# tsr = torch.arange(12, dtype=float).view(2, 6)
tsr = torch.arange(48, dtype=float).view(2, 6, 2, 2)
tsr = forward(tsr, 3, 0, False)
print(tsr)

tsr = torch.arange(48, dtype=float).view(2, 6, 2, 2)
tsr = forward2(tsr, 3, 0, False)
print(tsr)

# tsr = torch.arange(12, dtype=float).view(2, 6)
# print(tsr)