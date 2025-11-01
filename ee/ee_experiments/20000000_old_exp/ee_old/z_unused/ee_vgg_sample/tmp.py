from models import vgg

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

N = 2
ch = 3

# BL: original vgg model
m_bl = vgg.create_vgg8(6, ch, nb_fils=32, c_groups=1, n_groups=1, repeat_input=1, lmd=1)
# # BL: original vgg model (scaled up by fileter num)
# m_bl2 = vgg.create_vgg8(6, ch, nb_fils=16*2, c_groups=1, n_groups=1, repeat_input=1, lmd=1)
# # EE: EE-style vgg
# m_ee = vgg.create_vgg8(6, ch*N, nb_fils=16*N, c_groups=N, n_groups=N, repeat_input=1, lmd=1)
# # EE: EE-stype vgg (modality ensemble)
# m_ee_mod = vgg.create_vgg8(6, ch, nb_fils=16*N, c_groups=N, n_groups=N, repeat_input=1, lmd=1)

# print(m_bl)


import sys
from torchinfo import summary
batch_size_tmp = 100
flnm = "vgg_32.txt"
with open(flnm, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    summary(model=m_bl, input_size=(batch_size_tmp, 3, 100), verbose=1, col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"], row_settings=["var_names"])
    sys.stdout = original_stdout