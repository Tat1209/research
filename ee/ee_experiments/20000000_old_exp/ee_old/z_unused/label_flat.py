from pathlib import Path
import torch
from datasets import Datasets, dl
from trans import Trans



work_path = '/home/haselab/Documents/tat/Research/'
ds = Datasets(root=f"{work_path}assets/datasets/")

label_d, len_d = torch.load("/home/haselab/Documents/tat/Research/assets/datasets/caltech.ld")
# label_d, len_d = torch.load("/home/haselab/Documents/tat/Research/assets/datasets/cifar100_train.ld")

keys = list(len_d.keys())
values = list(len_d.values())

length = sum(values)
len_array = torch.empty(length, dtype=torch.int)

max_value = max(values)
index = 0
# len_arrayに格納されたkeyの個数をカウントするテンソルを初期化する
counter = torch.zeros(len(keys), dtype=torch.int)

for i in range(max_value):
    # len_dのkeyのうち、まだvalue個格納されていないものを選択する
    valid_keys = [key for key in keys if counter[key] < values[key]]
    
    m = len(valid_keys)
    if m > 0:
        # valid_keysをランダムにシャッフルする
        perm = torch.arange(m)
        # perm = torch.randperm(m)
        shuffled_keys = [valid_keys[p] for p in perm]
        
        # シャッフルされたkeyに対してループする
        for key in shuffled_keys:
            len_array[index] = key
            
            index += 1
            counter[key] += 1
            
            if index == length: break
        if index == length: break
        
    
print(len(len_array[101:201].unique()))

# len_arrayを出力する
with open("tmp.txt", "w") as f:
    f.write(str(len_array.tolist()))
    
    



