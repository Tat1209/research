import os
import fnmatch
from datetime import datetime

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

folder = "/root/app/"
pattern = "test*_re.csv"

paths = []
for filename in os.listdir(folder): 
    if fnmatch.fnmatch(filename, pattern): paths.append(os.path.join(folder, filename))
paths = sorted(paths)
print(len(paths))

dfs = [pd.read_csv(path, header=None, names=["0", "1", "2", "3"]) for path in paths]

# 空のDataFrameを作る
# リストの要素ごとに処理する
for i, ds in enumerate(dfs):
    if i == 0: df_m = ds
    else: df_m = df_m + ds
df_m = df_m / len(paths)
    
df_m["label"] = pd.read_csv("/root/app/label.csv")["label"]
df_m = df_m.sort_values(by='label')

pd.DataFrame(df_m).to_csv(f'test_reg.csv', index=False, header=True)
    


