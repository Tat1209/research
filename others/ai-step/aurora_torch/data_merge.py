import os
import fnmatch
from datetime import datetime

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

folder = "/root/app/"
pattern = "val*_re.csv"

paths = []
for filename in os.listdir(folder): 
    if fnmatch.fnmatch(filename, pattern): paths.append(os.path.join(folder, filename))
paths = sorted(paths)
print(len(paths))

dfs = [pd.read_csv(path, header=None, names=["0", "1", "2", "3","label"]) for path in paths]

# 空のDataFrameを作る
# リストの要素ごとに処理する
df_m = pd.DataFrame(columns=["0", "1", "2", "3","label"])
for i, ds in enumerate(dfs):
    df_m = pd.concat([df_m, ds], axis=0)
    
df_m["label"] = df_m["label"].astype(int)

pd.DataFrame(df_m).to_csv(f'val_reg.csv', index=False, header=True)
    


