import os
import fnmatch
from datetime import datetime

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

path = "/root/app/test_effs_15_n.csv"
df = pd.read_csv(path, header=None, names=["path", "tensor"])

# リストの要素ごとに処理する
df_classed = pd.DataFrame(columns=["0", "1", "2", "3", "path"])
for s in df["tensor"]:
    # 文字列をリストに変換する
    s = s.strip("[]").split()
    # リストの要素を浮動小数点に変換する
    s = [float(x) for x in s]
    # リストをDataFrameに変換して、元のDataFrameに追加する
    df_classed = df_classed._append(pd.DataFrame([s], columns=['0', '1', '2', '3']), ignore_index=True)
    df_classed["path"] = df["path"]
    
df_classed = df_classed.sort_values(by='path')
df_classed = df_classed.reset_index(drop=True)
print(df_classed)

# ft = datetime.now().strftime("%m%d_%H%M%S")
pd.DataFrame(df_classed).to_csv(f'/root/app/test_effs_15.csv', index=False, header=False)
    


