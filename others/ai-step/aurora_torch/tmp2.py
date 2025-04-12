import pandas as pd

rand_idxs = list(range(self.data_num))
random.seed(seed)
random.shuffle(self.rand_idxs)

df = pd.read_csv("/root/app/val_effs_15.csv")
df["label"] = df["label"].astype(int)
print(df)
print(df.dtypes)
df.to_csv('/root/app/val_effs_15.csv', index=False)