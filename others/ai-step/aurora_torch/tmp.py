import pandas as pd

df = pd.read_csv("/root/app/aurora_torch/aaaaaa.csv", header=None)
# df["label"] = df["label"].astype(int)

# df = df.sort_values(by=0)
# df = df.reset_index(drop=True)
df[0] = df[0].str[59:]
print(df)

df.to_csv('label.csv', index=False)