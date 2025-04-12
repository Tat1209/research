import os

# ディレクトリのパスを指定
dir_path = "/home/haselab/Documents/tat/Research/app/ee2/exp_tmp copy/runs"

# ディレクトリ内のすべてのファイル/ディレクトリを取得
dir_names = os.listdir(dir_path)

# すべてのディレクトリの語尾に "_" をつける
for name in dir_names:
    old_name = os.path.join(dir_path, name)
    new_name = os.path.join(dir_path, name + "_")
    os.rename(old_name, new_name)

# "_" をつけたディレクトリ名を 80, 81, 82... に変更する
for i, name in enumerate(sorted(os.listdir(dir_path)), start=80):
    old_name = os.path.join(dir_path, name)
    new_name = os.path.join(dir_path, str(i))
    os.rename(old_name, new_name)
