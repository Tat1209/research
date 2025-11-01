from datasets import Datasets, dl
from trans import Trans


work_path = '/home/haselab/Documents/tat/Research/'
ds = Datasets(root=f"{work_path}assets/datasets/")

# loader = dl(ds("fix_rand", [Trans.tsr], seed=5, in_range=(0, 0.001)), batch_size=5, shuffle=False)
loader = dl(ds("fix_rand", [Trans.tsr], seed=2, in_range=(0, 0.001)), batch_size=10, shuffle=True)


# loader = iter(loader)
# n = 5
# for _ in range(n):
#     try: input_b, label_b = next(loader)
#     except: break
#     print(label_b)

# loader = iter(loader)
# n = 5
# for _ in range(n):
#     try: input_b, label_b = next(loader)
#     except: break
#     print(label_b)


for i in range(4):
    for input_b, label_b in loader:
        print(label_b)
    



