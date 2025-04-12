import numpy as np

from prep import Prep
from model import Model
import post
from ens import Ens

from torchvision.models import efficientnet_v2_s as net0
# from torchvision.models import efficientnet_v2_m as net1
# from torchvision.models.regnet import regnet_y_3_2gf as net2
# from torchvision.models.regnet import regnet_y_800mf as net3
# from torchvision.models.regnet import regnet_y_400mf as net0

data_dir = "/root/app/competition01_gray_128x128/"
data_path = {"labeled":data_dir+"train_val", "unlabeled":data_dir+"test"}

batch_size = 120        # バッチサイズ (並列して学習を実施する数)  
epochs = 5000              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)

num_models = 10

model_list = []
for i in range(num_models):
    pr = Prep(data_path, batch_size, val_range=(i/num_models, (i+1)/num_models), seed=0)
    model = Model(pr, net0(num_classes=4), epochs, learning_rate, log_itv=100, fit_aug_ratio=1.0, mixup_alpha=0.2, pred_times=24, tta_aug_ratio=0.75)
    model_list.append(model)

ens = Ens()
mod_res = ens.cross(model_list, categorize=True)
print(mod_res["vAcc"])

val_log = np.concatenate((mod_res["val_feat"], mod_res["val_label"][:, None]), axis=1)
np.savetxt('val.csv', val_log, delimiter=',')


        


