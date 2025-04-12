from prep import Prep
from model import Model
import post
import ens

# from nn import NeuralNet as net
from torchvision.models.regnet import regnet_y_400mf as net

data_path = {"labeled":"/root/app/test_effs_15.csv", "unlabeled":data_dir+"test"}

batch_size = 120        # バッチサイズ (並列して学習を実施する数)  
epochs = 8000              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)


pr = Prep(data_path, batch_size, val_range=(0.9, 1.00))
network = net(num_classes=4)
model = Model(pr, network, epochs, learning_rate, log_itv=100, fit_aug_ratio=1.0, mixup_alpha=0.2, pred_times=25, tta_aug_ratio=0.75)

hist = model.fit()
# summary = model.pred(categorize=False, val=True)
result = model.pred(categorize=False)

post.postprocess(result["outputs"], hist, model)




