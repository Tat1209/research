from prep import Prep
import post
import torch

data_dir = "aurora/competition01_gray_128x128/"
data_path = {"labeled":data_dir+"train_val", "unlabeled":data_dir+"test"}

batch_size = 120        # バッチサイズ (並列して学習を実施する数)  

pr = Prep(data_path, batch_size, train_ratio=1, color=True)

model = torch.load("competition_model_0421_075244.pth")

result = model.pred_tta(pr, times=30, aug_ratio=0.8, categorize=True)

post.postprocess(pr, result, None, None)


