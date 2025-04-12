import torch

from prep import Prep
from model import Model
from trans import Trans

from torchvision.models import resnet18 as net


tr = Trans(info={'mean': [0.5070751309394836, 0.48654884099960327, 0.44091784954071045], 'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]})

batch_size = 250        # バッチサイズ (並列して学習を実施する数)  
epochs = 1000              # エポック数 (学習を何回実施するか？という変数)
learning_rate = 0.001   # 学習率 (重みをどの程度変更するか？)

network0 = net()
network0.load_state_dict(torch.load('model0_500.ckpt')["network_sd"])
state_dict0 = network0.state_dict()

network1 = net()
network1.load_state_dict(torch.load('model1_500.ckpt')["network_sd"])
state_dict1 = network1.state_dict()



network = net()

state_dict_ave = {}
for k in state_dict0.keys():
    state_dict_ave[k] = (state_dict0[k] + state_dict1[k]) / 2
    
    

learning_rate = learning_rate
loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
optimizer = torch.optim.RAdam(network.parameters(), lr=learning_rate)    

pr = Prep(batch_size, val_range=(0.9, 1.00), seed=0)
model = Model(pr, network, learning_rate, loss_func, optimizer)
model.load_ckpt("model0_500.ckpt")

model.network.load_state_dict(state_dict_ave)


for e in range(epochs):
    model.train_1epoch(tr.aug, mixup=True)
    model.val_1epoch(tr.gen)

    model.logging()
    model.printlog(e, epochs, log_itv=5)
    
model.hist_to_csv("modelC_500.csv")

