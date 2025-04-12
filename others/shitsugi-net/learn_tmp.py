import torch

from prep import Prep
from model import Model
from trans import Trans

from torchvision.models import resnet18 as net



def merge_sd(state_dict0, state_dict1):
    state_dict_ave = {}
    for k in state_dict0.keys():
        state_dict_ave[k] = (state_dict0[k] + state_dict1[k]) / 2
        
    return state_dict_ave

tr = Trans(info={'mean': [0.5070751309394836, 0.48654884099960327, 0.44091784954071045], 'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]})

network_m = net(pretrained=True)

batch_size = 400        # バッチサイズ (並列して学習を実施する数)  
learning_rate = 0.0001   # 学習率 (重みをどの程度変更するか？)

dist = 100
times = 5




for j in range(2):
    for i in range(times):
        model_name = f"pt_{j}_{(i)}times" 
        
        epochs = dist*(times-i)
        network = net(pretrained=True)
        loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)

        pr = Prep(batch_size)
        model = Model(network, loss_func, optimizer)

        for e in range(epochs):
            if i != 0  and  e == 0:
                model.load_ckpt(f"pt_{j}_{(i)*dist}.ckpt")
                model.network.load_state_dict(merge_sd(model.network.to("cuda").state_dict(), network_m.to("cuda").state_dict()))
            model.train_1epoch(pr.train(tr.crop), mixup=True)
            model.val_1epoch(pr.val(tr.gen))

            model.logging()
            model.printlog(e, epochs, log_itv=1)
            
            if (e+1) == dist: model.save_ckpt(f"pt_{j}_{(i+1)*dist}.ckpt")
            
        model.save_ckpt(f"{model_name}.ckpt")
        model.hist_to_csv(f"{model_name}.csv")
            


