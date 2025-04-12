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

tr = Trans()
pr = Prep()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

model_name = "pt"


learning_rate = 0.0001  
network_m = net(pretrained=True)

network = net(pretrained=True)
loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定


model = Model(network, loss_func, device=device)


dist = 100
times = 5

for i in range(times):
    epochs = dist*(times-i)

    batch_size_cf = 500        
    network.fc = fc_cifar = torch.nn.Linear(network.fc.in_features, 100, device=device)
    optimizer_cf = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler_cf = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cf, T_max=epochs, eta_min=0, last_epoch=-1)

    batch_size_in = 500        
    network.fc = fc_in = torch.nn.Linear(network.fc.in_features, 1000, device=device)
    optimizer_in = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # scheduler_in = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_in, T_max=epochs, eta_min=0, last_epoch=-1)


    for e in range(epochs):
        if i != 0  and  e == 0:
            model.load_ckpt(f"{model_name}_{i*dist}.ckpt")
            print(f"{model_name}_{(i)*dist}.ckpt loaded.")
            model.network.load_state_dict(merge_sd(model.network.to("cuda").state_dict(), network_m.to("cuda").state_dict()))

            vLoss, vAcc = model.val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size_cf))
            ivLoss, ivAcc = model.val_1epoch(pr.dl("imagenet", tr.in_gen, batch_size_in, in_range=(0.2, 0.4)))

            log_dict = {"epoch":float(model.get_last_epoch() + 1) - 0.5, "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc, "iLoss":iLoss, "iAcc":iAcc, "ivLoss":ivLoss, "ivAcc":ivAcc}
            model.logging(log_dict)
            model.printlog(log_dict, e, epochs, log_itv=1)

        model.network.fc = fc_cifar
        model.optimizer = optimizer_cf
        for param in model.network.parameters(): param.requires_grad = True

        Loss, Acc = model.train_1epoch(pr.dl("cifar_train", tr.cf_crop, batch_size_cf), mixup=True)
        vLoss, vAcc = model.val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size_cf))


        if (e+1) == 1: iepochs = 30
        else: iepochs = 5

        for ie in range(iepochs):
            model.network.fc = fc_in
            model.optimizer = optimizer_in
            for param in model.network.parameters(): param.requires_grad = False
            for param in model.network.fc.parameters(): param.requires_grad = True

            iLoss, iAcc = model.train_1epoch(pr.dl("imagenet", tr.in_gen, batch_size_in, in_range=(0, 0.2)), mixup=True)
            ivLoss, ivAcc = model.val_1epoch(pr.dl("imagenet", tr.in_gen, batch_size_in, in_range=(0.2, 0.4)))


        log_dict = {"epoch":float(int(model.get_last_epoch() + 1)), "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc, "iLoss":iLoss, "iAcc":iAcc, "ivLoss":ivLoss, "ivAcc":ivAcc}
        model.logging(log_dict)
        model.printlog(log_dict, e, epochs, log_itv=1)

        if (e+1) == dist: model.save_ckpt(f"{model_name}_{int(model.get_last_epoch())}.ckpt")

    model.save_ckpt(f"{model_name}_{i}times.ckpt")
    model.hist_to_csv(f"{model_name}_{i}times.csv")


