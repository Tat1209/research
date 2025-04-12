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


def merge_sd_wofc(state_dict0, state_dict1):
    state_dict_ave = {}
    for k in state_dict0.keys():
        if not k.startswith('fc'): state_dict_ave[k] = (state_dict0[k] + state_dict1[k]) / 2
        else: state_dict_ave[k] = state_dict0[k]
    return state_dict_ave
        

tr = Trans()
pr = Prep()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

model_name = "pt"


network_m = net(pretrained=True)

learning_rate = 0.0001  
loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定

# dist = 2
dist = 4
times = 5



for i in range(times):
    epochs = dist*(times-i)

    network_cf = net(pretrained=True)
    batch_size_cf = 500        
    network_cf.fc = torch.nn.Linear(network_cf.fc.in_features, 100, device=device)
    optimizer_cf = torch.optim.Adam(network_cf.parameters(), lr=learning_rate)
    scheduler_cf = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cf, T_max=epochs, eta_min=0, last_epoch=-1)
    model_cf = Model(network_cf, loss_func, device=device, optimizer=optimizer_cf)

    network_in = net(pretrained=True)
    batch_size_in = 500        
    # network_in.fc = torch.nn.Linear(network_in.fc.in_features, 1000, device=device)
    optimizer_in = torch.optim.Adam(network_in.parameters(), lr=learning_rate)
# scheduler_in = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_in, T_max=epochs, eta_min=0, last_epoch=-1)
    model_in = Model(network_in, loss_func, device=device, optimizer=optimizer_in)


    def copy_cf_in():
            new_state_dict = {}
            for k in model_in.network.state_dict().keys():
                if k.startswith('fc'): new_state_dict[k] = model_in.network.state_dict()[k]
                else: new_state_dict[k] = model_cf.network.state_dict()[k]
            model_in.network.load_state_dict(new_state_dict)


    for e in range(epochs):
        if i != 0  and  e == 0:
            model_cf.load_ckpt(f"{model_name}_{i*dist}.ckpt")
            print(f"{model_name}_{(i)*dist}.ckpt loaded.")
            for param in model_in.network.parameters(): param.requires_grad = True

            model_cf.network.load_state_dict(merge_sd_wofc(model_cf.network.to("cuda").state_dict(), network_m.to("cuda").state_dict()))
            copy_cf_in()


            # vLoss, vAcc = model_cf.val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size_cf))
            # ivLoss, ivAcc = model_in.val_1epoch(pr.dl("imagenet", tr.in_gen, batch_size_in, in_range=(0.8, 1)))
            vLoss, vAcc = model_cf.val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size_cf, in_range=(0.0001, 0.0002)))
            ivLoss, ivAcc = model_in.val_1epoch(pr.dl("imagenet", tr.in_gen, batch_size_in, in_range=(0.0001, 0.0002)))

            log_dict = {"epoch":float(model_cf.get_last_epoch() + 1) - 0.5, "vLoss":vLoss, "vAcc":vAcc, "ivLoss":ivLoss, "ivAcc":ivAcc}
            model_cf.logging(log_dict)
            model_cf.printlog(log_dict, e, epochs, log_itv=1)


        # Loss, Acc = model_cf.train_1epoch(pr.dl("cifar_train", tr.cf_crop, batch_size_cf), mixup=True)
        # vLoss, vAcc = model_cf.val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size_cf))
        Loss, Acc = model_cf.train_1epoch(pr.dl("cifar_train", tr.cf_crop, batch_size_cf, in_range=(0, 0.0001)), mixup=True)
        vLoss, vAcc = model_cf.val_1epoch(pr.dl("cifar_val", tr.cf_gen, batch_size_cf, in_range=(0, 0.0001)))


        if (e+1) == 1: iepochs = 6
        else: iepochs = 2

        copy_cf_in()
        # print(model_cf.network.state_dict()["layer4.1.bn2.weight"][0:5], model_cf.network.state_dict()["fc.weight"][0, 0:5])

        for param in model_in.network.parameters(): param.requires_grad = False
        for param in model_in.network.fc.parameters(): param.requires_grad = True

        for ie in range(iepochs):

            # iLoss, iAcc = model_in.train_1epoch(pr.dl("imagenet", tr.in_gen, batch_size_in, in_range=(0, 0.5)), mixup=True)
            # ivLoss, ivAcc = model_in.val_1epoch(pr.dl("imagenet", tr.in_gen, batch_size_in, in_range=(0.8, 1)))
            iLoss, iAcc = model_in.train_1epoch(pr.dl("imagenet", tr.in_gen, batch_size_in, in_range=(0, 0.0001)), mixup=True)
            ivLoss, ivAcc = model_in.val_1epoch(pr.dl("imagenet", tr.in_gen, batch_size_in, in_range=(0.0001, 0.0002)))

            # log_dict = {"epoch":float(int(model_in.get_last_epoch() + 1)), "iLoss":iLoss, "iAcc":iAcc, "ivLoss":ivLoss, "ivAcc":ivAcc}

            # model_in.logging(log_dict)
            # model_in.printlog(log_dict, ie, iepochs, log_itv=1)



        log_dict = {"epoch":float(int(model_cf.get_last_epoch() + 1)), "Loss":Loss, "Acc":Acc, "vLoss":vLoss, "vAcc":vAcc, "iLoss":iLoss, "iAcc":iAcc, "ivLoss":ivLoss, "ivAcc":ivAcc}

        model_cf.logging(log_dict)
        model_cf.printlog(log_dict, e, epochs, log_itv=1)

        if (e+1) == dist: model_cf.save_ckpt(f"{model_name}_{int(model_cf.get_last_epoch())}.ckpt")

    model_cf.save_ckpt(f"{model_name}_{i}times.ckpt")
    model_cf.hist_to_csv(f"{model_name}_{i}times.csv")


