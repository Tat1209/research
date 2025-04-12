import torch

from prep import Prep
from model import Model
from trans import Trans

from torchvision.models import resnet18 as net

for i in range(5):
    model_name = f"cal_xe_{i}times"

    tr = Trans(info={'mean': [0.5070751309394836, 0.48654884099960327, 0.44091784954071045], 'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]})

    network = net()
    Model.load_ckpt_network(network, f"xe_{i}times.ckpt")
    network.fc = torch.nn.Linear(network.fc.in_features, 101)

    for param in network.parameters(): param.requires_grad = False
    for param in network.fc.parameters(): param.requires_grad = True

    batch_size = 400
    learning_rate = 0.0008   # 学習率 (重みをどの程度変更するか？)
    loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    pr = Prep(batch_size, val_range=(0.6, 1.0), seed=0)
    model = Model(network, loss_func, optimizer)


    epochs = 200
    for e in range(epochs):
        model.train_1epoch(pr.cal_train(tr.calgen))
        model.val_1epoch(pr.cal_val(tr.calgen))

        # print(model.network.state_dict()["fc.weight"][0, 0:5])
        # print(model.network.state_dict()["layer4.1.bn2.weight"][0:5])
        
        model.logging()
        model.printlog(e, epochs, log_itv=5)

    model.save_ckpt(f"{model_name}.ckpt")
    model.hist_to_csv(f"{model_name}.csv")

