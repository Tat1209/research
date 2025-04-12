import torch

from prep import Prep
from model import Model
from trans import Trans

from torchvision.models import resnet18 as net


for j in range(2):
    for i in range(5):
        model_name = f"in_pt_{j}_{i}times"

        tr = Trans(info={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})

        network = net()
        Model.load_ckpt_network(network, f"/root/app/shitsugi_first/pt_{j}_{i}times.ckpt")
        network.fc = torch.nn.Linear(network.fc.in_features, 1000)

        for param in network.parameters(): param.requires_grad = False
        for param in network.fc.parameters(): param.requires_grad = True

        batch_size = 40
        learning_rate = 0.0008   # 学習率 (重みをどの程度変更するか？)
        loss_func = torch.nn.CrossEntropyLoss()  # 損失関数の設定
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        pr = Prep(batch_size, val_range=(0.6, 1.0), seed=0)
        model = Model(network, loss_func, optimizer)


        epochs = 100
        for e in range(epochs):
            model.train_1epoch(pr.in_train(tr.ingen))
            model.val_1epoch(pr.in_val(tr.ingen))

            model.logging()
            model.printlog(e, epochs, log_itv=5)

        model.save_ckpt(f"{model_name}.ckpt")
        model.hist_to_csv(f"{model_name}.csv")

