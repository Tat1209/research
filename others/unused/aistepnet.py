import torch
import torch.nn as nn
import torch.nn.functional as F


# モデルの構築
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        # 畳み込み - 活性化関数 (ReLU) - プーリング
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# この「model」という変数に，構築するモデルのすべての情報が入ります
