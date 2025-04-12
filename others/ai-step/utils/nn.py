import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64*21*21, 4)
        )
        

    def forward(self, x):
        x = self.net(x)
        return x
    


# """
            # nn.AdaptiveAvgPool2d(5),
            # # shape = (N, 64, 5, 5)
            # nn.Flatten(),
            # # shape = (N, 64 * 5 * 5)
            # nn.Linear(64 * 5 * 5, 4),
# """