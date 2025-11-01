import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_shape, force=False):
        super(AutoEncoder, self).__init__()
        self.force = force
        self.input_shape = input_shape

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),  # 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),  # 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),  # 8x8
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),  # 4x4
            nn.Conv2d(256, 8, kernel_size=1, padding=0),  # Embedding layer (4x4x8)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 128, kernel_size=2, stride=2, padding=0),  # 8x8
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2, padding=0),  # 16x16
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2, padding=0),  # 32x32
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0),  # 64x64
            nn.BatchNorm2d(512),
            nn.Conv2d(512, input_shape[0], kernel_size=1, padding=0),  # Output layer
        )

    def forward(self, x):
        embd = self.encoder(x)
        x = self.decoder(embd)

        if x.shape[1:] != self.input_shape and not self.force:
            # Adjust the output size if it's different from the input size
            padding = (2, 2, 2, 2)  # (padding_left, padding_right, padding_top, padding_bottom)
            x = F.pad(x, padding, mode='constant', value=0)
        
        return x

# # Example usage:
# input_shape = (3, 64, 64)  # Example input shape [channels, height, width]
# model = AutoEncoder(input_shape)