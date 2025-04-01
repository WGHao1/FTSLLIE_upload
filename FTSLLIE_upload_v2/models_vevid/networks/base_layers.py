import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.doubleconv = nn.Sequential(
            Conv2D(in_channels, out_channels, activation),
            Conv2D(out_channels,out_channels, activation)
        )

    def forward(self, x):
        return self.doubleconv(x)


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu', stride=1):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.conv_relu(x)

class MaxPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)

class Concat(nn.Module):
    def forward(self, x, y):
        _, _, xh, xw = x.size()
        _, _, yh, yw = y.size()
        diffY = xh - yh
        diffX = xw - yw
        y = F.pad(y, (diffX // 2, diffX - diffX//2, 
                      diffY // 2, diffY - diffY//2))
        return torch.cat((x, y), dim=1)