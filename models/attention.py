import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import conv1x1, conv3x3, BasicBlock, Bottleneck


def transpose(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def downsample(in_channels, out_channels):
    return nn.Sequential(
        conv1x1(in_channels, out_channels),
        nn.BatchNorm2d(num_features(out_channels)),
        nn.ReLU(inplace=True),
    )


class Attention0(nn.Module):
    def __init__(self, channels, block):
        super().__init__()
        self._trunk1 = block(channels, channels)
        self._trunk2 = block(channels, channels)

        self._enc = block(channels, channels)
        self._dec = block(channels, channels)

        self._conv1x1 = nn.Sequential(
            conv1x1(2 * channels, channels),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
        )

        self._mp = nn.MaxPool2d(3, 2, 1)
        self._relu = nn.ReLU(inplace=True)

    def enc(self, x):
        return self._enc(x)

    def dec(self, x):
        return self._dec(x)

    def trunking(self, x):
        return self._trunk2(self._trunk1(x))

    def masking(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return torch.sigmoid(x)

    def forward(self, x):
        trunk = self.trunking(x)
        mask = self.masking(x)
        return (1 + mask) * trunk


class Attention1(nn.Module):
    def __init__(self, channels, block):
        super().__init__()
        self._trunk1 = block(channels, channels)
        self._trunk2 = block(channels, channels)

        self._enc1 = block(channels, channels)
        self._enc2 = block(channels, channels)

        self._dec = block(channels, channels)
        self._conv1x1 = nn.Sequential(
            conv1x1(2 * channels, channels),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
        )

        self._trans = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
        )

        self._mp = nn.MaxPool2d(3, 2, 1)
        self._relu = nn.ReLU(inplace=True)

    def enc(self, x):
        x1 = self._enc1(x)
        x2 = self._enc2(self._mp(x1))
        return [x1, x2]

    def dec(self, x):
        x1, x2 = x
        x2 = self._trans(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self._conv1x1(x)
        return self._dec(x)

    def trunking(self, x):
        return self._trunk2(self._trunk1(x))

    def masking(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return torch.sigmoid(x)

    def forward(self, x):
        trunk = self.trunking(x)
        mask = self.masking(x)
        return (1 + mask) * trunk


class Attention2(nn.Module):
    def __init__(self, channels, block):
        super().__init__()
        self._trunk1 = block(channels, channels)
        self._trunk2 = block(channels, channels)

        self._enc1 = block(channels, channels)
        self._enc2 = block(channels, channels)
        self._enc3 = nn.Sequential(block(channels, channels), block(channels, channels))

        self._dec1 = nn.Sequential(
            conv1x1(2 * channels, channels),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            block(channels, channels),
        )
        self._dec2 = nn.Sequential(
            conv1x1(2 * channels, channels),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            block(channels, channels),
        )

        self._trans = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
        )

        self._mp = nn.MaxPool2d(3, 2, 1)
        self._relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # ''' try to open this line and see the change of acc
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
        # '''

    def enc(self, x):
        x1 = self._enc1(x)
        x2 = self._enc2(self._mp(x1))
        x3 = self._enc3(self._mp(x2))
        return [x1, x2, x3]

    def dec(self, x):
        x1, x2, x3 = x

        x2 = torch.cat([x2, self._trans(x3)], dim=1)
        x2 = self._dec1(x2)

        x3 = torch.cat([x1, self._trans(x2)], dim=1)
        x3 = self._dec1(x3)

        return x3

    def trunking(self, x):
        return self._trunk2(self._trunk1(x))

    def masking(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return torch.sigmoid(x)

    def forward(self, x):
        trunk = self.trunking(x)
        mask = self.masking(x)
        return (1 + mask) * trunk


def attention(channels, block=BasicBlock, depth=-1):
    if depth == 0:
        return Attention0(channels, block)
    elif depth == 1:
        return Attention1(channels, block)
    elif depth == 2:
        return Attention2(channels, block)
    else:
        traceback.print_exc()
        raise Exception("depth must be specified")
