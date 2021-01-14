import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualUnit, self).__init__()
        width = int(out_channels / 4)

        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = conv3x3(width, width)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = conv1x1(width, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # for downsample
        self._downsample = nn.Sequential(
            conv1x1(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self._downsample(identity)
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        pass

    def forward(self, x):
        pass


class BaseNet(nn.Module):
    """basenet for fer2013"""

    def __init__(self, in_channels=1, num_classes=7):
        super(BaseNet, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual_1 = ResidualUnit(in_channels=64, out_channels=256)
        self.residual_2 = ResidualUnit(in_channels=256, out_channels=512)
        self.residual_3 = ResidualUnit(in_channels=512, out_channels=1024)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.residual_1(x)
        x = self.residual_2(x)
        x = self.residual_3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def basenet(in_channels=1, num_classes=7):
    return BaseNet(in_channels, num_classes)


if __name__ == "__main__":
    net = BaseNet().cuda()
    from torchsummary import summary

    print(summary(net, input_size=(1, 48, 48)))
