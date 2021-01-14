"""no need residual :)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        _filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]
        _filters = [16, 32, 64, 128, 256]

        self.conv1 = block(in_channels, filters[0])
        self.conv2 = block(filters[0], filters[1])
        self.conv3 = block(filters[1], filters[2])
        self.conv4 = block(filters[2], filters[3])
        self.conv5 = block(filters[3], filters[4])
        self.down_pooling = nn.MaxPool2d(2)

        self.up_pool6 = up_pooling(filters[4], filters[3])
        self.conv6 = block(filters[4], filters[3])
        self.up_pool7 = up_pooling(filters[3], filters[2])
        self.conv7 = block(filters[3], filters[2])
        self.up_pool8 = up_pooling(filters[2], filters[1])
        self.conv8 = block(filters[2], filters[1])
        self.up_pool9 = up_pooling(filters[1], filters[0])
        self.conv9 = block(filters[1], filters[0])

        self.conv10 = nn.Conv2d(filters[0], num_classes, 1)

        # default xavier init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        # go up
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = torch.softmax(output, dim=1)
        return output


def basic_unet(in_channels, num_classes):
    return Unet(in_channels, num_classes)
