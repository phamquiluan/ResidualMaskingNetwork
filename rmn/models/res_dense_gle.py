import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import load_state_dict_from_url
from .resnet import resnet18
from .densenet import densenet121
from .googlenet import googlenet


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


class ResDenseGle(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(ResDenseGle, self).__init__()

        self.resnet = resnet18(in_channels, num_classes)
        # self.densenet = densenet121(in_channels, num_classes, pretrained=False)
        self.densenet = densenet121(in_channels, num_classes)
        # self.googlenet = googlenet(in_channels, num_classes, pretrained=False)
        self.googlenet = googlenet(in_channels, num_classes)

        # change fc to identity
        self.resnet.fc = nn.Identity()
        # self.densenet.fc = nn.Identity()
        self.densenet.classifier = nn.Identity()
        self.googlenet.fc = nn.Identity()

        # create new fc
        # self.fc = nn.Linear(512 * 3, 7)
        # avoid change fc inside trainer
        self._fc = nn.Linear(2536, 7)
        # self._fc = nn.Linear(2560, 7)

        # another options for fc
        self.fc1 = nn.Linear(2536, 512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x1 = self.resnet(x)
        x2 = self.densenet(x)
        x3 = self.googlenet(x)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self._fc(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        return x


def rdg(pretrained=False, progress=True, **kwargs):
    model = ResDenseGle(kwargs["in_channels"], kwargs["num_classes"])

    return model
