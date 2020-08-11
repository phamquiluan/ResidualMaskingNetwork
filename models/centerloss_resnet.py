import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import load_state_dict_from_url

from .resnet import ResNet, BasicBlock


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
}


class ResNetCenterLoss(ResNet):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2]):
        super(ResNetCenterLoss, self).__init__(
            block=BasicBlock, layers=layers, in_channels=3, num_classes=1000
        )
        state_dict = load_state_dict_from_url(model_urls["resnet18"])
        self.load_state_dict(state_dict)

        # for center loss
        self.center_loss_fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        features = self.relu(self.center_loss_fc(x))
        outputs = self.fc(x)
        # return outputs, features
        return outputs


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18_centerloss(pretrained=True, progress=True, **kwargs):
    model = ResNetCenterLoss()
    model.fc = nn.Linear(512, 7)
    return model
