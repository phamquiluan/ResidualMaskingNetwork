import torch
import torch.nn

from .utils import load_state_dict_from_url
from .resnet import ResNet, BasicBlock

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
}


class ResNet112(ResNet):
    def __init__(self, block, layers):
        super(ResNet112, self).__init__(
            block=block, layers=layers, in_channels=3, num_classes=1000
        )

        # state_dict = load_state_dict_from_url(model_urls['resnet18'])
        # self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def resnet18x112(pretrained=True, progress=True, **kwargs):
    model = ResNet112(block=BasicBlock, layers=[2, 2, 2, 2])
    state_dict = load_state_dict_from_url(model_urls["resnet18"])
    model.load_state_dict(state_dict)
    return model


def resnet34x112(pretrained=True, progress=True, **kwargs):
    model = ResNet112(block=BasicBlock, layers=[3, 4, 6, 3])
    state_dict = load_state_dict_from_url(model_urls["resnet34"])
    model.load_state_dict(state_dict)
    return model
