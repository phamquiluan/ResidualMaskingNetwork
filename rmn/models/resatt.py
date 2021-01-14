import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import load_state_dict_from_url
from .attention import attention
from .resnet import BasicBlock, Bottleneck, ResNet, resnet18


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


class ResAtt(ResNet):

    # def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
    #              groups=1, width_per_group=64, replace_stride_with_dilation=None,
    #              norm_layer=None, in_channels=3):
    def __init__(self):
        super(ResAtt, self).__init__(
            block=BasicBlock, layers=[2, 2, 2, 2], in_channels=3, num_classes=1000
        )
        # state_dict = load_state_dict_from_url(model_urls['resnet18'])
        # self.load_state_dict(state_dict)

        self.att12 = attention(channels=64, block=BasicBlock, depth=2)
        self.att23 = attention(channels=128, block=BasicBlock, depth=1)
        self.att34 = attention(channels=256, block=BasicBlock, depth=0)
        # self.fc = nn.Linear(512, 7)

        # self.init_att()
        # self.init_mask()

    def init_att(self):
        self.att12._trunk1 = copy.deepcopy(self.layer1[1])
        self.att12._trunk2 = copy.deepcopy(self.layer1[1])

        self.att23._trunk1 = copy.deepcopy(self.layer2[1])
        self.att23._trunk2 = copy.deepcopy(self.layer2[1])

        self.att34._trunk1 = copy.deepcopy(self.layer3[1])
        self.att34._trunk2 = copy.deepcopy(self.layer3[1])

    def init_mask(self):
        self.att12._enc = copy.deepcopy(self.layer1[1])
        self.att12._dec = copy.deepcopy(self.layer1[1])

        self.att23._enc1 = copy.deepcopy(self.layer2[1])
        self.att23._enc2 = copy.deepcopy(self.layer2[1])
        self.att23._dec = copy.deepcopy(self.layer2[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.att12(x)
        x = self.layer2(x)
        x = self.att23(x)
        x = self.layer3(x)
        x = self.att34(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def resatt18(pretrained=True, progress=True, **kwargs):
    model = ResAtt()
    model.fc = nn.Linear(512, 7)
    return model
