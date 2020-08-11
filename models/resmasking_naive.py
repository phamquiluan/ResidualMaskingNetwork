import copy
import torch
import torch.nn as nn

from .utils import load_state_dict_from_url
from .resnet import BasicBlock, Bottleneck, ResNet, resnet18


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


from .masking import masking


class ResMaskingNaive(ResNet):
    def __init__(self, weight_path):
        super(ResMaskingNaive, self).__init__(
            block=BasicBlock, layers=[3, 4, 6, 3], in_channels=3, num_classes=1000
        )
        # state_dict = torch.load('saved/checkpoints/resnet18_rot30_2019Nov05_17.44')['net']
        state_dict = load_state_dict_from_url(model_urls["resnet34"], progress=True)
        self.load_state_dict(state_dict)

        self.fc = nn.Linear(512, 7)

        """
        # freeze all net
        for m in self.parameters():
            m.requires_grad = False
        """

        self.mask1 = masking(64, 64, depth=4)
        self.mask2 = masking(128, 128, depth=3)
        self.mask3 = masking(256, 256, depth=2)
        self.mask4 = masking(512, 512, depth=1)

    def forward(self, x):  # 224
        x = self.conv1(x)  # 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56

        x = self.layer1(x)  # 56
        # m = self.mask1(x)
        # x = x * m

        x = self.layer2(x)  # 28
        # m = self.mask2(x)
        # x = x * m

        x = self.layer3(x)  # 14
        # m = self.mask3(x)
        # x = x * m

        x = self.layer4(x)  # 7
        # m = self.mask4(x)
        # x = x * m

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def resmasking_naive_dropout1(in_channels=3, num_classes=7, weight_path=""):
    model = ResMaskingNaive(weight_path)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, 7)
        # nn.Linear(512, num_classes)
    )
    return model
