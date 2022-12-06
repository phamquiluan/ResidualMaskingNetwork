import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model

from .alexnet import AlexNet, alexnet
from .brain_humor import (
    DeepResUNet,
    DoubleConv,
    DownBlock,
    ONet,
    PreActivateDoubleConv,
    PreActivateResBlock,
    PreActivateResUpBlock,
    ResBlock,
    ResUNet,
    UNet,
    UpBlock,
    deepresunet,
)
from .centerloss_resnet import resnet18_centerloss
from .densenet import DenseNet, densenet121, densenet161, densenet169, densenet201
from .fer2013_models import BaseNet, BasicBlock, ResidualUnit, basenet, conv1x1, conv3x3
from .googlenet import GoogLeNet, googlenet
from .inception import Inception3, inception_v3
from .inception_resnet_v1 import (
    BasicConv2d,
    Block8,
    Block17,
    Block35,
    InceptionResnetV1,
    Mixed_6a,
    Mixed_7a,
    get_torch_home,
    inception_resnet_v1,
    load_weights,
)
from .masking import masking
from .res_dense_gle import ResDenseGle, rdg
from .resatt import ResAtt, resatt18
from .residual_attention_network import ResidualAttentionModel, res_attention
from .resmasking import (
    resmasking,
    resmasking50_dropout1,
    resmasking_dropout1,
    resmasking_dropout2,
)
from .resmasking_naive import resmasking_naive_dropout1
from .resnet import (
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
)
from .resnet50_scratch_dims_2048 import resnet50_pretrained_vgg
from .resnet112 import resnet18x112
from .runet import (
    Attention_block,
    AttU_Net,
    ContractiveBlock,
    ConvolutionBlock,
    ExpansiveBlock,
    NestedUNet,
    R2AttU_Net,
    R2U_Net,
    Recurrent_block,
    RRCNN_block,
    U_Net,
    Unet_dict,
    conv_block,
    conv_block_nested,
    up_conv,
)
from .vgg import VGG, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn


def resattnet56(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("resattnet56", pretrained=False)
    model.output = nn.Linear(2048, 7)
    return model


def cbam_resnet50(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("cbam_resnet50", pretrained=True)
    model.output = nn.Linear(2048, 7)
    return model


def bam_resnet50(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("bam_resnet50", pretrained=True)
    model.output = nn.Linear(2048, 7)
    return model


def efficientnet_b7b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b7b", pretrained=True)
    model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(2560, 7))
    return model


def efficientnet_b3b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b3b", pretrained=True)
    model.output = nn.Sequential(nn.Dropout(p=0.3, inplace=False), nn.Linear(1536, 7))
    return model


def efficientnet_b2b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b2b", pretrained=True)
    model.output = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False), nn.Linear(1408, 7, bias=True)
    )
    return model


def efficientnet_b1b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b1b", pretrained=True)
    print(model)
    model.output = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False), nn.Linear(1280, 7, bias=True)
    )
    return model
