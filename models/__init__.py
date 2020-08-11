from .vgg import *
from .resnet import *
from .resnet112 import resnet18x112
from .resnet50_scratch_dims_2048 import resnet50_pretrained_vgg
from .centerloss_resnet import resnet18_centerloss
from .resatt import *
from .alexnet import *
from .densenet import *
from .googlenet import *
from .inception import *
from .inception_resnet_v1 import *
from .residual_attention_network import *
from .fer2013_models import *
from .res_dense_gle import *
from .masking import masking
from .resmasking import (
    resmasking,
    resmasking_dropout1,
    resmasking_dropout2,
    resmasking50_dropout1,
)
from .resmasking_naive import resmasking_naive_dropout1
from .brain_humor import *
from .runet import *
from pytorchcv.model_provider import get_model as ptcv_get_model


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
