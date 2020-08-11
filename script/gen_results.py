import os
import random
import json
import imgaug
import torch
import numpy as np

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tqdm import tqdm
import models
import torch.nn.functional as F
from utils.datasets.fer2013dataset import fer2013
from utils.generals import make_batch

model_dict = [
    ("resnet18", "resnet18_rot30_2019Nov05_17.44"),
    # ('resatt18': 'resatt18_rot30_2019Nov06_18.56',
    # ('rdg': 'rdg_rot30_2019Nov08_05.57',
    ("googlenet", "googlenet_rot30_2019Nov11_15.20"),
    ("googlenet", "googlenet_rot30_freeze_idx7_2019Nov11_16.00"),
    ("densenet121", "densenet121_rot30_2019Nov11_14.23"),
    ("inception_v3", "inception_v3_rot30_2019Nov11_16.55"),
    ("resnext50_32x4d", "resnext50_32x4d_rot30_freeze_idx7_2019Nov11_16.29"),
    ("resnet50_pretrained_vgg", "resnet50_pretrained_vgg_rot30_2019Nov13_08.20"),
    ("resnet34", "resnet34_rot30_2019Nov14_10.42"),
    ("resnet50", "resnet50_rot30_2019Nov14_16.09"),
    ("resnet101", "resnet101_rot30_2019Nov14_18.12"),
    ("resnet152", "resnet152_rot30_2019Nov14_12.47"),
    ("cbam_resnet50", "cbam_resnet50_rot30_2019Nov15_12.40"),
    ("bam_resnet50", "bam_resnet50_rot30_2019Nov15_17.10"),
    ("efficientnet_b2b", "efficientnet_b2b_rot30_2019Nov15_20.02"),
    ("resmasking_dropout1", "resmasking_dropout1_rot30_2019Nov17_14.33"),
    ("vgg19", "vgg19_rot30_2019Dec01_14.01"),
    ("resnet18_centerloss", "resnet18_centerloss_rot30_2019Nov09_18.24"),
    ("resnet18", "resnet18_rot30_no_fixed_2019Nov11_08.33"),
    ("resnet18", "resnet18_rot30_fixed_layers_1_2019Nov11_10.03"),
    ("resnet18", "resnet18_rot30_fixed_layers_1234_2019Nov11_11.16"),
    ("resnet18", "resnet18_rot30_2019Nov11_12.56"),
    # ('resmasking', 'resmasking_rot30_2019Nov13_11.37'),
    # ('resmasking', 'resmasking_rot30_2019Nov13_14.12'),
    # ('resmasking', 'resmasking_rot30_2019Nov13_18.58'),
    ("resmasking", "resmasking_rot30_2019Nov14_04.38"),
    ("resmasking", "resmasking_rot30_2019Nov17_06.13"),
    ("resmasking_dropout2", "resmasking_dropout2_rot30_2019Nov17_14.34"),
]


def main():
    with open("./configs/fer2013_config.json") as f:
        configs = json.load(f)

    test_set = fer2013("test", configs, tta=True, tta_size=8)

    for model_name, checkpoint_path in model_dict:
        prediction_list = []  # each item is 7-ele array

        print("Processing", checkpoint_path)
        if os.path.exists("./saved/results/{}.npy".format(checkpoint_path)):
            continue

        model = getattr(models, model_name)
        model = model(in_channels=3, num_classes=7)

        state = torch.load(os.path.join("saved/checkpoints", checkpoint_path))
        model.load_state_dict(state["net"])

        model.cuda()
        model.eval()

        with torch.no_grad():
            for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
                images, targets = test_set[idx]
                images = make_batch(images)
                images = images.cuda(non_blocking=True)

                outputs = model(images).cpu()
                outputs = F.softmax(outputs, 1)
                outputs = torch.sum(outputs, 0)  # outputs.shape [tta_size, 7]

                outputs = [round(o, 4) for o in outputs.numpy()]
                prediction_list.append(outputs)

        np.save("./saved/results/{}.npy".format(checkpoint_path), prediction_list)


if __name__ == "__main__":
    main()
