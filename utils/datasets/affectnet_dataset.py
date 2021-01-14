import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import sys

sys.path.append("/home/z/research/tee/")
from utils.augmenters.augment import seg


EMOTION_DICT = {
    0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
    8: "None",
    9: "Uncertain",
    10: "No-Face",
}


class AffectNet(Dataset):
    def __init__(self, stage, configs, tta=False, tta_size=48):
        self._stage = stage
        self._configs = configs
        self._tta = tta
        self._tta_size = tta_size

        self._image_size = (configs["image_size"], configs["image_size"])

        if stage == "train":
            self._data = pd.read_csv(os.path.join(configs["data_path"], "train.npy"))
        elif stage == "val":
            self._data = pd.read_csv(os.path.join(configs["data_path"], "val.npy"))
        else:
            raise Exception("just train or val")

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    def is_tta(self):
        return self._tta == True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        image_path, label = self._data[idx]
        image_path = os.path.join(self._configs["data_path"], image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, self._image_size)

        if self._stage == "train":
            image = seg(image=image)

        if self._stage == "test" and self._tta == True:
            images = [seg(image=image) for i in range(self._tta_size)]
            images = list(map(self._transform, images))
            return images, expression

        image = self._transform(image)
        return image, label


def affectnet(stage, configs=None, tta=False, tta_size=48):
    return AffectNet(stage, configs, tta, tta_size)


if __name__ == "__main__":
    data = affectnet(
        "val",
        {
            "_data_path": "/home/z/research/tee/saved/data/fer2013/",
            "data_path": "/data/emotion_data/ANDB/affectnet_db/small_zips/Manually_Annotated_compressed",
            "image_size": 224,
            "in_channels": 3,
        },
    )
    import cv2
    from barez import pp

    targets = []

    cnt = 0
    for i in range(len(data)):
        image, target = data[i]
        if image is None:
            continue
        cv2.imwrite("debug/{}_{}.png".format(EMOTION_DICT[target], i), image)
        if cnt == 200:
            break

        cnt += 1
