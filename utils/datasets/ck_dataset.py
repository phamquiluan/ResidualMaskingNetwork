import os
import json
import glob

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import sys

sys.path.append("/home/z/research/tee")
from utils.augmenters.augment import seg

"""
0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
"""


class CkDataset(Dataset):
    def __init__(self, stage, fold_idx, configs):
        """ fold_idx: test fold """
        self._configs = configs
        self._stage = stage
        self._fold_idx = fold_idx
        self._data = []

        for fold_path in glob.glob("./saved/data/CK+/npy_folds/*.npy"):
            fold_name = os.path.basename(fold_path)
            if str(fold_idx) == fold_name[5:-4] and stage == "test":
                self._data = np.load(fold_path, allow_pickle=True).tolist()
            if str(fold_idx) != fold_name[5:-4] and stage == "train":
                self._data.extend(np.load(fold_path, allow_pickle=True).tolist())

        self._image_size = (configs["image_size"], configs["image_size"])
        self._transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        image_name, image, label = self._data[idx]
        image = cv2.resize(image, self._image_size)

        assert image.shape[2] == 3
        assert label <= 7 and label >= 0

        if self._stage == "train":
            image = seg(image=image)

        return self._transform(image), label


def ckdataset(stage, fold_idx, configs):
    return CkDataset(stage, fold_idx, configs)


if __name__ == "__main__":
    data = ckdataset(
        stage="train", fold_idx=1, configs={"image_size": 224, "in_channels": 3}
    )
    import cv2
    from barez import pp

    targets = []

    for i in range(len(data)):
        image, target = data[i]
        cv2.imwrite("debug/{}.png".format(i), image)
        if i == 200:
            break
