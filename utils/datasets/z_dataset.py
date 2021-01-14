import os
import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset

# import sys
# sys.path.append('/home/z/research/tee/')
from utils.augmenters.augment import seg


EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


class Z(Dataset):
    def __init__(self, stage, configs, tta=False, tta_size=48):
        self._stage = stage
        self._root_dir = configs["data_path"]
        self._configs = configs
        self._tta = tta
        self._tta_size = tta_size

        self._image_size = (configs["image_size"], configs["image_size"])

        self._images = []

        npy_path = os.path.join(self._root_dir, "{}_data.npy".format(stage))

        if os.path.exists(npy_path):
            self._images = np.load(npy_path, allow_pickle=True).tolist()
        else:
            self._data = np.load(
                os.path.join(configs["data_path"], "{}.npy".format(stage))
            )

            for image_path, label in self._data:
                print("Read", image_path)
                image = cv2.imread(os.path.join(self._configs["data_path"], image_path))
                # image = cv2.resize(image, self._image_size)
                self._images.append([image, label])

            np.save(npy_path, self._images)

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    def is_tta(self):
        return self._tta == True

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        # image_path, label =  self._data[idx]
        # image_path = os.path.join(self._configs['data_path'], image_path)
        # image = cv2.imread(image_path)
        image, label = self._images[idx]
        image = cv2.resize(image, self._image_size)

        if self._stage == "train":
            image = seg(image=image)

        if self._stage == "test" and self._tta == True:
            images = [seg(image=image) for i in range(self._tta_size)]
            images = list(map(self._transform, images))
            return images, int(label)

        image = self._transform(image)
        return image, int(label)


def z(stage, configs=None, tta=False, tta_size=48):
    return Z(stage, configs, tta, tta_size)


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
