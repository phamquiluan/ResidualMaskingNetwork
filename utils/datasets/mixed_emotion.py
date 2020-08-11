import os
import json
import glob

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# from utils.utils import ensure_color

# TODO: check what if we capitalize the class Name and
# make another function to return this class


class mixed_emotion(Dataset):
    def __init__(self, root_dir, stage, configs, transform=None):
        self._emo_dict = {
            "ne": 0,
            "an": 1,
            "di": 2,
            "fe": 3,
            "ha": 4,
            "sa": 5,
            "su": 6,
        }

        self._configs = configs
        self._root_dir = root_dir
        self._stage = stage
        self._image_dir = os.path.join(self._root_dir, "images")
        self._is_cached = self._configs["cached_npy"] == 1
        self._image_paths = self._read_info()
        self._image_size = self._configs["image_size"]

        # set the image list up
        self._images = []
        if self._is_npy():
            print("Load npy..")
            self._images = self._load_npy()

        if len(self._images) == 0 and self._is_cached:
            for image_path in self._image_paths:
                print("Read", image_path)
                self._images.append(
                    cv2.imread(os.path.join(self._image_dir, image_path))
                )

            np.save(
                os.path.join(self._root_dir, "{}.npy".format(self._stage)), self._images
            )

        if transform:
            self._transform = transform
        else:
            self._transform = transforms.Compose(
                [transforms.ToPILImage(), transforms.ToTensor()]
            )

    def _read_info(self):
        with open(
            os.path.join(self._root_dir, "{}.json".format(self._stage))
        ) as info_file:
            return json.load(info_file)

    def _is_npy(self):
        return self._is_cached and os.path.exists(
            os.path.join(self._root_dir, "{}.npy".format(self._stage))
        )

    def _load_npy(self):
        npy_path = os.path.join(self._root_dir, "{}.npy".format(self._stage))
        return np.load(npy_path, allow_pickle=True).tolist()

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        if self._is_cached:
            image = self._images[idx]
        else:
            image = cv2.imread(os.path.join(self._image_dir, image_path))

        image = cv2.resize(image, (self._image_size, self._image_size))
        emo = self._emo_dict[image_path[:2]]
        return self._transform(image), emo


class mixed_infer(Dataset):
    def __init__(self, root_dir, configs, transform=None):
        self._emo_dict = {
            "ne": 0,
            "an": 1,
            "di": 2,
            "fe": 3,
            "ha": 4,
            "sa": 5,
            "su": 6,
        }

        self._configs = configs
        self._root_dir = root_dir
        self._image_dir = os.path.join(self._root_dir, "images")
        self._image_paths = self._read_info()
        self._image_size = self._configs["image_size"]

        if transform:
            self._transform = transform
        else:
            self._transform = transforms.Compose(
                [transforms.ToPILImage(), transforms.ToTensor()]
            )

    def _read_info(self):
        return sorted(
            glob.glob(os.path.join(self._image_dir, "**/*.png"), recursive=True)
        )

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        image = cv2.imread(os.path.join(self._image_dir, image_path))
        image = cv2.resize(image, (self._image_size, self._image_size))
        return self._transform(image), os.path.join(self._image_dir, image_path)


if __name__ == "__main__":
    from barez import show

    dataset = mixed_emotion("/data/emotion_data/MixedEmotion/CK+/", "test", {})

    cnt = 0
    while True:
        cnt += 1
        image, emo = dataset[cnt]
        show(image)
