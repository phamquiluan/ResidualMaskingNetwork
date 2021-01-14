import os
import cv2
import pandas as pd
import numpy as np
import imgaug.augmenters as iaa
import torch

from torch.utils.data import Dataset
from torchvision import transforms

# from imgaug.augmentables.segmaps import SegmentationMapOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from utils.utils import read_unicode_image as imread
from skimage.morphology import skeletonize


def get_skeletion(m):
    m = m.astype(np.uint8) / 255.0
    m = skeletonize(m)
    m = (m * 255).astype(np.uint8)
    return m


STANDARD_DOC_IMAGE_VSHAPE = 2000
PATCH_SIZE = 256


class TableDataset(Dataset):
    def __init__(self, stage, configs=None):
        root_dir = configs["data_path"]
        self._root_dir = root_dir
        self._images_dir = os.path.join(root_dir, "images")

        self._stage = stage

        self._info_path = os.path.join(root_dir, "{}.txt".format(stage))
        self._info = pd.read_csv(self._info_path, header=None, error_bad_lines=False)

        self._is_cached = configs["cached_npy"] == 1

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        self._images = {}
        if self._is_cached and os.path.exists(
            os.path.join(root_dir, "{}.npy".format(stage))
        ):
            self._images = np.load(
                os.path.join(root_dir, "{}.npy".format(stage)), allow_pickle=True
            ).tolist()

        if len(self._images) == 0 and self._stage != "little":
            for idx in range(len(self._info)):
                image_name = self._info.iloc[idx, 0]
                if image_name in self._images:
                    continue

                print("Read {}".format(image_name))
                self._images[image_name] = imread(
                    os.path.join(self._images_dir, image_name)
                )

            if self._is_cached:
                np.save(
                    os.path.join(self._root_dir, "{}.npy".format(self._stage)),
                    self._images,
                )

    def __len__(self):
        return len(self._info)

    def _ensure_standard_shape(self, image):
        ratio = STANDARD_DOC_IMAGE_VSHAPE / image.shape[0]
        return cv2.resize(image, None, fx=ratio, fy=ratio)

    def _random_crop(self, image, mask):
        height, width = image.shape[:2]
        new_height, new_width = [PATCH_SIZE] * 2

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)
        if self._stage == "little":
            top, left = PATCH_SIZE, PATCH_SIZE
            # top, left = 100, 100

        image = image[top : top + new_height, left : left + new_width]
        mask = mask[top : top + new_height, left : left + new_width]
        return image, mask

    def _aug(self, image, mask):
        mask = SegmentationMapsOnImage(mask, image.shape[:2])

        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Rot90([0, 3]),
                iaa.SomeOf(
                    1,
                    [
                        iaa.Affine(scale={"x": (0.7, 1.5), "y": (1.6, 1.5)}),
                        iaa.Affine(rotate=(-30, 30)),
                        #     iaa.Add((-110, 111)),
                        #     iaa.GaussianBlur(sigma=1.8 * np.random.rand()),
                        #     iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                        #     iaa.AdditiveGaussianNoise(scale=0.05*255),
                        #     iaa.Multiply((0.5, 1.5)),
                        #     iaa.Affine(shear=(-20, 20)),
                        #     iaa.PiecewiseAffine(scale=(0.01, 0.02)),
                        iaa.PerspectiveTransform(scale=(0.01, 0.1)),
                    ],
                ),
            ],
            random_order=True,
        )

        image, mask = seq(image=image, segmentation_maps=mask)
        mask = mask.get_arr_int().astype(np.uint8)
        return image, mask

    def __getitem__(self, idx):
        image_name = self._info.iloc[idx, -1]

        if len(self._images) == 0:
            image = imread(os.path.join(self._images_dir, image_name))
        else:
            image = self._images[image_name]

        image = self._ensure_standard_shape(image)
        center = image.shape[1] // 2
        mask = image[:, center:]
        image = image[:, :center]

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        mask = get_skeletion(mask)
        mask = cv2.dilate(mask, np.ones((5, 5)))

        mask[mask == 255] = 1

        if self._stage == "train":
            try:
                image, mask = self._aug(image, mask)
                mask *= 255
            except Exception as e:
                print("Exception", e)

        image, mask = self._random_crop(image, mask)
        image = self._transform(image)
        mask = torch.LongTensor(mask)
        mask = torch.clamp(mask, 0, 1)
        return image, mask
