from imgaug import augmenters as iaa

seg = iaa.Sequential(
    [
        # iaa.Fliplr(p=0.5, deterministic=True),
        iaa.Fliplr(p=0.5),
        # iaa.Affine(rotate=(-30, 30), deterministic=True),
        iaa.Affine(rotate=(-30, 30)),
        # iaa.GaussianBlur(sigma=(0., 4.0), deterministic=True),
        # iaa.Dropout((0., 0.15), deterministic=True),
        # iaa.Add((-25, 25), deterministic=True),
        # iaa.CropAndPad(percent=(-0.05, 0.1), pad_cval=(0, 255), deterministic=True)
    ]
)
