import imgaug
from imgaug import augmenters as iaa

imgaug.seed(1)

seg = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-20, 20)),
        iaa.OneOf(
            [
                iaa.GaussianBlur(sigma=(0.0, 4.0)),
                iaa.Dropout((0.0, 0.15), per_channel=0.5),
            ]
        ),
        iaa.OneOf(
            [
                iaa.Add((-60, 60), per_channel=True),
                iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.5, 1.5)),
                iaa.GammaContrast(gamma=(0.5, 1.0), per_channel=True),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.14 * 255, 0.15 * 255), per_channel=0.5
                ),
            ]
        ),
    ]
)
