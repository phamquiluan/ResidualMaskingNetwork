import imgaug
from imgaug import augmenters as iaa

imgaug.seed(1)

seg = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-10, 10)),
    ]
)
