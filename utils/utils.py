import cv2
import numpy as np


def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image


def ensure_gray(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    return image


def read_unicode_image(path, gray=1):
    """read unicode image"""
    stream = open(path, "rb")
    byte_array = bytearray(stream.read())
    numpy_array = np.asarray(byte_array, dtype=np.uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    image = np.array(image)
    if gray == 0:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
