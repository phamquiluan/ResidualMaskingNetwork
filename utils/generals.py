import torch


def make_batch(images):
    if not isinstance(images, list):
        images = [images]
    return torch.stack(images, 0)
