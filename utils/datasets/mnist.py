from torchvision import datasets
from torchvision.transforms import transforms


def mnist(root_dir, stage, configs):
    if stage == "train":
        return datasets.MNIST(
            root_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

    return datasets.MNIST(
        root_dir,
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
