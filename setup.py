import os

from setuptools import find_packages, setup

version = None
with open("README.md") as ref:
    data = ref.readlines()[3]
    version_prefix = "version-v"
    version = data[data.find(version_prefix) + len(version_prefix) : data.find("-blue")]
    assert version is not None, data

cwd = os.path.dirname(os.path.abspath(__file__))
version_path = os.path.join(cwd, "rmn", "version.py")
with open(version_path, "w") as ref:
    ref.write(f"__version__ = '{version}'\n")


with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rmn",
    description="Facial Expression Recognition using Residual Masking Network",
    homepage="https://github.dev/phamquiluan/ResidualMaskingNetwork",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=version,
    author="Luan Pham",
    author_email="phamquiluan@gmail.com",
    packages=find_packages(
        exclude=[
            "docs",
            "tests",
            "env",
            "script",
            "trainers",
            "utils",
            "pretrained_ckpt",
        ]
    ),
    include_package_data=True,
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "torchvision",
        "requests",
        "pytorchcv",
        "tqdm",
    ],
)
