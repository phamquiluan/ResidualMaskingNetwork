import os

from setuptools import find_packages, setup

version = None
with open("README.md") as ref:
    data = ref.readlines()[2]
    version = data[data.find("version-v")+9:data.find("-blue")]
    assert version is not None, data

cwd = os.path.dirname(os.path.abspath(__file__))

def write_version_file():
    version_path = os.path.join(cwd, "rmn", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")


write_version_file()


with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rmn",
    description="Facial Expression Recognition using Residual Masking Network",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=version,
    author="Luan Pham",
    author_email="phamquiluan@gmail.com",
    packages=find_packages(
        exclude=["docs", "tests", "env", "script", "trainers", "utils", "pretrained_ckpt"]
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
