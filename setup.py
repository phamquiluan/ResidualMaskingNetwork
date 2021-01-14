import os

from setuptools import find_packages, setup

version = "0.0.1"
cwd = os.path.dirname(os.path.abspath(__file__))



def write_version_file():
    version_path = os.path.join(cwd, "rmn", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))


write_version_file()

setup(
    name="rmn",
    description="Facial Expression Recognition using Residual Masking Network",
    version=version,
    packages=find_packages(
        exclude=["docs", "tests", "configs", "env", "script", "trainers", "utils"]
    ),
    include_package_data=True,
    install_requires=["numpy", "opencv-python", "torch", "torchvision", "requests" ,"pytorchcv", "tqdm"],
)
