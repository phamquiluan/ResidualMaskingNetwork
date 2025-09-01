import os

from setuptools import find_packages, setup

version = None
with open("README.md") as ref:
    lines = ref.readlines()
    version_prefix = "version-v"
    
    # Search for the line containing the version badge
    for line in lines:
        if version_prefix in line and "-blue" in line:
            version = line[line.find(version_prefix) + len(version_prefix) : line.find("-blue")]
            break
    
    if version is None:
        raise ValueError("Could not find version pattern 'version-v...blue' in README.md")
    
    assert version is not None, f"Version parsing failed for README.md"

cwd = os.path.dirname(os.path.abspath(__file__))
version_path = os.path.join(cwd, "rmn", "version.py")
with open(version_path, "w") as ref:
    ref.write(f"__version__ = '{version}'\n")


with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rmn",
    description="Facial Expression Recognition using Residual Masking Network",
    url="https://github.com/phamquiluan/ResidualMaskingNetwork",
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
