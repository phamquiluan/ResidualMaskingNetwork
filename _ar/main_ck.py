import os
import sys
import glob
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import models
from models import segmentation
from utils import datasets


def create_fold_and_train(fold_idx, configs):
    model = get_model(configs)
    train_set = CKDataset("train", fold_idx, configs)
    test_set = CKDataset("test", fold_idx, configs)

    trainer = CKPlusTrainer(model, train_set, test_set, fold_idx, configs)

    trainer.train()


def main(config_path, fold_idx):
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()

    processes = []

    # load model and data_loader
    from utils.datasets.ck_dataset import ckdataset
    from trainers.ck_trainer import CkTrainer
    import time

    # for fold_idx in range(1, 11, 1):
    #     for r_time in range(5):
    #         time.sleep(1)
    #         print("start fold {} in {} seconds..".format(fold_idx, 5 - r_time))

    model = get_model(configs)
    train_set = ckdataset(stage="train", fold_idx=fold_idx, configs=configs)
    test_set = ckdataset(stage="test", fold_idx=fold_idx, configs=configs)

    trainer = CkTrainer(model, train_set, test_set, fold_idx, configs)
    trainer.train()


def get_model(configs):
    """
    This function get raw models from models package

    Parameters:
    ------------
    configs : dict
        configs dictionary
    """
    try:
        return models.__dict__[configs["arch"]]
    except KeyError:
        return segmentation.__dict__[configs["arch"]]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) != 1:
        raise Exception("You need to declare fold_idx")
    main("./configs/ck_config.json", argv[0])
