import os
import sys
import glob
import json
import random
import warnings
warnings.filterwarnings("ignore")
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=DeprecationWarning)

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


def main(config_path):
    """
    This is the main function to make the training up

    Parameters:
    -----------
    config_path : srt
        path to config file
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs['cwd'] = os.getcwd()

    # load model and data_loader
    model = get_model(configs)

    train_set, val_set, test_set = get_dataset(configs)

    from trainers.table_trainer import TableTrainer
    trainer = TableTrainer(model, train_set, val_set, test_set, configs)

    # TODO: do mp here
    if configs['distributed'] == 1:
        ngpus = torch.cuda.device_count()
        mp.spawn(trainer.train, nprocs=ngpus, args=())
    else:
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
        return models.__dict__[configs['arch']]
    except KeyError:
        return segmentation.__dict__[configs['arch']]


def get_dataset(configs):
    """
    This function get raw dataset
    """
    from utils.datasets.table_dataset import TableDataset

    # todo: add transform
    if configs['little'] == 1:
        train_set = TableDataset('little', configs)
        val_set = TableDataset('little', configs)
        test_set = val_set
        return train_set, val_set, test_set
    else:
        train_set = TableDataset('train', configs)
        val_set = TableDataset('val', configs)
        test_set = val_set
        return train_set, val_set, test_set


if __name__ == "__main__":
    main('./configs/table_config.json')
