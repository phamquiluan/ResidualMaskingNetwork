import os
import sys
import glob
import json

import torch
import numpy as np
import imgaug
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

import models
from models import segmentation
from utils import datasets
# from trainers.trainer import TeeTrainer
from trainers.emma_trainer import EmmaTrainer
from trainers.ck_trainer import CKPlusTrainer


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

    # init trainer and make a training
    if configs['trainer_name'] == 'emma':
        trainer = EmmaTrainer(model, train_set, val_set, configs)
    else:
        trainer = TeeTrainer(model, train_set, val_set, configs)

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
    dataset_init_fn = datasets.__dict__[configs['dataset_name']]
    dataset_root_dir = configs['data_path']

    if configs['little'] == 1:
        train_set = dataset_init_fn(dataset_root_dir, 'little', configs)
        test_set = val_set = train_set
    else:
        train_set = dataset_init_fn(dataset_root_dir, 'train', configs)
        val_set = dataset_init_fn(dataset_root_dir, 'val', configs)
        if configs['data_test'] == 0:
            test_set = val_set
        else:
            test_set = dataset_init_fn(dataset_root_dir, 'test', configs)
    return train_set, val_set, test_set


if __name__ == "__main__":
    main('./configs/config.json')
