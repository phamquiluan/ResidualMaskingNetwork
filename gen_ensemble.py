import os
import random
import json
import imgaug
import torch
import numpy as np
from itertools import product

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
 
from tqdm import tqdm
import models
import torch.nn.functional as F
from utils.datasets.fer2013dataset import fer2013
from utils.generals import make_batch

model_dict = [ 
    ('resnet18', 'resnet18_rot30_2019Nov05_17.44'),
    # ('googlenet', 'googlenet_rot30_2019Nov11_15.20'),
    # ('googlenet', 'googlenet_rot30_freeze_idx7_2019Nov11_16.00'),
    # ('densenet121', 'densenet121_rot30_2019Nov11_14.23'),
    # ('inception_v3', 'inception_v3_rot30_2019Nov11_16.55'),
    ('resnet50_pretrained_vgg', 'resnet50_pretrained_vgg_rot30_2019Nov13_08.20'),
    # ('resnet34', 'resnet34_rot30_2019Nov14_10.42'),
    # ('resnet50', 'resnet50_rot30_2019Nov14_16.09'),
    ('resnet101', 'resnet101_rot30_2019Nov14_18.12'),
    # ('resnet152', 'resnet152_rot30_2019Nov14_12.47'),
    ('cbam_resnet50', 'cbam_resnet50_rot30_2019Nov15_12.40'),
    # ('bam_resnet50', 'bam_resnet50_rot30_2019Nov15_17.10'),
    ('efficientnet_b2b', 'efficientnet_b2b_rot30_2019Nov15_20.02'),
    ('resmasking_dropout1', 'resmasking_dropout1_rot30_2019Nov17_14.33'),
    # ('vgg19', 'vgg19_rot30_2019Dec01_14.01'),
    # ('resnet18_centerloss', 'resnet18_centerloss_rot30_2019Nov09_18.24'),
    # ('resnet18', 'resnet18_rot30_no_fixed_2019Nov11_08.33'),
    # ('resnet18', 'resnet18_rot30_fixed_layers_1_2019Nov11_10.03'),
    # ('resnet18', 'resnet18_rot30_fixed_layers_1234_2019Nov11_11.16'),
    # ('resnet18', 'resnet18_rot30_2019Nov11_12.56'),
    ('resmasking', 'resmasking_rot30_2019Nov14_04.38'),


    # ('resmasking', 'resmasking_rot30_2019Nov17_06.13'),
    # ('resmasking_dropout2', 'resmasking_dropout2_rot30_2019Nov17_14.34')
]


# print("We have {} models".format(len(model_dict)))

model_dict_proba_list = list(map(list, product([0, 1], repeat=len(model_dict))))
total_proba = len(model_dict_proba_list)

weights_and_acc = []


def main():
    # load val_results_list 
    val_results_list = []
    for model_name, checkpoint_path in model_dict:
        # val_results = np.load('./saved/val_results/{}.npy'.format(checkpoint_path), allow_pickle=True)
        val_results = np.load('./saved/results/{}.npy'.format(checkpoint_path), allow_pickle=True)
        val_results_list.append(val_results)
        print(val_results.shape)
    val_results_list = np.array(val_results_list) 

    # load val targets
    val_targets = np.load('./saved/test_targets.npy', allow_pickle=True)
    

    # loop weight for val_results_dict -> then compare with target
    # for proba_idx, model_dict_proba in enumerate(model_dict_proba_list):
    for proba_idx in tqdm(
            range(total_proba),
            total=total_proba,
            leave=False
        ):
        model_dict_proba = model_dict_proba_list[proba_idx]

        # if model_dict_proba[13] == 0:
        #     continue


        tmp_val_result_list = []
        for idx in range(len(model_dict_proba)):
            tmp_val_result_list.append(model_dict_proba[idx] *  val_results_list[idx])
        tmp_val_result_list = np.array(tmp_val_result_list)
        tmp_val_result_list = np.sum(tmp_val_result_list, axis=0)
        tmp_val_result_list = np.argmax(tmp_val_result_list, axis=1)

        correct = np.sum(np.equal(tmp_val_result_list, val_targets))

        
        acc = correct / 3589 * 100
        weights_and_acc.append([model_dict_proba, acc])
        # print("{:05d}/{} acc {:.3f}".format(proba_idx, total_proba, correct / 3589 * 100))
    
    print(max(weights_and_acc, key=lambda x:x[1])) 



def checking_on_test_set(proba):
    with open('./configs/fer2013_config.json') as f:
        configs = json.load(f)
 
    test_set = fer2013('test', configs, tta=True, tta_size=8)
    test_npy = []
    for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
        images, targets = test_set[idx]
        test_npy.append(targets)
    np.save('./saved/test_targets.npy', test_npy)
 
 


if __name__ == "__main__":
    main()
    # checking_on_test_set([])
