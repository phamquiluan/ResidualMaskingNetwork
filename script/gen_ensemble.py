import random
import imgaug
import torch
import numpy as np

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model_dict = [
    ("resnet18", "resnet18_rot30_2019Nov05_17.44"),
    ("resnet50_pretrained_vgg", "resnet50_pretrained_vgg_rot30_2019Nov13_08.20"),
    ("resnet101", "resnet101_rot30_2019Nov14_18.12"),
    ("cbam_resnet50", "cbam_resnet50_rot30_2019Nov15_12.40"),
    ("efficientnet_b2b", "efficientnet_b2b_rot30_2019Nov15_20.02"),
    ("resmasking_dropout1", "resmasking_dropout1_rot30_2019Nov17_14.33"),
    ("resmasking", "resmasking_rot30_2019Nov14_04.38"),
]


# model_dict_proba_list = list(map(list, product([0, 1], repeat=len(model_dict))))

model_dict_proba_list = [[1, 1, 1, 1, 1, 1, 1]]


def main():
    test_results_list = []
    for model_name, checkpoint_path in model_dict:
        test_results = np.load(
            "./saved/results/{}.npy".format(checkpoint_path), allow_pickle=True
        )
        test_results_list.append(test_results)
    test_results_list = np.array(test_results_list)

    # load test targets
    test_targets = np.load("./saved/test_targets.npy", allow_pickle=True)

    model_dict_proba = [1, 1, 1, 1, 1, 1, 1]

    tmp_test_result_list = []
    for idx in range(len(model_dict_proba)):
        tmp_test_result_list.append(model_dict_proba[idx] * test_results_list[idx])
    tmp_test_result_list = np.array(tmp_test_result_list)
    tmp_test_result_list = np.sum(tmp_test_result_list, axis=0)
    tmp_test_result_list = np.argmax(tmp_test_result_list, axis=1)

    correct = np.sum(np.equal(tmp_test_result_list, test_targets))

    acc = correct / 3589 * 100
    print(acc)


if __name__ == "__main__":
    main()
