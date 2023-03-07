import json
import random

import cv2
import imgaug
import numpy as np
import torch
import torch.nn.functional as F

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from tqdm import tqdm

from utils.datasets.fer2013dataset import fer2013
from utils.generals import make_batch

class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


checkpoint_name = "resmasking_dropout1_rot30_2019Nov17_14.33"
# checkpoint_name = 'Z_resmasking_dropout1_rot30_2019Nov30_13.32'


def main():
    with open("./configs/fer2013_config.json") as f:
        configs = json.load(f)

    state = torch.load("./saved/checkpoints/{}".format(checkpoint_name))

    from models import resmasking_dropout1

    model = resmasking_dropout1

    model = model(in_channels=3, num_classes=7).cuda()
    model.load_state_dict(state["net"])
    model.eval()

    total = 0

    test_set = fer2013("test", configs, tta=True, tta_size=8)
    hold_test_set = fer2013("test", configs, tta=False, tta_size=0)

    with torch.no_grad():
        for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
            images, targets = test_set[idx]

            images = make_batch(images)
            images = images.cuda(non_blocking=True)

            outputs = model(images).cpu()
            outputs = F.softmax(outputs, 1)

            # outputs.shape [tta_size, 7]
            outputs = torch.sum(outputs, 0)
            outputs = torch.argmax(outputs, 0)
            outputs = outputs.item()
            targets = targets.item()
            total += 1
            if outputs != targets:
                image, target = hold_test_set[idx]
                image = image.permute(1, 2, 0).numpy() * 255
                image = image.astype(np.uint8)

                cv2.imwrite(
                    "./wrong_in_fer/{}->{}_{}.png".format(
                        class_names[target], class_names[outputs], idx
                    ),
                    image,
                )


if __name__ == "__main__":
    main()
