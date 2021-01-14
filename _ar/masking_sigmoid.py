import os
import glob
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from natsort import natsorted
from models import resmasking_dropout1
from utils.datasets.fer2013dataset import EMOTION_DICT
from barez import show

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)


def activations_mask(tensor):
    tensor = torch.squeeze(tensor, 0)
    tensor = torch.mean(tensor, 0)
    tensor = tensor.detach().cpu().numpy()
    tensor = np.maximum(tensor, 0)
    tensor = cv2.resize(tensor, (224, 224))
    tensor = tensor - np.min(tensor)
    tensor = tensor / np.max(tensor)

    # print(np.unique(tensor))

    heatmap = cv2.applyColorMap(np.uint8(255 * tensor), cv2.COLORMAP_JET)
    return heatmap
    # return tensor


model = resmasking_dropout1(3, 7)
state = torch.load(
    "./saved/checkpoints/resmasking_naive_dropout1__sigmoid_2019Dec17_14.40"
)
model.load_state_dict(state["net"])
model.cuda()
model.eval()

# for image_path  in natsorted(glob.glob('/home/z/research/bkemo/images/**/*.png', recursive=True)):
for image_path in natsorted(
    glob.glob("/home/z/research/bkemo/debug/**/*.png", recursive=True)
):
    image_name = os.path.basename(image_path)

    if not os.path.exists("./landmark_false/{}".format(image_name)):
        continue

    print(image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    tensor = transform(image)
    tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.cuda()

    # output = model(tensor)

    x = model.conv1(tensor)  # 112
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)  # 56

    x = model.layer1(x)  # 56
    m = model.mask1(x)
    x = x * m

    x = model.layer2(x)  # 28
    m = model.mask2(x)
    x = x * m

    x = model.layer3(x)  # 14
    heat_1 = activations_mask(x)
    m = model.mask3(x)
    x = x * m
    heat_2 = activations_mask(m)

    # x = model.layer4(x)  # 7
    # m = model.mask4(x)
    # x = x * m

    # x = model.avgpool(x)
    # x = torch.flatten(x, 1)

    # output = model.fc(x)

    # show(np.concatenate((image, heat_1, heat_2), axis=1))
    debug_image = np.concatenate((image, heat_1, heat_2), axis=1)
    cv2.imwrite("./debug/{}".format(image_name), debug_image)
    # cv2.imshow('disp', debug_image)
    # if cv2.waitKey(0) == ord('w'):

    # cv2.imwrite(
    #     './debug/{}'.format(image_name),
    #     (image * np.dstack([heat_2] * 3)).astype(np.uint8)
    # )

    # cv2.destroyAllWindows()

    # cv2.imwrite('./masking_provements/{}'.format(image_name),
    #     np.concatenate((image, heat_1), axis=1))
