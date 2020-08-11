import os
import glob
import json
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from models import densenet121, resmasking_dropout1
from barez import show, ensure_gray, ensure_color

# haar = '/home/z/anaconda3/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml'
haar = "/home/z/anaconda3/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(haar)

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def main(image_path):
    # load configs and set random seed
    configs = json.load(open("./configs/fer2013_config.json"))
    image_size = (configs["image_size"], configs["image_size"])

    # model = densenet121(in_channels=3, num_classes=7)
    model = resmasking_dropout1(in_channels=3, num_classes=7)
    model.cuda()

    state = torch.load("./saved/checkpoints/resmasking_dropout1_rot30_2019Nov17_14.33")
    model.load_state_dict(state["net"])
    model.eval()

    image = cv2.imread(image_path)

    faces = face_cascade.detectMultiScale(image, 1.15, 5)
    gray = ensure_gray(image)
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (179, 255, 179), 2)

        face = gray[y : y + h, x : x + w]
        face = ensure_color(face)

        face = cv2.resize(face, image_size)
        face = transform(face).cuda()
        face = torch.unsqueeze(face, dim=0)

        output = torch.squeeze(model(face), 0)
        proba = torch.softmax(output, 0)

        emo_proba, emo_idx = torch.max(proba, dim=0)
        emo_idx = emo_idx.item()
        emo_proba = emo_proba.item()

        emo_label = FER_2013_EMO_DICT[emo_idx]

        label_size, base_line = cv2.getTextSize(
            "{}: 000".format(emo_label), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(
            image,
            (x + w, y + 1 - label_size[1]),
            (x + w + label_size[0], y + 1 + base_line),
            (223, 128, 255),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            "{}: {}".format(emo_label, int(emo_proba * 100)),
            (x + w, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

    from barez import show

    show(image)
    """
    cv2.imshow('disp', image)
    key = cv2.waitKey(0)
    if key == ord('w'):
        cv2.imwrite('./real_life_demo/{}'.format(os.path.basename(image_path)), image)

    cv2.destroyAllWindows()
    """


if __name__ == "__main__":
    import sys

    argv = sys.argv[1]
    assert isinstance(argv, str) and os.path.exists(argv)
    main(argv)
