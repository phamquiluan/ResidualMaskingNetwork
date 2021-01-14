import os
import glob
import json
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from models import densenet121, resmasking_dropout1


checkpoint_url = "https://github.com/phamquiluan/ResidualMaskingNetwork/releases/download/v0.0.1/Z_resmasking_dropout1_rot30_2019Nov30_13.32"
local_checkpoint_path = "pretrained_ckpt"

prototxt_url = "https://github.com/phamquiluan/ResidualMaskingNetwork/releases/download/v0.0.1/deploy.prototxt.txt"
local_prototxt_path = "deploy.prototxt.txt"

ssd_checkpoint_url = "https://github.com/phamquiluan/ResidualMaskingNetwork/releases/download/v0.0.1/res10_300x300_ssd_iter_140000.caffemodel"
local_ssd_checkpoint_path = "res10_300x300_ssd_iter_140000.caffemodel"


def download_checkpoint(remote_url, local_path):
    from tqdm import tqdm
    import requests

    response = requests.get(remote_url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(
        desc=f"Downloading {local_path}..",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(local_path, "wb") as ref:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            ref.write(data)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


for remote_path, local_path in [
    (checkpoint_url, local_checkpoint_path),
    (prototxt_url, local_prototxt_path),
    (ssd_checkpoint_url, local_ssd_checkpoint_path),
]:
    if not os.path.exists(local_path):
        print(f"{local_path} does not exists!")
        download_checkpoint(remote_url=remote_path, local_path=local_path)


def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image


net = cv2.dnn.readNetFromCaffe(
    prototxt=local_prototxt_path,
    caffeModel=local_ssd_checkpoint_path,
)


transform = transforms.Compose(
    transforms=[transforms.ToPILImage(), transforms.ToTensor()]
)


FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


# load configs and set random seed
package_root_dir = os.path.dirname(__file__)
config_path = os.path.join(package_root_dir, "configs/fer2013_config.json")
with open(config_path) as ref:
    configs = json.load(ref)

image_size = (configs["image_size"], configs["image_size"])

model = resmasking_dropout1(in_channels=3, num_classes=7)
model.cuda()

state = torch.load(local_checkpoint_path)
model.load_state_dict(state["net"])
model.eval()


def video_demo():
    vid = cv2.VideoCapture(0)

    with torch.no_grad():
        while True:
            ret, frame = vid.read()
            if frame is None or ret is not True:
                continue

            try:
                frame = np.fliplr(frame).astype(np.uint8)
                # frame += 50
                h, w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # gray = frame

                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)),
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0),
                )
                net.setInput(blob)
                faces = net.forward()

                for i in range(0, faces.shape[2]):
                    confidence = faces[0, 0, i, 2]
                    if confidence < 0.5:
                        continue
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    start_x, start_y, end_x, end_y = box.astype("int")

                    # covnert to square images
                    center_x, center_y = (start_x + end_x) // 2, (start_y + end_y) // 2
                    square_length = ((end_x - start_x) + (end_y - start_y)) // 2 // 2

                    square_length *= 1.1

                    start_x = int(center_x - square_length)
                    start_y = int(center_y - square_length)
                    end_x = int(center_x + square_length)
                    end_y = int(center_y + square_length)

                    cv2.rectangle(
                        frame, (start_x, start_y), (end_x, end_y), (179, 255, 179), 2
                    )

                    face = gray[start_y:end_y, start_x:end_x]

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
                        f"{emo_label}: 000", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )

                    cv2.rectangle(
                        frame,
                        (end_x, start_y + 1 - label_size[1]),
                        (end_x + label_size[0], start_y + 1 + base_line),
                        (223, 128, 255),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        frame,
                        f"{emo_label} {int(emo_proba * 100)}",
                        (end_x, start_y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        2,
                    )

                cv2.imshow("disp", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

            except:
                continue
        cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
