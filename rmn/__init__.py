import glob
import json
import os

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torchvision.transforms import transforms

from models import densenet121, resmasking_dropout1

from .version import __version__


def show(img, name="disp", width=1000):
    """
    name: name of window, should be name of img
    img: source of img, should in type ndarray
    """
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(name, width, 1000)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


hf_repo_id = "phamquiluan/ResidualMaskingNetwork"
checkpoint_filename = "Z_resmasking_dropout1_rot30_2019Nov30_13.32"
yunet_checkpoint_filename = "face_detection_yunet_2023mar.onnx"

# pre-downloaded files in the working directory take precedence,
# otherwise checkpoints are fetched from the Hugging Face Hub cache
local_checkpoint_path = "pretrained_ckpt"
local_yunet_checkpoint_path = yunet_checkpoint_filename

if not os.path.exists(local_checkpoint_path):
    local_checkpoint_path = hf_hub_download(hf_repo_id, checkpoint_filename)

if not os.path.exists(local_yunet_checkpoint_path):
    local_yunet_checkpoint_path = hf_hub_download(hf_repo_id, yunet_checkpoint_filename)


def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image


def ensure_gray(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    return image


def get_yunet_face_detector():
    yunet_face_detector = cv2.FaceDetectorYN.create(
        model=local_yunet_checkpoint_path,
        config="",
        input_size=(320, 320),
        score_threshold=0.5,
    )
    return yunet_face_detector


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

is_cuda = torch.cuda.is_available()

# load configs and set random seed
package_root_dir = os.path.dirname(__file__)
config_path = os.path.join(package_root_dir, "configs/fer2013_config.json")
with open(config_path) as ref:
    configs = json.load(ref)

image_size = (configs["image_size"], configs["image_size"])


def get_emo_model():
    emo_model = resmasking_dropout1(in_channels=3, num_classes=7)
    if is_cuda:
        emo_model.cuda(0)
    state = torch.load(local_checkpoint_path, map_location="cpu")
    emo_model.load_state_dict(state["net"])
    emo_model.eval()
    return emo_model


def convert_to_square(xmin, ymin, xmax, ymax):
    # convert to square location
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    square_length = ((xmax - xmin) + (ymax - ymin)) // 2 // 2
    square_length *= 1.1

    xmin = int(center_x - square_length)
    ymin = int(center_y - square_length)
    xmax = int(center_x + square_length)
    ymax = int(center_y + square_length)
    return xmin, ymin, xmax, ymax


class RMN:
    def __init__(self, face_detector=True):
        if face_detector is True:
            self.face_detector = get_yunet_face_detector()
        self.emo_model = get_emo_model()

    @torch.no_grad()
    def detect_emotion_for_single_face_image(self, face_image):
        """
        Params:
        -----------
        face_image : np.ndarray
            a cropped face image

        Return:
        -----------
        emo_label : str
            dominant emotion label

        emo_proba : float
            dominant emotion proba

        proba_list : list
            all emotion label and their proba
        """
        assert isinstance(face_image, np.ndarray)
        face_image = ensure_color(face_image)
        face_image = cv2.resize(face_image, image_size)

        face_image = transform(face_image)
        if is_cuda:
            face_image = face_image.cuda(0)

        face_image = torch.unsqueeze(face_image, dim=0)

        output = torch.squeeze(self.emo_model(face_image), 0)
        proba = torch.softmax(output, 0)

        # get dominant emotion
        emo_proba, emo_idx = torch.max(proba, dim=0)
        emo_idx = emo_idx.item()
        emo_proba = emo_proba.item()
        emo_label = FER_2013_EMO_DICT[emo_idx]

        # get proba for each emotion
        proba = proba.tolist()
        proba_list = []
        for emo_idx, emo_name in FER_2013_EMO_DICT.items():
            proba_list.append({emo_name: proba[emo_idx]})

        return emo_label, emo_proba, proba_list

    @torch.no_grad()
    def video_demo(self):
        vid = cv2.VideoCapture(0)

        while True:
            ret, frame = vid.read()
            if frame is None or ret is not True:
                continue

            try:
                frame = np.fliplr(frame).astype(np.uint8)

                results = self.detect_emotion_for_single_frame(frame)
                frame = self.draw(frame, results)

                cv2.rectangle(frame, (1, 1), (220, 25), (223, 128, 255), cv2.FILLED)
                cv2.putText(
                    frame,
                    f"press q to exit",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2,
                )
                cv2.imshow("disp", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

            except Exception as err:
                print(err)
                continue

        cv2.destroyAllWindows()

    @staticmethod
    def draw(frame, results):
        """
        Params:
        ---------
        frame : np.ndarray

        results : list of dict.keys('xmin', 'xmax', 'ymin', 'ymax', 'emo_label', 'emo_proba')

        Returns:
        ---------
        frame : np.ndarray
        """
        for r in results:
            xmin = r["xmin"]
            xmax = r["xmax"]
            ymin = r["ymin"]
            ymax = r["ymax"]
            emo_label = r["emo_label"]
            emo_proba = r["emo_proba"]

            label_size, base_line = cv2.getTextSize(
                f"{emo_label}: 000", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )

            # draw face
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (179, 255, 179), 2)

            cv2.rectangle(
                frame,
                (xmax, ymin + 1 - label_size[1]),
                (xmax + label_size[0], ymin + 1 + base_line),
                (223, 128, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                f"{emo_label} {int(emo_proba * 100)}",
                (xmax, ymin + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )

        return frame

    def detect_faces(self, frame):
        frame = ensure_color(frame)
        h, w = frame.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(frame)

        face_results = []
        if faces is None:
            return face_results
        for face in faces:
            xmin, ymin, bw, bh = face[:4].astype("int")
            xmin, ymin, xmax, ymax = convert_to_square(
                xmin, ymin, xmin + bw, ymin + bh
            )
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            if xmax <= xmin or ymax <= ymin:
                continue

            face_results.append(
                {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )
        return face_results

    @torch.no_grad()
    def detect_emotion_for_single_frame(self, frame):
        gray = ensure_gray(frame)

        results = []
        face_results = self.detect_faces(frame)
        print(f"num faces: {len(face_results)}")

        for face in face_results:
            xmin = face["xmin"]
            ymin = face["ymin"]
            xmax = face["xmax"]
            ymax = face["ymax"]

            face_image = gray[ymin:ymax, xmin:xmax]

            if face_image.shape[0] < 10 or face_image.shape[1] < 10:
                continue
            (
                emo_label,
                emo_proba,
                proba_list,
            ) = self.detect_emotion_for_single_face_image(face_image)

            results.append(
                {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "emo_label": emo_label,
                    "emo_proba": emo_proba,
                    "proba_list": proba_list,
                }
            )
        return results
