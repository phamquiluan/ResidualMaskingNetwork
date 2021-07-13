import os
import glob
import json
import cv2
import numpy as np
import torch
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


def ensure_gray(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    return image


def get_ssd_face_detector():
    ssd_face_detector = cv2.dnn.readNetFromCaffe(
        prototxt=local_prototxt_path,
        caffeModel=local_ssd_checkpoint_path,
    )
    return ssd_face_detector


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
            self.face_detector = get_ssd_face_detector()
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
                cv2.putText(frame, f"press q to exit", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
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
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            False, 
            False
        )
        self.face_detector.setInput(blob)
        faces = self.face_detector.forward()

        face_results = []
        for i in range(0, faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence < 0.5:
                continue
            xmin, ymin, xmax, ymax = (faces[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
            xmin, ymin, xmax, ymax = convert_to_square(xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
        
            face_results.append({
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            })
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
            emo_label, emo_proba, proba_list = self.detect_emotion_for_single_face_image(face_image)
            
            results.append({
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "emo_label": emo_label,
                "emo_proba": emo_proba,
                "proba_list": proba_list
            })
        return results


