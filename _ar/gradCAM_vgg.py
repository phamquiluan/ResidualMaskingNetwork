import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse


EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


class FeatureExtractor:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model):
        self.model = model
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    """
    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x
    """

    def __call__(self, x):
        outputs = []
        self.gradients = []

        x = self.model.conv1(x)  # 112
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)  # 56

        x = self.model.layer1(x)  # 56
        m = self.model.mask1(x)
        x = x * (1 + m)

        x = self.model.layer2(x)  # 28

        x.register_hook(self.save_gradient)

        m = self.model.mask2(x)
        x = x * (1 + m)

        x = self.model.layer3(x)  # 14
        m = self.model.mask3(x)
        x = x * (1 + m)

        x = self.model.layer4(x)  # 7

        m = self.model.mask4(x)
        x = x * (1 + m)

        # CHO NO QUAY LEN O LOP CUOI CUNG,
        # KHONG BIET CHO NAY CO LAY GRAD DUOC KHONG

        outputs += [x]
        return outputs, x
        """
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x
        """


class ModelOutputs:
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        # output = output.view(output.size(0), -1)
        output = self.model.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.model.fc(output)
        # print(torch.argmax(output, 1).item())
        return target_activations, output


def show_cam_on_image(img, mask, image_name=""):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input), input, positive_mask
        )
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1
            ),
            positive_mask_2,
        )

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == "ReLU":
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        # one_hot.backward(retain_variables=True)
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Use NVIDIA GPU acceleration",
    )
    parser.add_argument(
        "--image-path", type=str, default="./examples/both.png", help="Input image path"
    )
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


if __name__ == "__main__":
    """python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization."""

    args = get_args()

    from models.vgg import vgg19

    model = vgg19()
    state = torch.load("./saved/checkpoints/vgg19_rot30_2019Dec01_14.01")
    model.load_state_dict(state["net"])

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    """
    grad_cam = GradCam(
        model=models.vgg19(pretrained=True),
        target_layer_names = ["35"],
        use_cuda=args.use_cuda
    )
    """
    grad_cam = GradCam(model=model, target_layer_names=["35"], use_cuda=args.use_cuda)

    from torchvision.transforms import transforms

    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    import os
    import glob
    from natsort import natsorted

    for image_path in natsorted(
        glob.glob("/home/z/research/archive_for_tee/face_CK+/only_face/*.png")
    ):
        image_name = os.path.basename(image_path)
        print(image_name)
        image = cv2.imread(image_path)
        # image = cv2.imread('sample.png')
        image = cv2.resize(image, (224, 224))
        tensor = transform(image)
        tensor = torch.unsqueeze(tensor, 0)

        # img = np.float32(cv2.resize(img, (224, 224))) / 255
        # input = preprocess_image(img)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None

        mask = grad_cam(tensor, target_index)

        image = np.float32(cv2.resize(image, (224, 224))) / 255
        # show_cam_on_image(image, mask, image_name='')

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(image)
        cam = cam / np.max(cam)
        # cv2.imwrite("cam.jpg", np.uint8(255 * cam))
        cv2.imwrite("./resmasking_gradcam/{}".format(image_name), np.uint8(255 * cam))
