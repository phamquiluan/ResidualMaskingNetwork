import numpy as np
import torch
from torchvision import transforms


image = np.random.rand(48, 48, 1)
image = image * 255
image = image.astype(np.uint8)

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

tensor = transform(image)
tensor = torch.unsqueeze(tensor, 0)


from fer2013_models import BaseNet

model = BaseNet()
print(model(tensor))
