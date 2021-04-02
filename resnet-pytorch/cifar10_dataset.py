from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import numpy as np


transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

images, labels = next(iter(trainloader))
image = images[0] / 2 + .5 # Unormalize the image
image = image.numpy()
image = np.transpose(image, (1, 2, 0))
image *= 255.0
image = image.astype(np.uint8)
label = labels[0].item()

print(image.shape, label)
print(type(image))