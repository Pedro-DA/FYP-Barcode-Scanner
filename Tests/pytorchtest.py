import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random
from PIL import Image

# Setup path to data folder
image_path = Path("Dataset")

# Setup train and testing paths
train_dir = image_path / "Train"
test_dir = image_path / "Test"

#Visualize images in dataset
random.seed(35)

image_path_list = list(image_path.glob("*/*/*.jpg"))

random_image_path = random.choice(image_path_list)

image_class = random_image_path.parent.stem

img = Image.open(random_image_path)

print(random_image_path)
print(image_class)
print(img.height)
print(img.width)


