import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
image_path_list = list(image_path.glob("*/*/*.jpg"))

data_tranform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

train_data=datasets.ImageFolder(root=train_dir,
                                transform=data_tranform,
                                target_transform=None)

test_data=datasets.ImageFolder(root=test_dir,
                                transform=data_tranform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")


