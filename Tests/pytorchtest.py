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

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=1,
                             shuffle=False)

img, label = next(iter(train_dataloader))
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")


