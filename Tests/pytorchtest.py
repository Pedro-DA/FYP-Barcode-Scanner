import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random
from PIL import Image
import os

# Setup path to data folder
imagePath = Path("Dataset")

# Setup train and testing paths
trainDir = imagePath / "Train"
testDir = imagePath / "Test"

#Visualize images in dataset
imagePathList = list(imagePath.glob("*/*/*.jpg"))

trainTransform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

testTransform = transforms.Compose([#
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

trainData=datasets.ImageFolder(root=trainDir,
                                transform=trainTransform,
                                target_transform=None)

testData=datasets.ImageFolder(root=testDir,
                                transform=testTransform)

batchSize = 6
numWorkers = os.cpu_count()

trainDataloader = DataLoader(dataset=trainData,
                              batch_size = batchSize,
                              shuffle=True,
                              num_workers=numWorkers)

testDataloader = DataLoader(dataset=testData,
                             batch_size = batchSize,
                             shuffle=False,
                             num_workers=numWorkers)

class TinyVGG(nn.Module):
    def __init__(self, inputShape: int,hiddenUnits: int, outputShape: int) -> None:
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=inputShape,
                      out_channels=hiddenUnits,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(hiddenUnits,hiddenUnits,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(hiddenUnits,hiddenUnits,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddenUnits*16*16,
                      out_features=outputShape)
        )
    def forward(self, x: torch.Tensor):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.classifier(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(74)
model0 = TinyVGG(inputShape=3,
                 hiddenUnits=10,
                 outputShape=len(trainData.classes)).to(device)

print(model0)
