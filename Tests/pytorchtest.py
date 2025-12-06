import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchinfo

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                              shuffle=True)

testDataloader = DataLoader(dataset=testData,
                             batch_size = batchSize,
                             shuffle=False)

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
    

torch.manual_seed(74)
model0 = TinyVGG(inputShape=3,
                 hiddenUnits=10,
                 outputShape=len(trainData.classes)).to(device)

# 1. Get a batch of images and labels from the DataLoader
imgBatch, labelBatch = next(iter(trainDataloader))

# 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
imgSingle, labelSingle = imgBatch[0].unsqueeze(dim=0), labelBatch[0]
print(f"Single image shape: {imgSingle.shape}\n")

# 3. Perform a forward pass on a single image
model0.eval()
with torch.inference_mode():
    pred = model0(imgSingle.to(device))
    
# 4. Print out what's happening and convert model logits -> pred probs -> pred label
print(f"Output logits:\n{pred}\n")
print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{labelSingle}")
