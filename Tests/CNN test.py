import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import cv2
import random
import os

from PIL import Image
from pathlib import Path

import pandas as pd
from xml.dom import minidom
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary

def extractXmlContents(annotDir, imageDir):
    file = minidom.parse(str(annotDir))
    
    # Get the height and width for our image
    height, width = cv2.imread(str(imageDir)).shape[:2]
        
    #bounding box co-ordinates 
    xmin = file.getElementsByTagName('xmin')
    x1 = float(xmin[0].firstChild.data)

    ymin = file.getElementsByTagName('ymin')
    y1 = float(ymin[0].firstChild.data)

    xmax = file.getElementsByTagName('xmax')
    x2 = float(xmax[0].firstChild.data)

    ymax = file.getElementsByTagName('ymax')
    y2 = float(ymax[0].firstChild.data)

    class_name = file.getElementsByTagName('name')

    if class_name[0].firstChild.data == "barcode":
        class_num = 0
    else:
        class_num = 1

    files = file.getElementsByTagName('filename')
    filename = files[0].firstChild.data

    # Return the extracted attributes
    return filename,  width, height, class_num, x1,y1,x2,y2


numToLabels = {0:"barcode",1:"qr"}

def xmlToCsv():
    # List containing all our attributes regarding each image
    xmlList = []

    path = Path("Dataset/Kaggle")
    annotDir = path / "Annotations"
    imageDir = path / "Images"

    # Get each file using iterdir() and sort to ensure matching order
    annotfiles = sorted(annotDir.iterdir())
    imagefiles = sorted(imageDir.iterdir())

    # Loop over each image and its label
    for annotPath, imagePath in zip(annotfiles, imagefiles):
        value = extractXmlContents(annotPath, imagePath)
        xmlList.append(value)

    # Columns for Pandas DataFrame
    columnName = ['filename', 'width', 'height', 'class_num', 'xmin', 'ymin', 'xmax', 'ymax']

    # Create the DataFrame from mat_list
    xmlDf = pd.DataFrame(xmlList, columns=columnName)

    # Return the dataframe
    return xmlDf

# The Classes we will use for our training
classesList = sorted(['cat',  'dog'])

if not Path("Dataset/Kaggle/dataset.csv").exists():
    # Run the function to convert all the xml files to a Pandas DataFrame
    labelsDf = xmlToCsv()

    # Saving the Pandas DataFrame as CSV File
    labelsDf.to_csv(('Dataset/Kaggle/dataset.csv'), index=None)

def preprocessDataset():
    labels = []
    boxes = []
    imgList = []
    imageDir = Path("Dataset/Kaggle/Images")
    csv_path = Path("Dataset/Kaggle/dataset.csv")

    with csv_path.open() as csvfile:
        rows = csv.reader(csvfile)
        columns = next(rows)
        for row in rows:
            labels.append(int(row[3]))

            img_width = float(row[1])
            img_height = float(row[2])

            arr = [float(row[4]) / img_width,   # xmin
                   float(row[5]) / img_height,  # ymin
                   float(row[6]) / img_width,   # xmax
                   float(row[7]) / img_height]  # ymax
            boxes.append(arr)

            img_path = imageDir / row[0]
            img = cv2.imread(str(img_path))
            image = cv2.resize(img, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float") / 255.0
            imgList.append(image)

    return labels, boxes, imgList

# All images will resized to 300, 300 
image_size = 256

# Get Augmented images and bounding boxes
labels, boxes, imgList = preprocessDataset()

# Now we need to shuffle the data, so zip all lists and shuffle
combinedList = list(zip(imgList, boxes, labels))
random.shuffle(combinedList)

# Extract back the contents of each list
imgList, boxes, labels = zip(*combinedList)

class FlexibleDetectionNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_units: int = 64):
        super().__init__()

        # Shared CNN Backbone
        self.backbone = nn.Sequential(
            self._conv_block(in_channels,  hidden_units),        # block 1
            self._conv_block(hidden_units, hidden_units * 2),    # block 2
            self._conv_block(hidden_units * 2, hidden_units * 4),# block 3
            self._conv_block(hidden_units * 4, hidden_units * 8),# block 4
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))  # forces a fixed output size regardless of input

        flat_features = hidden_units * 8 * 4 * 4

        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(flat_features, 240),
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Linear(120, num_classes),
            nn.Softmax(dim=1)
        )

        # Bounding box head
        self.box_head = nn.Sequential(
            nn.Linear(flat_features, 240),
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Linear(120, 4),
            nn.Sigmoid()
        )

    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)

        return [self.class_head(x), self.box_head(x)]