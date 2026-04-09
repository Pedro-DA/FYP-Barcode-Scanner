from sklearn.model_selection import train_test_split
import cv2
import random
import os
import numpy as np
import time
from pathlib import Path

import pandas as pd
from xml.dom import minidom
import csv

import torch
import torch.optim as optim
import torch.nn.functional as F

import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Split the data of images, labels and their annotations
trainImages, valImages, trainLabels, \
valLabels, trainBoxes, valBoxes = train_test_split( np.array(imgList), np.array(labels), np.array(boxes), test_size = 0.2, random_state = 43)

print('Training Images Count: {}, Validation Images Count: {}'.format(len(trainImages), len(valImages) ))

class Dataset():
    def __init__(self, trainImages, trainLabels, trainBoxes):
        self.images = torch.permute(torch.from_numpy(trainImages),(0,3,1,2)).float()
        self.labels = torch.from_numpy(trainLabels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(trainBoxes).float()

    def __len__(self):
        return len(self.labels)

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        return (self.images[idx],
              self.labels[idx],
              self.boxes[idx])

# Inheriting from Dataset class

class ValDataset(Dataset):

    def __init__(self, valImages, valLabels, valBoxes):

        self.images = torch.permute(torch.from_numpy(valImages),(0,3,1,2)).float()
        self.labels = torch.from_numpy(valLabels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(valBoxes).float()

dataset = Dataset(trainImages, trainLabels, trainBoxes)
valdataset = ValDataset(valImages, valLabels, valBoxes)

def getNumCorrect(preds, labels):
    return torch.round(preds).argmax(dim=1).eq(labels).sum().item()

dataloader = torch.utils.data.DataLoader(
       dataset, batch_size=32, shuffle=True)
valDataloader = torch.utils.data.DataLoader(
       valdataset, batch_size=32, shuffle=True)

def train(model):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1
    )

    numEpochs = 30
    lambdaClass = 1.0
    lambdaBox = 0.5
    epochs = []
    losses = []
    bestLoss = float('inf')
    valDataSize = len(valDataloader.dataset)

    os.makedirs('models', exist_ok=True)

    for epoch in range(numEpochs):
        trainTotalLoss = 0.0
        valTotalLoss = 0.0
        totalCorrect = 0
        trainStart = time.time()

        # Training phase
        model.train()
        for batch, (x, y, z) in enumerate(dataloader):
            x, y, z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()

            [yPred, zPred] = model(x)

            classLoss = F.cross_entropy(yPred, y)
            boxLoss = F.mse_loss(zPred, z)
            batchLoss = (lambdaClass * classLoss) + (lambdaBox * boxLoss)
            batchLoss.backward()

            optimizer.step()
            trainTotalLoss += batchLoss.item()
            print(f"Train batch: {batch+1} epoch: {epoch}",
                  f"time: {(time.time()-trainStart)/60:.2f}mins", end='\r')

        # Validation phase
        model.eval()
        for batch, (x, y, z) in enumerate(valDataloader):
            x, y, z = x.to(device), y.to(device), z.to(device)

            with torch.no_grad():
                [yPred, zPred] = model(x)
                class_loss = F.cross_entropy(yPred, y)
                box_loss = F.mse_loss(zPred, z)

            valTotalLoss += (lambdaClass * class_loss.item()) + (lambdaBox * box_loss.item())
            totalCorrect += getNumCorrect(yPred, y)
            print(f"Val batch: {batch+1} epoch: {epoch}",
                  f"time: {(time.time()-trainStart)/60:.2f}mins", end='\r')

        # Epoch end
        accuracy = (totalCorrect / valDataSize) * 100
        epochs.append(epoch)
        losses.append(valTotalLoss)

        print(f"Epoch {epoch+1} | "
              f"Train Loss: {trainTotalLoss:.4f} | "
              f"Val Loss: {valTotalLoss:.4f} | "
              f"Accuracy: {accuracy:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']} | "
              f"Time: {(time.time()-trainStart)/60:.2f}mins")

        scheduler.step(valTotalLoss)

        if valTotalLoss < bestLoss:
            bestLoss = valTotalLoss
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"  -> Model improved, saved.")


model = Model.FlexibleDetectionNet(in_channels=3, num_classes=2, hidden_units=12)
model = model.to(device)
train(model)