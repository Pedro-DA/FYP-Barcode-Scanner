from sklearn.model_selection import train_test_split
import cv2
import random
import numpy as np
from pathlib import Path

import pandas as pd
from xml.dom import minidom
import csv

import torch

def xmlToCsv():
    csvPath = Path("Dataset/Kaggle/dataset.csv")
    if csvPath.exists():
        return pd.read_csv(csvPath)   #use cache if available

    xmlList = []
    path = Path("Dataset/Kaggle")
    annotDir = path / "Annotations"
    imageDir = path / "Images"

    annotFiles = sorted(annotDir.iterdir())
    imageFiles = sorted(imageDir.iterdir())

    for annotPath, imagePath in zip(annotFiles, imageFiles):
        value = extractXmlContents(annotPath, imagePath)
        xmlList.append(value)

    columnName = ['filename', 'width', 'height', 'class_num', 'xmin', 'ymin', 'xmax', 'ymax']
    xmlDf = pd.DataFrame(xmlList, columns=columnName)
    xmlDf.to_csv(csvPath, index=None)   #save cache for next time
    return xmlDf

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

def preprocessDataset():
    labels = []
    boxes = []
    imgList = []
    xmlToCsv()  # generates CSV if it doesn't exist, uses cache if it does
    imageDir = Path("Dataset/Kaggle/Images")
    csvPath = Path("Dataset/Kaggle/dataset.csv")

    with csvPath.open() as csvfile:
        rows = csv.reader(csvfile)
        columns = next(rows)
        for row in rows:
            labels.append(int(row[3]))

            imgWidth = float(row[1])
            imgHeight = float(row[2])

            arr = [float(row[4]) / imgWidth,   # xmin
                   float(row[5]) / imgHeight,  # ymin
                   float(row[6]) / imgWidth,   # xmax
                   float(row[7]) / imgHeight]  # ymax
            boxes.append(arr)

            imgPath = imageDir / row[0]
            img = cv2.imread(str(imgPath))
            image = cv2.resize(img, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float") / 255.0
            imgList.append(image)

    return labels, boxes, imgList

class barcodeDataset(torch.utils.data.Dataset):
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

class barcodeValDataset(barcodeDataset):
    def __init__(self, valImages, valLabels, valBoxes):
        self.images = torch.permute(torch.from_numpy(valImages),(0,3,1,2)).float()
        self.labels = torch.from_numpy(valLabels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(valBoxes).float()

def buildDataloaders(batchSize=32, testSize=0.2, randomState=43):
    labels, boxes, imgList = preprocessDataset()

    combinedList = list(zip(imgList, boxes, labels))
    random.shuffle(combinedList)
    imgList, boxes, labels = zip(*combinedList)

    trainImages, valImages, trainLabels, valLabels, trainBoxes, valBoxes = train_test_split(
        np.array(imgList), np.array(labels), np.array(boxes),
        test_size=testSize, random_state=randomState
    )

    print(f'Training Images: {len(trainImages)}, Validation Images: {len(valImages)}')

    trainDataset = barcodeDataset(trainImages, trainLabels, trainBoxes)
    valDataset = barcodeValDataset(valImages, valLabels, valBoxes)

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    valLoader = torch.utils.data.DataLoader(valDataset, batch_size=batchSize, shuffle=True)
    valDataSize = len(valDataset)

    return trainLoader, valLoader, valDataSize