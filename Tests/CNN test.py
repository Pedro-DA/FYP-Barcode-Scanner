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