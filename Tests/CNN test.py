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

path = Path("Dataset/Kaggle")
annotDir = path / "Annotations"
imageDir = path / "Images"

testImage = imageDir / "05102009081.jpg"
testAnnot = annotDir / "05102009081.xml"

print(extractXmlContents(testAnnot, testImage))