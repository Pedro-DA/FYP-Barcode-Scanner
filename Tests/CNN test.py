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

# Create a Matplotlib figure
plt.figure(figsize=(20,20));

# Generate a random sample of images each time the cell is run 
random_range = random.sample(range(1, len(imgList)), 20)

for itr, i in enumerate(random_range, 1):

    # Bounding box of each image
    a1, b1, a2, b2 = boxes[i];
    img_size = 256

    # Rescaling the boundig box values to match the image size
    x1 = a1 * img_size
    x2 = a2 * img_size
    y1 = b1 * img_size
    y2 = b2 * img_size

    # The image to visualize
    image = imgList[i]

    # Draw bounding boxes on the image
    cv2.rectangle(image, (int(x1),int(y1)),
          (int(x2),int(y2)),
                  (0,255,0),
                  3);
    
    # Clip the values to 0-1 and draw the sample of images
    img = np.clip(imgList[i], 0, 1)
    plt.subplot(4, 5, itr);
    plt.imshow(img);
    plt.axis('off');

plt.show()