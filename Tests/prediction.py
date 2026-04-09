import cv2
import numpy as np
import torch
import Model
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(img, image_size = 256):
    
    image = cv2.resize(img, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0 

    # Expand dimensions as predict expect image in batches
    image = np.expand_dims(image, axis=0) 
    return image

def postprocess(image, results, orig_h, orig_w):
    numToLabels = {0:"barcode", 1:"qr"}
    [classProbs, boundingBox] = results
    classIndex = torch.argmax(classProbs)
    classLabel = numToLabels[classIndex.item()]

    h, w = orig_h, orig_w 
    x1, y1, x2, y2 = boundingBox[0]
    x1 = int(w * x1)
    x2 = int(w * x2)
    y1 = int(h * y1)
    y2 = int(h * y2)
    return classLabel, (x1, y1, x2, y2), torch.max(classProbs) * 100

def predict(image, scale=0.5):
    model = Model.FlexibleDetectionNet(in_channels=3, num_classes=2, hidden_units=12)
    model = model.to(device)
    model.load_state_dict(torch.load("models/bestModel.pth"))
    model.eval()

    img = cv2.imread(image)
    orig_h, orig_w = img.shape[:2]

    processed_image = preprocess(img)
    result = model(torch.permute(torch.from_numpy(processed_image).float(), (0,3,1,2)).to(device))

    label, (x1, y1, x2, y2), confidence = postprocess(image, result, orig_h, orig_w)

    # Now annotate the image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 2)
    print('{}, CONFIDENCE: {}'.format(label, confidence))

    # Show the Image with matplotlib
    plt.figure(figsize=(10,10))
    plt.imshow(img[:,:,::-1])
    plt.show()

image = Path("Dataset/Test/Barcode/EAN13_09_0094.jpg")
predict(str(image))