import cv2
import time
import numpy as np
import torch
from pathlib import Path
from model import FlexibleDetectionNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
numToLabels = {0: "barcode", 1: "qr"}

def loadModel():
    modelPath = Path(__file__).parent / "models" / "bestModel.pth"
    model = FlexibleDetectionNet(in_channels=3, num_classes=2, hidden_units=12)
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess(bgrFrame, imageSize=256):
    image = cv2.resize(bgrFrame, (imageSize, imageSize))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def postprocess(results, origH, origW):
    [classProbs, boundingBox] = results
    classIndex = torch.argmax(classProbs)
    label = numToLabels[classIndex.item()]
    confidence = torch.max(classProbs).item() * 100

    x1, y1, x2, y2 = boundingBox[0]
    x1 = int(origW * x1)
    x2 = int(origW * x2)
    y1 = int(origH * y1)
    y2 = int(origH * y2)

    return label, (x1, y1, x2, y2), confidence

def runInference(model, bgrFrame):
    origH, origW = bgrFrame.shape[:2]
    processed = preprocess(bgrFrame)
    tensor = torch.permute(torch.from_numpy(processed).float(), (0, 3, 1, 2)).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        results = model(tensor)
    latencyMs = (time.perf_counter() - start) * 1000

    label, bbox, confidence = postprocess(results, origH, origW)
    return label, bbox, confidence, latencyMs

def predict(imagePath):
    model = loadModel()
    img = cv2.imread(str(imagePath))
    label, (x1, y1, x2, y2), confidence, latencyMs = runInference(model, img)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 2)
    print(f"{label}, CONFIDENCE: {confidence:.1f}%, LATENCY: {latencyMs:.1f}ms")

    origH, origW = img.shape[:2]
    scale = min(800 / origW, 600 / origH)
    displayImg = cv2.resize(img, (int(origW * scale), int(origH * scale)))
    cv2.imshow("Prediction", displayImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    imagePath = Path("Dataset/Test/Barcode/EAN13_09_0072.jpg")
    predict(str(imagePath))
