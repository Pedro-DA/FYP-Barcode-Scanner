import cv2
import time
import torch
from pathlib import Path
from model import GridDetectionNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
numToLabels = {0: "barcode", 1: "qr"}

def loadModel(modelPath, S=8, hiddenUnits=32):
    if modelPath is None:
        modelPath = Path(__file__).parent / "models" / "bestModel.pth"
    model = GridDetectionNet(S=S, hidden_units=hiddenUnits).to(device)
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()
    return model

def preprocess(bgrFrame, imageSize=256):
    image = cv2.resize(bgrFrame, (imageSize, imageSize))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.to(device)

def decodeGrid(output, origH, origW, confThreshold=0.5):
    S = output.shape[1]
    detections = []

    for row in range(S):
        for col in range(S):
            cell = output[0, row, col]  # (6,)
            conf = cell[0].item()
            if conf < confThreshold:
                continue

            cx = (col + cell[1].item()) / S * origW
            cy = (row + cell[2].item()) / S * origH
            w  = cell[3].item() * origW
            h  = cell[4].item() * origH

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            label = "qr" if cell[5].item() > 0.5 else "barcode"
            detections.append((label, (x1, y1, x2, y2), conf))

    return detections

def computeIou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def nms(detections, iouThreshold=0.5):
    detections = sorted(detections, key=lambda d: d[2], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if computeIou(best[1], d[1]) < iouThreshold]
    return kept

def runInference(model, bgrFrame, confThreshold=0.5, iouThreshold=0.5):
    origH, origW = bgrFrame.shape[:2]
    tensor = preprocess(bgrFrame)

    start = time.perf_counter()
    with torch.no_grad():
        output = model(tensor)
    latencyMs = (time.perf_counter() - start) * 1000

    detections = decodeGrid(output, origH, origW, confThreshold)
    detections = nms(detections, iouThreshold)
    return detections, latencyMs
