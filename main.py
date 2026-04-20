import argparse
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from dataset import buildDataloaders
from model import GridDetectionNet
from train import train
from inference import loadModel, runInference, decodeCrop, parseDecodeString, computeIou
from telemetry import telemetry

def drawDetections(frame, detections, decoded=None):
    for i, (label, (x1, y1, x2, y2), conf, angle) in enumerate(detections):
        summary = decoded[i] if decoded else None
        overlayText = f"{label} {conf:.2f}" if summary is None else f"{label} {conf:.2f} | {summary}"

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        box = cv2.boxPoints(((cx, cy), (w, h), angle))
        box = np.int32(box)
        cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, overlayText, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def singleImage(imagePath, modelPath=None):
    model = loadModel(modelPath)
    frame = cv2.imread(imagePath)
    detections, latencyMs = runInference(model, frame)

    print(f"Detections: {len(detections)}  |  Latency: {latencyMs:.1f} ms")
    decoded = []
    for label, bbox, conf, angle in detections:
        text = decodeCrop(frame, bbox)
        summary = parseDecodeString(text)
        decoded.append(summary)
        print(f"  {label}  conf={conf:.3f}  bbox={bbox}  decoded={text}")

    frame = drawDetections(frame, detections, decoded)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def liveCamera(modelPath=None):
    model = loadModel(modelPath)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera")
        return

    cache = {}   # tracks known barcodes across frames: id -> {bbox, label, conf, text, framesMissing}
    nextId = 0
    iouEvictThreshold = 0.3  # min IoU to consider a detection the same barcode as a cached entry
    maxFramesMissing = 10    # frames a cached entry survives without being matched before eviction
    tel = telemetry()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, latencyMs = runInference(model, frame, confThreshold=0.3)  # low threshold to catch candidates for decoding

        displayDetections = [d for d in detections if d[2] >= 0.5]  # only render boxes above 0.5 to avoid noisy overlays
        tel.recordFrame(latencyMs, displayDetections)

        for entry in cache.values():
            entry['framesMissing'] += 1

        decoded = []
        for label, bbox, conf, angle in displayDetections:
            matched = None
            for entry in cache.values():
                if computeIou(bbox, entry['bbox']) >= iouEvictThreshold:
                    matched = entry
                    break

            if matched is not None:
                matched['bbox'] = bbox
                matched['label'] = label
                matched['conf'] = conf
                matched['framesMissing'] = 0
                if matched['text'] is None:  # only attempt decode once - avoids repeated zxingcpp calls on the same barcode
                    matched['text'] = decodeCrop(frame, bbox)
                    if matched['text'] is not None:
                        tel.markDecoded(label, conf)
                decoded.append(parseDecodeString(matched['text']))
            else:
                text = decodeCrop(frame, bbox)
                if text is not None:
                    tel.markDecoded(label, conf) 
                cache[nextId] = {
                    'bbox': bbox, 'label': label, 'conf': conf,
                    'text': text, 'framesMissing': 0
                }
                nextId += 1
                decoded.append(parseDecodeString(text))

        cache = {k: v for k, v in cache.items() if v['framesMissing'] <= maxFramesMissing}  # evict stale entries

        frame = drawDetections(frame, displayDetections, decoded)
        cv2.putText(frame, f"{latencyMs:.1f} ms", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Barcode Scanner", frame)
        key = cv2.waitKey(1) & 0xFF  # & 0xFF masks to 8 bits for cross-platform key code compatibility
        windowClosed = cv2.getWindowProperty("Barcode Scanner", cv2.WND_PROP_VISIBLE) < 1  # detect window X button
        if key == ord('q') or windowClosed:
            break

    cap.release()
    cv2.destroyAllWindows()
    tel.report() 

def trainModel(cache=False, datasetPath='Dataset'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainLoader, valLoader = buildDataloaders(cache=cache, datasetPath=datasetPath)
    model = GridDetectionNet().to(device)
    config = {
        'numEpochs': 200,
        'lr': 0.005,
        'lambdaCoord': 5.0,
        'lambdaNoobj': 0.5,
        'lambdaAngle': 0.125,
        'weightDecay': 1e-3,
        'gradClipNorm': 10,
        'etaMin': 1e-6,
        'batchSize': 64,
        'tMax': 200,
        'earlyStoppingPatience': 40,
        'momentum': 0.9,
    }
    train(model, trainLoader, valLoader, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Barcode Scanner")
    parser.add_argument('mode', choices=['image', 'camera', 'train'])
    parser.add_argument('--imagePath', type=str, default=None)
    parser.add_argument('--modelPath', type=str, default=None)
    parser.add_argument('--cache', action='store_true', default=False)
    args = parser.parse_args()

    if args.mode == 'image':
        if args.imagePath is None:
            parser.error("--imagePath required for image mode")
        singleImage(args.imagePath, args.modelPath)
    elif args.mode == 'camera':
        liveCamera(args.modelPath)
    elif args.mode == 'train':
        trainModel(cache=args.cache, datasetPath='Dataset')


