import argparse
import cv2
import matplotlib.pyplot as plt
import torch

from dataset import buildDataloaders
from model import GridDetectionNet
from train import train
from inference import loadModel, runInference

def drawDetections(frame, detections):
    for label, (x1, y1, x2, y2), conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def singleImage(imagePath, modelPath=None):
    model = loadModel(modelPath)
    frame = cv2.imread(imagePath)
    detections, latencyMs = runInference(model, frame)

    print(f"Detections: {len(detections)}  |  Latency: {latencyMs:.1f} ms")
    for label, bbox, conf in detections:
        print(f"  {label}  conf={conf:.3f}  bbox={bbox}")

    frame = drawDetections(frame, detections)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def liveCamera(modelPath=None):
    model = loadModel(modelPath)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, latencyMs = runInference(model, frame)
        frame = drawDetections(frame, detections)

        cv2.putText(frame, f"{latencyMs:.1f} ms", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Barcode Scanner", frame)
        key = cv2.waitKey(1) & 0xFF
        windowClosed = cv2.getWindowProperty("Barcode Scanner", cv2.WND_PROP_VISIBLE) < 1
        if key == ord('q') or windowClosed:
            break

    cap.release()
    cv2.destroyAllWindows()

def trainModel(cache=False, datasetPath='Dataset/BarBeR'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainLoader, valLoader = buildDataloaders(cache=cache, datasetPath=datasetPath)
    model = GridDetectionNet().to(device)
    config = {
        'numEpochs': 100,
        'lr': 0.005,
        'lambdaCoord': 5.0,
        'lambdaNoobj': 0.5,
        'batchSize': 64,
        'tMax': 30,
        'earlyStoppingPatience': 20,
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
        trainModel(cache=args.cache, datasetPath='Dataset/BarBeR')


