import json
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

BARBER_CLASS_MAP = {
    'Code 128': 0, 'Code 39': 0, 'EAN-2': 0, 'EAN-8': 0, 'EAN-13': 0,
    'GS1-128': 0, 'IATA 2 of 5': 0, 'Intelligent Mail Barcode': 0,
    'Interleaved 2 of 5': 0, 'Japan Postal Barcode': 0, 'KIX-code': 0,
    'PostNet': 0, 'RoyalMail Code': 0, 'UPC': 0, '1D': 0,
    'Aztec': 1, 'Datamatrix': 1, 'PDF-417': 1, 'QR Code': 1
}

def parseBarBeRJson(datasetPath='Dataset/BarBeR'):
    datasetPath = Path(datasetPath)
    imageDir = datasetPath / 'Images'
    annotDir = datasetPath / 'Annotations'

    samples = []

    for jsonPath in sorted(annotDir.glob('*.json')):
        with open(jsonPath) as f:
            data = json.load(f)

        for entry in data.values():
            filename = entry['filename']
            imagePath = imageDir / filename

            if not imagePath.exists():
                continue

            objects = []
            for region in entry['regions']:
                shapeAttr = region['shape_attributes']
                regionAttr = region['region_attributes']

                barcodeType = regionAttr.get('Type', '1D')
                classNum = BARBER_CLASS_MAP.get(barcodeType, 0)

                xs = shapeAttr['all_points_x']
                ys = shapeAttr['all_points_y']

                objects.append({
                    'class': classNum,
                    'xmin': min(xs),
                    'ymin': min(ys),
                    'xmax': max(xs),
                    'ymax': max(ys)
                })

            if objects:
                samples.append({'imagePath': imagePath, 'objects': objects})

    return samples

def encodeLabelGrid(objects, imgW, imgH, S=8):
    target = torch.zeros((S, S, 6))

    for obj in objects:
        cx = (obj['xmin'] + obj['xmax']) / 2 / imgW
        cy = (obj['ymin'] + obj['ymax']) / 2 / imgH
        w  = (obj['xmax'] - obj['xmin']) / imgW
        h  = (obj['ymax'] - obj['ymin']) / imgH

        cellX = min(int(cx * S), S - 1)
        cellY = min(int(cy * S), S - 1)

        offsetX = cx * S - cellX
        offsetY = cy * S - cellY

        # only write if cell is empty (first object wins on collision)
        if target[cellY, cellX, 0] == 0:
            target[cellY, cellX] = torch.tensor([
                1.0, offsetX, offsetY, w, h, float(obj['class'])
            ])

    return target

class barcodeDataset(Dataset):
    def __init__(self, samples, S=8):
        self.samples = samples
        self.S = S

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = cv2.imread(str(sample['imagePath']))
        imgH, imgW = img.shape[:2]
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        target = encodeLabelGrid(sample['objects'], imgW, imgH, self.S)

        return img, target

def buildDataloaders(batchSize=32, testSize=0.2, randomState=43, S=8):
    samples = parseBarBeRJson()

    random.seed(randomState)
    random.shuffle(samples)

    splitIdx = int(len(samples) * (1 - testSize))
    trainSamples = samples[:splitIdx]
    valSamples = samples[splitIdx:]

    print(f'Training samples: {len(trainSamples)}, Validation samples: {len(valSamples)}')

    trainDataset = barcodeDataset(trainSamples, S=S)
    valDataset = barcodeDataset(valSamples, S=S)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)

    return trainLoader, valLoader, len(valDataset)