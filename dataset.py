import json
import random
from pathlib import Path
import numpy as np
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

def augmentSample(img, objects, imgW, imgH):
    # H-flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        newObjs = []
        for obj in objects:
            newPts = [(imgW - x, y) for x, y in obj['points']]
            xs = [p[0] for p in newPts]; ys = [p[1] for p in newPts]
            newObjs.append({**obj, 'xmin': min(xs), 'xmax': max(xs),
                            'ymin': min(ys), 'ymax': max(ys), 'points': newPts})
        objects = newObjs

    # V-flip
    if random.random() < 0.5:
        img = cv2.flip(img, 0)
        newObjs = []
        for obj in objects:
            newPts = [(x, imgH - y) for x, y in obj['points']]
            xs = [p[0] for p in newPts]; ys = [p[1] for p in newPts]
            newObjs.append({**obj, 'xmin': min(xs), 'xmax': max(xs),
                            'ymin': min(ys), 'ymax': max(ys), 'points': newPts})
        objects = newObjs

    # Rotation +- 30 degrees
    if random.random() < 0.5:
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((imgW / 2, imgH / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (imgW, imgH), borderMode=cv2.BORDER_REPLICATE)
        newObjs = []
        for obj in objects:
            pts = np.array(obj['points'], dtype=np.float32)
            pts_h = np.column_stack([pts, np.ones(len(pts))])
            rotPts = (M @ pts_h.T).T.tolist()
            xs = [p[0] for p in rotPts]; ys = [p[1] for p in rotPts]
            newObjs.append({**obj, 'xmin': max(0, min(xs)), 'xmax': min(imgW, max(xs)),
                            'ymin': max(0, min(ys)), 'ymax': min(imgH, max(ys)), 'points': rotPts})
        objects = newObjs

    # Colour jitter — image only, no point transforms
    if random.random() < 0.5:
        factor = random.uniform(0.6, 1.4)   # brightness
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return img, objects

def parseBarBeRJson(datasetPath='Dataset/BarBeR'):
    datasetPath = Path(datasetPath)
    imageDir = datasetPath / 'Images'
    annotDir = datasetPath / 'Annotations'

    samples = []

    for jsonPath in sorted(annotDir.glob('*.json')):
        with open(jsonPath) as f:
            data = json.load(f)['_via_img_metadata']

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

                shapeAttr = region['shape_attributes']
                regionAttr = region['region_attributes']

                if shapeAttr.get('name') != 'polygon':
                    continue

                xs = shapeAttr['all_points_x']
                ys = shapeAttr['all_points_y']

                objects.append({
                    'class': classNum,
                    'xmin': min(xs),
                    'ymin': min(ys),
                    'xmax': max(xs),
                    'ymax': max(ys),
                    'points': list(zip(xs, ys))
                })

            if objects:
                samples.append({'imagePath': imagePath, 'objects': objects})

    return samples

def encodeLabelGrid(objects, imgW, imgH, S=8):
    target = torch.zeros((S, S, 7))

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
            pts = np.array(obj['points'], dtype=np.float32)
            rect = cv2.minAreaRect(pts)
            angleNorm = abs(rect[2]) / 90.0

            target[cellY, cellX] = torch.tensor([
                1.0, offsetX, offsetY, w, h, float(obj['class']), angleNorm
            ])

    return target

class barcodeDataset(Dataset):
    def __init__(self, samples, S=8, augment=False):
        self.samples = samples
        self.S = S
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(str(sample['imagePath']))
        imgH, imgW = img.shape[:2]
        objects = sample['objects']

        if self.augment:
            img, objects = augmentSample(img, objects, imgW, imgH)

        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = encodeLabelGrid(objects, imgW, imgH, self.S)
        return img, target

    
class cachedBarcodeDataset(Dataset):
    def __init__(self, samples, S=8, augment=False):
        self.S = S
        self.augment = augment
        self.cache = []
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                print(f"  Caching {i}/{len(samples)}")
            img = cv2.imread(str(sample['imagePath']))
            if img is None:
                continue
            imgH, imgW = img.shape[:2]
            self.cache.append((img, sample['objects'], imgW, imgH))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        img, objects, imgW, imgH = self.cache[idx]
        img = img.copy()   # don't mutate the cache

        if self.augment:
            img, objects = augmentSample(img, objects, imgW, imgH)

        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = encodeLabelGrid(objects, imgW, imgH, self.S)
        return img, target


def buildDataloaders(batchSize=32, testSize=0.2, randomState=43, S=8, cache=False, datasetPath='Dataset/BarBeR'):
    samples = parseBarBeRJson(datasetPath=datasetPath)

    random.seed(randomState)
    random.shuffle(samples)

    splitIdx = int(len(samples) * (1 - testSize))
    trainSamples = samples[:splitIdx]
    valSamples = samples[splitIdx:]

    print(f'Training samples: {len(trainSamples)}, Validation samples: {len(valSamples)}')

    datasetClass = cachedBarcodeDataset if cache else barcodeDataset
    trainDataset = datasetClass(trainSamples, S=S, augment=True)
    valDataset   = datasetClass(valSamples,   S=S, augment=False)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, pin_memory=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, pin_memory=True)

    return trainLoader, valLoader