# Barcode Scanner

A real-time barcode and QR code detection system built as a Final Year Project. A lightweight custom CNN detects barcode regions in a frame, then passes crops to ZXing for decoding.

Supports two barcode classes: **1D** (Code 128, EAN, UPC, etc.) and **2D** (QR, Data Matrix, PDF417, etc.).

## Requirements

```
pip install -r requirements.txt
```

Python 3.10+ recommended. A CUDA-capable GPU is optional but speeds up training significantly.

## Dataset

Place your dataset in the following structure:

```
Dataset/
  Images/        # image files (.jpg / .png)
  Annotations/   # VGG Image Annotator (VIA) JSON annotation files
```

## Usage

### Live camera

```bash
python main.py camera
python main.py camera --modelPath models/my_model.pth
```

### Single image

```bash
python main.py image --imagePath path/to/image.jpg
python main.py image --imagePath path/to/image.jpg --modelPath models/my_model.pth
```

### Train

```bash
python main.py train
python main.py train --cache      # pre-loads dataset into RAM for faster training
```

Trained model checkpoints are saved to the `models/` directory. If `--modelPath` is not specified, the most recently saved checkpoint is loaded automatically.

## Project Structure

| File | Description |
|------|-------------|
| `model.py` | CNN architecture (`GridDetectionNet`) |
| `dataset.py` | Dataset loading, augmentation, and label encoding |
| `train.py` | Training loop with early stopping and cosine LR schedule |
| `inference.py` | Detection pipeline and ZXing crop decoding |
| `telemetry.py` | Frame-level performance metrics and session report |
| `main.py` | Entry point — camera, image, and train modes |
