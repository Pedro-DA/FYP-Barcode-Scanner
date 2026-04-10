import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def yoloLoss(pred, target, lambdaCoord=5.0, lambdaNoobj=0.5):
    # pred, target: (batch, S, S, 6) — [conf, x, y, w, h, class]
    objMask  = target[..., 0] == 1   # (batch, S, S) bool
    nobjMask = ~objMask

    # Confidence — BCE, downweight no-obj cells
    rawConfLoss = F.binary_cross_entropy(pred[..., 0], target[..., 0], reduction='none')
    confLoss = (objMask.float() * rawConfLoss + lambdaNoobj * nobjMask.float() * rawConfLoss).sum()

    # Bbox — MSE with sqrt on w/h, only obj cells
    bboxLoss = torch.tensor(0.0, device=pred.device)
    if objMask.any():
        predBox = pred[objMask][..., 1:5]    # (N, 4) [x, y, w, h]
        targBox = target[objMask][..., 1:5]
        predSqrt = torch.cat([predBox[..., :2], predBox[..., 2:].clamp(min=1e-6).sqrt()], dim=-1)
        targSqrt = torch.cat([targBox[..., :2], targBox[..., 2:].clamp(min=1e-6).sqrt()], dim=-1)
        bboxLoss = lambdaCoord * F.mse_loss(predSqrt, targSqrt, reduction='sum')

    # Class — BCE, only obj cells
    classLoss = torch.tensor(0.0, device=pred.device)
    if objMask.any():
        classLoss = F.binary_cross_entropy(
            pred[objMask][..., 5], target[objMask][..., 5], reduction='sum'
        )

    batchSize = pred.shape[0]
    total = (confLoss + bboxLoss + classLoss) / batchSize
    return total, confLoss.item() / batchSize, bboxLoss.item() / batchSize, classLoss.item() / batchSize

def generateRunId():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def appendEpochRow(csvPath, row):
    writeHeader = not csvPath.exists()
    with open(csvPath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if writeHeader:
            writer.writeheader()
        writer.writerow(row)

def saveRunMetadata(metaCsvPath, runId, config):
    row = {'runId': runId, **config}
    writeHeader = not metaCsvPath.exists()
    with open(metaCsvPath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if writeHeader:
            writer.writeheader()
        writer.writerow(row)

def plotTrainingCurves(history, runDir):
    epochs     = [r['epoch']          for r in history]
    trainLoss  = [r['trainLoss']      for r in history]
    valLoss    = [r['valLoss']        for r in history]
    valRecall = [r['objRecallPct'] for r in history]
    lr         = [r['learningRate']   for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Training Curves — {runDir.name}", fontsize=13)

    axes[0].plot(epochs, trainLoss, label='Train Loss')
    axes[0].plot(epochs, valLoss,   label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train vs Val Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, valRecall, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Obj Recall (%)')
    axes[1].set_title('Object Cell Recall')
    axes[1].grid(True)

    axes[2].plot(epochs, lr, color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(runDir / 'trainingCurves.png', dpi=150, bbox_inches='tight')
    plt.close()


def train(model, trainLoader, valLoader, config):
    """
    Required config:
        numEpochs     int
        lr            float
        lambdaCoord   float   weight on bbox loss (default 5.0)
        lambdaNoobj   float   weight on no-object confidence loss (default 0.5)

    Metadata config:
        modelVariant    str
        datasetVersion  str
        batchSize       int
        notes           str   text description of this run
    """
    runId     = generateRunId()
    modelsDir = Path('models')
    runDir    = modelsDir / 'runs' / runId
    runDir.mkdir(parents=True, exist_ok=True)

    saveRunMetadata(modelsDir / 'runsMetadata.csv', runId, {
        'modelVariant':    config.get('modelVariant',   ''),
        'datasetVersion':  config.get('datasetVersion', ''),
        'numEpochs':       config['numEpochs'],
        'batchSize':       config.get('batchSize',      ''),
        'learningRate':    config['lr'],
        'coordLossWeight': config.get('lambdaCoord', 5.0),
        'noobjLossWeight': config.get('lambdaNoobj', 0.5),
        'notes':           config.get('notes',          ''),
    })

    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1
    )

    bestValLoss = float('inf')
    history     = []

    for epoch in range(config['numEpochs']):
        epochStart     = time.time()
        trainTotalLoss = 0.0
        valTotalLoss   = 0.0
        valConfLoss    = 0.0
        valBboxLoss    = 0.0
        valClassLoss   = 0.0
        totalObjCells  = 0
        correctObjCells = 0

        # Training phase
        model.train()
        for imgs, targets in trainLoader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss, _, _, _ = yoloLoss(pred, targets, config.get('lambdaCoord', 5.0), config.get('lambdaNoobj', 0.5))
            loss.backward()
            optimizer.step()
            trainTotalLoss += loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for imgs, targets in valLoader:
                imgs, targets = imgs.to(device), targets.to(device)
                pred = model(imgs)
                loss, cLoss, bLoss, klLoss = yoloLoss(pred, targets, config.get('lambdaCoord', 5.0), config.get('lambdaNoobj', 0.5))

                valTotalLoss  += loss.item()
                valConfLoss   += cLoss
                valBboxLoss   += bLoss
                valClassLoss  += klLoss

                objMask = targets[..., 0] == 1
                totalObjCells   += objMask.sum().item()
                correctObjCells += (pred[..., 0][objMask] > 0.5).sum().item()

        currentLr    = optimizer.param_groups[0]['lr']
        scheduler.step(valTotalLoss)

        objRecall    = (correctObjCells / totalObjCells * 100) if totalObjCells > 0 else 0.0
        epochDuration = time.time() - epochStart
        isBest       = valTotalLoss < bestValLoss

        if isBest:
            bestValLoss = valTotalLoss
            torch.save(model.state_dict(), runDir / 'bestModel.pth')

        epochRow = {
            'runId':          runId,
            'epoch':          epoch + 1,
            'trainLoss':      round(trainTotalLoss, 4),
            'valLoss':        round(valTotalLoss,   4),
            'objRecallPct':   round(objRecall,      2),
            'valConfLoss':    round(valConfLoss,    4),
            'valBboxLoss':    round(valBboxLoss,    4),
            'valClassLoss':   round(valClassLoss,   4),
            'learningRate':   currentLr,
            'epochDurationS': round(epochDuration,  1),
            'isBestEpoch':    int(isBest),
        }
        history.append(epochRow)
        appendEpochRow(modelsDir / 'trainingHistory.csv', epochRow)

        print(f"Epoch {epoch + 1:>3}/{config['numEpochs']} | "
              f"Train: {trainTotalLoss:.4f} | "
              f"Val: {valTotalLoss:.4f} | "
              f"Recall: {objRecall:.1f}% | "
              f"LR: {currentLr:.2e} | "
              f"{'*' if isBest else ' '} "
              f"{epochDuration:.0f}s")

    with open(runDir / 'training_history.json', 'w') as f:
        json.dump({'runId': runId, 'config': config, 'history': history}, f, indent=2)

    plotTrainingCurves(history, runDir)
    print(f"\nRun {runId} complete. Artefacts saved to {runDir}")
    return history

