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

def yoloLoss(pred, target, config):
    # pred, target: (batch, S, S, 6) — [conf, x, y, w, h, class]
    objMask  = target[..., 0] == 1   # (batch, S, S) bool
    nobjMask = ~objMask

    lambdaCoord = config.get('lambdaCoord', 5.0)
    lambdaNoobj = config.get('lambdaNoobj', 0.5)
    lambdaAngle = config.get('lambdaAngle', 1.0)

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

    angleLoss = torch.tensor(0.0, device=pred.device)
    if objMask.any():
        angleLoss = lambdaAngle * F.mse_loss(
            pred[objMask][..., 6], target[objMask][..., 6], reduction='sum'
        )


    batchSize = pred.shape[0]
    total = (confLoss + bboxLoss + classLoss + angleLoss) / batchSize
    return total, confLoss.item() / batchSize, bboxLoss.item() / batchSize, classLoss.item() / batchSize, angleLoss.item() / batchSize

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
    epochs = [r['epoch'] for r in history]
    trainLoss = [r['trainLoss'] for r in history]
    valLoss = [r['valLoss'] for r in history]
    valRecall = [r['objRecallPct'] for r in history]
    lr = [r['learningRate'] for r in history]
    angleLoss = [r['valAngleLoss'] for r in history]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
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

    axes[3].plot(epochs, angleLoss, color='red')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Angle Loss')
    axes[3].set_title('Val Angle Loss')
    axes[3].grid(True)

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
        lambdaAngle   float   weight on angle loss (default 1.0);

    Metadata config:
        modelVariant    str
        datasetVersion  str
        batchSize       int
        notes           str   text description of this run
    """
    runId = generateRunId()
    modelsDir = Path('models')
    runDir = modelsDir / 'runs' / runId
    runDir.mkdir(parents=True, exist_ok=True)

    saveRunMetadata(modelsDir / 'runsMetadata.csv', runId, {
        'modelVariant': config.get('modelVariant',   ''),
        'datasetVersion': config.get('datasetVersion', ''),
        'numEpochs': config['numEpochs'],
        'batchSize': config.get('batchSize',      ''),
        'learningRate': config['lr'],
        'coordLossWeight': config.get('lambdaCoord', 5.0),
        'noobjLossWeight': config.get('lambdaNoobj', 0.5),
        'notes': config.get('notes',          ''),
    })

    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('tMax', 'numEpochs'), eta_min=1e-6
    )


    #loading universal best run loss value
    bestLossPath = modelsDir / 'bestValLoss.json'
    if bestLossPath.exists():
        with open(bestLossPath) as f:
            bestValLoss = json.load(f)['valLoss']
        print(f"Loaded cross-run best val loss: {bestValLoss:.4f}")
    else:
        bestValLoss = float('inf')
    #individual best run
    runBestValLoss = float('inf')
    history = []
    patience = config.get('earlyStoppingPatience', 15)
    epochsWithoutImprovement = 0

    for epoch in range(config['numEpochs']):
        epochStart = time.time()
        trainTotalLoss = 0.0
        valTotalLoss = 0.0
        valConfLoss = 0.0
        valBboxLoss = 0.0
        valClassLoss = 0.0
        totalObjCells = 0
        correctObjCells = 0
        valAngleLoss = 0.0

        # Training phase
        model.train()
        for imgs, targets in trainLoader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(imgs)
            loss, _, _, _, _ = yoloLoss(pred.float(), targets, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            trainTotalLoss += loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for imgs, targets in valLoader:
                imgs, targets = imgs.to(device), targets.to(device)
                pred = model(imgs)
                loss, cLoss, bLoss, klLoss, aLoss = yoloLoss(pred.float(), targets, config)

                valTotalLoss += loss.item()
                valConfLoss += cLoss
                valBboxLoss += bLoss
                valClassLoss += klLoss
                valAngleLoss += aLoss

                objMask = targets[..., 0] == 1
                totalObjCells   += objMask.sum().item()
                correctObjCells += (pred[..., 0][objMask] > 0.5).sum().item()

        currentLr = optimizer.param_groups[0]['lr']
        scheduler.step()

        objRecall = (correctObjCells / totalObjCells * 100) if totalObjCells > 0 else 0.0
        epochDuration = time.time() - epochStart
        numValBatches = len(valLoader)
        avgValLoss = valTotalLoss / numValBatches
        isBest = avgValLoss < bestValLoss
        isRunBest = avgValLoss < runBestValLoss

        if isRunBest:
            runBestValLoss = avgValLoss
            torch.save(model.state_dict(), runDir / 'bestModel.pth')

        if isBest:
            bestValLoss = avgValLoss
            torch.save(model.state_dict(), modelsDir / 'bestModel.pth')
            with open(modelsDir / 'bestValLoss.json', 'w') as f:
                json.dump({'valLoss': avgValLoss, 'runId': runId}, f)
            epochsWithoutImprovement = 0
        else:
            epochsWithoutImprovement += 1
            if epochsWithoutImprovement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        epochRow = {
            'runId': runId,
            'epoch': epoch + 1,
            'trainLoss': round(trainTotalLoss, 4),
            'valLoss': round(valTotalLoss, 4),
            'objRecallPct': round(objRecall, 2),
            'valConfLoss': round(valConfLoss, 4),
            'valBboxLoss': round(valBboxLoss, 4),
            'valClassLoss': round(valClassLoss, 4),
            'learningRate': currentLr,
            'epochDurationS': round(epochDuration, 1),
            'isBestEpoch': int(isRunBest),
            'isCrossRunBest': int(isBest),
            'valAngleLoss': round(valAngleLoss, 4),
        }
        history.append(epochRow)
        appendEpochRow(modelsDir / 'trainingHistory.csv', epochRow)

        print(f"Epoch {epoch + 1:>3}/{config['numEpochs']} | "
              f"Train: {trainTotalLoss:.4f} | "
              f"Val: {valTotalLoss:.4f} | "
              f"Recall: {objRecall:.1f}% | "
              f"LR: {currentLr:.2e} | "
              f"{'*' if isBest else ' '} "
              f"{'*' if isRunBest else ' '} "
              f"{epochDuration:.0f}s")

    with open(runDir / 'training_history.json', 'w') as f:
        json.dump({'runId': runId, 'config': config, 'history': history}, f, indent=2)

    plotTrainingCurves(history, runDir)
    print(f"\nRun {runId} complete. Artefacts saved to {runDir}")
    return history

