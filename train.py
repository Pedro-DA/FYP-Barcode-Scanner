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
    valAcc     = [r['valAccuracyPct'] for r in history]
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

    axes[1].plot(epochs, valAcc, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].grid(True)

    axes[2].plot(epochs, lr, color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(runDir / 'trainingCurves.png', dpi=150, bbox_inches='tight')
    plt.close()


def train(model, trainLoader, valLoader, valDataSize, config):
    """
    Required config:
        numEpochs     int
        lr            float
        lambdaClass   float   weight on classification loss
        lambdaBox     float   weight on bounding box loss

    Metadata config:
        modelVariant    str   
        datasetVersion  str   
        batchSize       int
        notes           str   text description of this run
    """
    runId   = generateRunId()
    modelsDir = Path('models')
    runDir    = modelsDir / 'runs' / runId
    runDir.mkdir(parents=True, exist_ok=True)

    saveRunMetadata(modelsDir / 'runsMetadata.csv', runId, {
        'modelVariant':    config.get('modelVariant',   ''),
        'datasetVersion':  config.get('datasetVersion', ''),
        'numEpochs':       config['numEpochs'],
        'batchSize':       config.get('batchSize',      ''),
        'learningRate':    config['lr'],
        'classLossWeight': config['lambdaClass'],
        'boxLossWeight':   config['lambdaBox'],
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
        valClassLoss   = 0.0
        valBoxLoss     = 0.0
        totalCorrect   = 0

        # Training phase
        model.train()
        for x, y, z in trainLoader:
            x, y, z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()

            classOut, boxOut = model(x)
            classLoss = F.cross_entropy(classOut, y)
            boxLoss   = F.mse_loss(boxOut, z)
            loss      = (config['lambdaClass'] * classLoss) + (config['lambdaBox'] * boxLoss)
            loss.backward()
            optimizer.step()
            trainTotalLoss += loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for x, y, z in valLoader:
                x, y, z = x.to(device), y.to(device), z.to(device)
                classOut, boxOut = model(x)
                classLoss = F.cross_entropy(classOut, y)
                boxLoss   = F.mse_loss(boxOut, z)

                valClassLoss   += classLoss.item()
                valBoxLoss     += boxLoss.item()
                valTotalLoss   += (config['lambdaClass'] * classLoss.item()) + (config['lambdaBox'] * boxLoss.item())
                totalCorrect   += classOut.argmax(dim=1).eq(y).sum().item()

        currentLr    = optimizer.param_groups[0]['lr']
        scheduler.step(valTotalLoss)

        accuracy     = (totalCorrect / valDataSize) * 100
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
            'valAccuracyPct': round(accuracy,        2),
            'classLoss':      round(valClassLoss,    4),
            'boxLoss':        round(valBoxLoss,      4),
            'learningRate':   currentLr,
            'epochDurationS': round(epochDuration,   1),
            'isBestEpoch':    int(isBest),
        }
        history.append(epochRow)
        appendEpochRow(modelsDir / 'trainingHistory.csv', epochRow)

        print(f"Epoch {epoch + 1:>3}/{config['numEpochs']} | "
              f"Train: {trainTotalLoss:.4f} | "
              f"Val: {valTotalLoss:.4f} | "
              f"Acc: {accuracy:.1f}% | "
              f"LR: {currentLr:.2e} | "
              f"{'*' if isBest else ' '} "
              f"{epochDuration:.0f}s")

    with open(runDir / 'training_history.json', 'w') as f:
        json.dump({'runId': runId, 'config': config, 'history': history}, f, indent=2)

    plotTrainingCurves(history, runDir)
    print(f"\nRun {runId} complete. Artefacts saved to {runDir}")
    return history
