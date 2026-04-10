from dataset import buildDataloaders
from model import FlexibleDetectionNet
from train import train
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainLoader, valLoader, valDataSize = buildDataloaders()

net = FlexibleDetectionNet(in_channels=3, num_classes=2, hidden_units=12).to(device)

config = {
    'numEpochs':   100,
    'lr':          0.01,
    'lambdaClass': 1.0,
    'lambdaBox':   0.5,
    'batchSize':   32,
}
train(net, trainLoader, valLoader, valDataSize, config)