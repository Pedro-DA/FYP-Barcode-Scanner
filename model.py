import torch
from torch import nn
    
class GridDetectionNet(nn.Module):
    def __init__(self, in_channels: int = 3, S: int = 8, hidden_units: int = 64):
        super().__init__()
        self.S = S

        # Shared CNN Backbone
        self.backbone = nn.Sequential(
            self.convBlock(in_channels,  hidden_units), # block 1
            self.convBlock(hidden_units, hidden_units * 2), # block 2
            self.convBlock(hidden_units * 2, hidden_units * 4), # block 3
            self.convBlock(hidden_units * 4, hidden_units * 8), # block 4
        )

        self.spatialPool = nn.AdaptiveAvgPool2d((S, S))

        self.detectionHead = nn.Sequential(
            nn.Conv2d(hidden_units * 8, hidden_units * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(hidden_units * 4, 7, kernel_size=1),
        )

    @staticmethod
    def convBlock(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.spatialPool(x) # (batch, C, S, S)
        x = self.detectionHead(x) # (batch, 6, S, S)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1) # (batch, S, S, 6)
        return x