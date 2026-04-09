import torch
from torch import nn

class FlexibleDetectionNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_units: int):
        super().__init__()

        # Shared CNN Backbone
        self.backbone = nn.Sequential(
            self._conv_block(in_channels,  hidden_units),        # block 1
            self._conv_block(hidden_units, hidden_units * 2),    # block 2
            self._conv_block(hidden_units * 2, hidden_units * 4),# block 3
            self._conv_block(hidden_units * 4, hidden_units * 8),# block 4
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))  # forces a fixed output size regardless of input

        flat_features = hidden_units * 8 * 4 * 4

        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(flat_features, 240),
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Linear(120, num_classes),
            nn.Softmax(dim=1)
        )

        # Bounding box head
        self.box_head = nn.Sequential(
            nn.Linear(flat_features, 240),
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Linear(120, 4),
            nn.Sigmoid()
        )
    
    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)

        return [self.class_head(x), self.box_head(x)]