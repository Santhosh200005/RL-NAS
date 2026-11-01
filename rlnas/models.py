import torch.nn as nn
from .search_space import Architecture, NUM_CLASSES

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


class CandidateCNN(nn.Module):
    def __init__(self, arch: Architecture, in_ch: int = 3, num_classes: int = NUM_CLASSES):
        super().__init__()
        layers, c = [], in_ch
        for f, k, p in arch.layers:
            layers += [nn.Conv2d(c, f, k, padding=k // 2, bias=False), nn.BatchNorm2d(f), _ACT[arch.act]()]
            if p == "max":
                layers.append(nn.MaxPool2d(2))
            elif p == "avg":
                layers.append(nn.AvgPool2d(2))
            c = f
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(arch.dropout)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x):
        return self.fc(self.drop(self.gap(self.features(x)).flatten(1)))
