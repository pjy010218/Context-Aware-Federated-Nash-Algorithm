import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBackbone(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=[256,128], embed_dim=128):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, embed_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)  # returns embedding

class Head(nn.Module):
    def __init__(self, embed_dim=128, out_dim=2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, out_dim)
    def forward(self, x):
        return self.fc(x)
