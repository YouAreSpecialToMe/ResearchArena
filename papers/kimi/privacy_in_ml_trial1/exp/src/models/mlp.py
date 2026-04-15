"""
MLP for tabular data (Purchase-100 dataset).
Architecture: input=600, hidden=[1024,512,256], output=100.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=600, hidden_dims=[1024, 512, 256], num_classes=100):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
