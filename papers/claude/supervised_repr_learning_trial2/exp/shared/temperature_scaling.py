"""Temperature scaling for post-hoc calibration."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def learn_temperature(model, val_loader, device='cuda', max_iter=100, lr=0.01):
    """Learn optimal temperature on validation set.

    Returns:
        optimal_temperature: float
    """
    model.eval()

    # Collect all logits and labels
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            all_logits.append(logits)
            all_labels.append(targets)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Optimize temperature
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(logits)
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    optimal_temp = scaler.temperature.item()
    return optimal_temp
