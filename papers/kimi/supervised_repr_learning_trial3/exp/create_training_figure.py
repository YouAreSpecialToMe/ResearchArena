import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Load ETF-SCL results
with open('results/cifar100_etfscl_if100_seed42/results.json', 'r') as f:
    data = json.load(f)

history = data['history']
epochs = list(range(1, len(history['total_loss']) + 1))

# Subsample for cleaner visualization
subsampling = 5
epochs_sub = epochs[::subsampling]
total_loss_sub = history['total_loss'][::subsampling]
scl_loss_sub = history['scl_loss'][::subsampling]
etf_loss_sub = history['etf_loss'][::subsampling]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left plot: SCL Loss
ax1.plot(epochs_sub, scl_loss_sub, 'b-', linewidth=2, label='SCL Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Supervised Contrastive Loss', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_xlim([0, 200])

# Right plot: ETF Loss
ax2.plot(epochs_sub, etf_loss_sub, 'r-', linewidth=2, label='ETF Regularization Loss')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('ETF Geometry Regularization', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)
ax2.set_xlim([0, 200])

plt.tight_layout()
plt.savefig('figures/training_curves.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/training_curves.png', bbox_inches='tight', dpi=300)
print("Training curves saved to figures/training_curves.pdf and .png")
