"""Generate gating visualization figures showing which tokens get gated under corruption."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
os.environ.setdefault('TMPDIR', '/var/tmp')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from data_loader import (ImageNetValDataset, CorruptedImageNetDataset, get_transform,
                         get_dataloader, get_calibration_indices)
from models import load_model
from stg import SpectralTokenGating

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def get_single_image(dataset, idx):
    """Get a single image and label from dataset."""
    img, label = dataset[idx]
    return img.unsqueeze(0), label


def denormalize(tensor):
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    model_key = 'deit_small'
    model, config = load_model(model_key)

    # Load calibration
    stg = SpectralTokenGating(model, model_key, K=3, alpha=5.0, tau_percentile=95)
    cal_path = os.path.join(CHECKPOINT_DIR, f'calibration_{model_key}_seed42.pt')
    stg.load_calibration(cal_path)
    stg.enable()

    # Load datasets
    clean_dataset = ImageNetValDataset(DATA_DIR, transform=get_transform())

    corruptions = ['gaussian_noise', 'defocus_blur', 'frost', 'jpeg_compression']
    corruption_labels = ['Gaussian Noise', 'Defocus Blur', 'Frost', 'JPEG Compression']

    # Pick 4 diverse example indices
    example_indices = [0, 100, 500, 1000]

    n_rows = len(example_indices)
    n_cols = 2 + len(corruptions)  # clean + gating_clean + corruptions with gating

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, img_idx in enumerate(example_indices):
        # Clean image
        img, label = get_single_image(clean_dataset, img_idx)

        # Run through model to get clean gating scores
        with torch.no_grad():
            img_gpu = img.to('cuda')
            _ = model(img_gpu)

        clean_scores = stg.get_gating_scores()
        # Use layer 6 (middle layer) for visualization
        viz_layer = 6 if 6 in clean_scores else list(clean_scores.keys())[1]
        clean_gating = clean_scores[viz_layer][0].cpu().numpy()  # (196,) for 14x14
        grid_size = int(np.sqrt(len(clean_gating)))
        clean_gating_map = clean_gating.reshape(grid_size, grid_size)

        # Show clean image
        axes[row, 0].imshow(denormalize(img[0]))
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title('Clean Image', fontsize=9, fontweight='bold')

        # Show clean gating map
        im = axes[row, 1].imshow(clean_gating_map, cmap='RdYlBu', vmin=0.3, vmax=1.0,
                                  interpolation='nearest')
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        if row == 0:
            axes[row, 1].set_title('Clean Gating', fontsize=9, fontweight='bold')

        # For each corruption
        for c_idx, corruption in enumerate(corruptions):
            corr_dataset = CorruptedImageNetDataset(DATA_DIR, corruption, severity=5)
            corr_img, corr_label = get_single_image(corr_dataset, img_idx)

            with torch.no_grad():
                corr_gpu = corr_img.to('cuda')
                _ = model(corr_gpu)

            corr_scores = stg.get_gating_scores()
            corr_gating = corr_scores[viz_layer][0].cpu().numpy()
            corr_gating_map = corr_gating.reshape(grid_size, grid_size)

            # Show corrupted gating map
            axes[row, 2 + c_idx].imshow(corr_gating_map, cmap='RdYlBu', vmin=0.3, vmax=1.0,
                                         interpolation='nearest')
            axes[row, 2 + c_idx].set_xticks([])
            axes[row, 2 + c_idx].set_yticks([])
            if row == 0:
                axes[row, 2 + c_idx].set_title(corruption_labels[c_idx], fontsize=9, fontweight='bold')

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label('Gating Score (1=keep, 0=suppress)', fontsize=8)

    plt.suptitle('STG Gating Score Maps at Layer 6 (DeiT-S, Severity 5)\n'
                 'Blue=kept tokens, Red=suppressed tokens',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'gating_visualization.pdf'),
                bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(FIGURES_DIR, 'gating_visualization.png'),
                bbox_inches='tight', dpi=150)
    print(f"Saved gating visualization to {FIGURES_DIR}/gating_visualization.pdf")

    stg.disable()


if __name__ == '__main__':
    main()
