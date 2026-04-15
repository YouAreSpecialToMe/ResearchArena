#!/usr/bin/env python3
"""Ablation: Symmetric vs Asymmetric attention."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')

import subprocess
import json
import os

# Train with symmetric attention (set use_asymmetric=False)
config = {
    'seed': 42,
    'epochs': 50,
    'batch_size': 64,
    'lr': 1e-4,
    'hidden_dim': 384,
    'num_layers': 4,
    'num_heads': 6,
    'use_cell_type_cond': True,
    'use_asymmetric': False,  # Ablation: symmetric attention
    'predict_sign': True,
    'grad_clip': 1.0,
    'save_model': False,
    'model_path': 'models/ablation_symmetric.pt'
}

# Import and run training
from exp.crossgrn_main.train import train_crossgrn

results = train_crossgrn(
    'data/pbmc_rna_preprocessed.h5ad',
    'data/pbmc_atac_preprocessed.h5ad',
    'exp/ablation_symmetric/results.json',
    config
)

print("Symmetric attention ablation complete!")
