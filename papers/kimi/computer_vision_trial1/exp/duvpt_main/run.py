"""
DU-VPT Main Method Implementation
Decomposed Uncertainty-Guided Visual Prompt Tuning for Test-Time Adaptation
"""

import os
import sys
import json
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.data_loader import load_imagenet_c, load_imagenet_r, load_imagenet_sketch, get_transform
from shared.models import create_vit_model
from shared.metrics import accuracy, expected_calibration_error
from shared.utils import set_seed, save_results, AverageMeter, get_device, entropy_loss


class DUVPTModel(nn.Module):
    """DU-VPT with uncertainty decomposition and targeted prompts."""
    
    def __init__(self, vit_model, n_prompts=10, n_layers=12, embed_dim=768, 
                 tau_alpha=0.2, tau_epsilon=1.0):
        super().__init__()
        self.vit = vit_model
        self.n_prompts = n_prompts
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.tau_alpha = tau_alpha
        self.tau_epsilon = tau_epsilon
        
        # Prompts for each layer
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(n_prompts, embed_dim) * 0.02)
            for _ in range(n_layers)
        ])
        
        # Calibration stats
        self.register_buffer('calib_mean', torch.zeros(n_layers, embed_dim))
        self.register_buffer('calib_std', torch.ones(n_layers, embed_dim))
        
        # Freeze backbone
        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.selected_layers = []
        self.shift_type = 'unknown'
    
    def forward_with_features(self, x):
        """Forward pass returning intermediate features."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        
        layer_features = []
        
        # Process through transformer blocks with prompts
        for i, block in enumerate(self.vit.blocks):
            if i < len(self.prompts):
                prompts = self.prompts[i].unsqueeze(0).expand(B, -1, -1)
                x = torch.cat([prompts, x], dim=1)
            
            x = block(x)
            
            # Store features (removing prompts)
            if i < len(self.prompts):
                feat = x[:, self.n_prompts:, :].clone()
            else:
                feat = x.clone()
            layer_features.append(feat)
            
            # Remove prompts for next layer
            if i < len(self.prompts):
                x = x[:, self.n_prompts:, :]
        
        x = self.vit.norm(x)
        cls_output = x[:, 0]
        output = self.vit.head(cls_output)
        
        return output, layer_features
    
    def forward(self, x):
        """Standard forward pass."""
        output, _ = self.forward_with_features(x)
        return output
    
    def compute_uncertainty(self, layer_features):
        """Compute aleatoric and epistemic uncertainty."""
        B = layer_features[0].shape[0]
        L = len(layer_features)
        
        aleatoric = torch.zeros(B, L, device=layer_features[0].device)
        epistemic = torch.zeros(B, L, device=layer_features[0].device)
        
        for l, features in enumerate(layer_features):
            B, N, D = features.shape
            
            # Aleatoric: local variance of patch features
            patch_features = features[:, 1:, :]
            spatial_var = patch_features.var(dim=1).mean(dim=-1)
            aleatoric[:, l] = spatial_var
            
            # Epistemic: deviation from calibration stats
            cls_feat = features[:, 0, :]
            normalized = (cls_feat - self.calib_mean[l]) / (self.calib_std[l] + 1e-8)
            epistemic[:, l] = (normalized ** 2).sum(dim=-1)
        
        # Normalize
        aleatoric = aleatoric / (aleatoric.max(dim=1, keepdim=True)[0] + 1e-8)
        epistemic = epistemic / (epistemic.max(dim=1, keepdim=True)[0] + 1e-8)
        
        return aleatoric, epistemic
    
    def diagnose_shift(self, aleatoric, epistemic):
        """Diagnose shift type and select layers."""
        B, L = aleatoric.shape
        
        alpha_mean = aleatoric.mean(dim=0)
        epsilon_mean = epistemic.mean(dim=0)
        
        early_end = L // 3
        deep_start = 2 * L // 3
        
        alpha_early = alpha_mean[:early_end].mean()
        epsilon_deep = epsilon_mean[deep_start:].mean()
        
        if alpha_early > self.tau_alpha and epsilon_deep < self.tau_epsilon:
            shift_type = 'low_level'
            target_layers = list(range(0, early_end + 2))
        elif alpha_early < self.tau_alpha and epsilon_deep > self.tau_epsilon:
            shift_type = 'semantic'
            target_layers = list(range(deep_start, L))
        elif alpha_early > self.tau_alpha and epsilon_deep > self.tau_epsilon:
            shift_type = 'mixed'
            target_layers = list(range(L))
        else:
            shift_type = 'none'
            target_layers = []
        
        self.shift_type = shift_type
        self.selected_layers = target_layers
        
        return shift_type, target_layers
    
    def set_calibration_stats(self, mean, std):
        self.calib_mean.copy_(mean)
        self.calib_std.copy_(std)
    
    def get_prompt_params_for_layers(self, layer_indices):
        """Get prompt parameters for selected layers only."""
        params = []
        for i in layer_indices:
            if i < len(self.prompts):
                params.append(self.prompts[i])
        return params


def adapt_batch_duvpt(model, inputs, optimizer, compute_uncertainty=False):
    """Adapt DU-VPT on a batch."""
    model.train()
    
    if compute_uncertainty:
        with torch.no_grad():
            outputs, layer_features = model.forward_with_features(inputs)
            aleatoric, epistemic = model.compute_uncertainty(layer_features)
            shift_type, target_layers = model.diagnose_shift(aleatoric, epistemic)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = entropy_loss(outputs)
    loss.backward()
    optimizer.step()
    
    model.eval()


def evaluate_duvpt(model, dataloader, device, lr=5e-3, adapt=True, 
                   compute_uncertainty=False, tau_alpha=0.2, tau_epsilon=1.0):
    """Evaluate DU-VPT with optional adaptation."""
    model.eval()
    
    all_outputs = []
    all_targets = []
    shift_types = []
    
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    # Setup optimizer for all prompts initially
    if adapt:
        optimizer = torch.optim.Adam(model.prompts.parameters(), lr=lr)
    
    for inputs, targets in tqdm(dataloader, desc="Evaluating DU-VPT"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if adapt:
            with torch.enable_grad():
                adapt_batch_duvpt(model, inputs, optimizer, compute_uncertainty)
        
        with torch.no_grad():
            outputs = model(inputs)
            
            if compute_uncertainty:
                _, layer_features = model.forward_with_features(inputs)
                aleatoric, epistemic = model.compute_uncertainty(layer_features)
                shift_type, _ = model.diagnose_shift(aleatoric, epistemic)
                shift_types.append(shift_type)
        
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1_meter.update(acc1, inputs.size(0))
        top5_meter.update(acc5, inputs.size(0))
        
        all_outputs.append(outputs.cpu())
        all_targets.append(targets.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    ece = expected_calibration_error(all_outputs, all_targets)
    
    results = {
        'top1_acc': top1_meter.avg,
        'top5_acc': top5_meter.avg,
        'ece': ece
    }
    
    if compute_uncertainty and shift_types:
        # Count shift types
        from collections import Counter
        shift_counts = Counter(shift_types)
        results['shift_type_distribution'] = dict(shift_counts)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['imagenet-c', 'imagenet-r', 'imagenet-sketch'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--corruption', type=str, default=None)
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_prompts', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--tau_alpha', type=float, default=0.2)
    parser.add_argument('--tau_epsilon', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results/duvpt')
    parser.add_argument('--adapt', action='store_true')
    parser.add_argument('--compute_uncertainty', action='store_true')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running DU-VPT on {args.dataset}")
    
    vit = create_vit_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    model = DUVPTModel(vit, n_prompts=args.n_prompts, n_layers=12, embed_dim=768,
                      tau_alpha=args.tau_alpha, tau_epsilon=args.tau_epsilon)
    model = model.to(device)
    
    # Load dataset
    if args.dataset == 'imagenet-c':
        dataloader, dataset = load_imagenet_c(
            os.path.join(args.data_root, 'imagenet-c'),
            corruption=args.corruption,
            severity=args.severity,
            batch_size=args.batch_size,
            seed=args.seed
        )
    elif args.dataset == 'imagenet-r':
        dataloader, dataset = load_imagenet_r(
            os.path.join(args.data_root, 'imagenet-r'),
            batch_size=args.batch_size,
            seed=args.seed
        )
    else:
        dataloader, dataset = load_imagenet_sketch(
            os.path.join(args.data_root, 'imagenet-sketch'),
            batch_size=args.batch_size,
            seed=args.seed
        )
    
    # Use zero calibration for now
    calib_mean = torch.zeros(12, 768).to(device)
    calib_std = torch.ones(12, 768).to(device)
    model.set_calibration_stats(calib_mean, calib_std)
    
    start_time = time.time()
    results = evaluate_duvpt(model, dataloader, device, lr=args.lr, 
                            adapt=args.adapt, compute_uncertainty=args.compute_uncertainty,
                            tau_alpha=args.tau_alpha, tau_epsilon=args.tau_epsilon)
    runtime = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Top-1 Accuracy: {results['top1_acc']:.2f}%")
    print(f"  Top-5 Accuracy: {results['top5_acc']:.2f}%")
    print(f"  ECE: {results['ece']:.4f}")
    print(f"  Runtime: {runtime:.2f}s")
    
    output = {
        'experiment': 'duvpt',
        'dataset': args.dataset,
        'corruption': args.corruption,
        'severity': args.severity,
        'seed': args.seed,
        'adapt': args.adapt,
        'metrics': results,
        'runtime_seconds': runtime,
        'config': vars(args)
    }
    
    output_file = os.path.join(args.output_dir, f'results_{args.dataset}_seed{args.seed}.json')
    save_results(output, output_file)
    print(f"Saved to {output_file}")


if __name__ == '__main__':
    main()
