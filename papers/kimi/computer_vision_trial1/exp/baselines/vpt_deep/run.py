"""
VPT-Deep Baseline: Uniform prompts at all layers
"""

import os
import sys
import json
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from shared.data_loader import load_imagenet_c, load_imagenet_r, load_imagenet_sketch, get_transform
from shared.models import create_vit_model
from shared.metrics import accuracy, expected_calibration_error
from shared.utils import set_seed, save_results, AverageMeter, get_device, entropy_loss


class VPTDeepModel(nn.Module):
    """VPT-Deep: Prompts at every layer."""
    
    def __init__(self, vit_model, n_prompts=10, n_layers=12, embed_dim=768):
        super().__init__()
        self.vit = vit_model
        self.n_prompts = n_prompts
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        
        # Prompts for every layer
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(n_prompts, embed_dim) * 0.02)
            for _ in range(n_layers)
        ])
        
        # Freeze backbone
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        
        # Process through transformer blocks with prompts
        for i, block in enumerate(self.vit.blocks):
            if i < len(self.prompts):
                prompts = self.prompts[i].unsqueeze(0).expand(B, -1, -1)
                x = torch.cat([prompts, x], dim=1)
            
            x = block(x)
            
            # Remove prompts for next layer
            if i < len(self.prompts):
                x = x[:, self.n_prompts:, :]
        
        x = self.vit.norm(x)
        x = x[:, 0]  # CLS token
        x = self.vit.head(x)
        
        return x
    
    def get_prompt_params(self):
        return list(self.prompts.parameters())


def adapt_batch(model, inputs, optimizer, n_iter=1):
    """Adapt prompts using entropy minimization."""
    model.train()
    
    for _ in range(n_iter):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = entropy_loss(outputs)
        loss.backward()
        optimizer.step()
    
    model.eval()


def evaluate_vpt_deep(model, dataloader, device, lr=5e-3, adapt=True):
    """Evaluate VPT-Deep with optional adaptation."""
    model.eval()
    
    if adapt:
        optimizer = torch.optim.Adam(model.get_prompt_params(), lr=lr)
    
    all_outputs = []
    all_targets = []
    
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    for inputs, targets in tqdm(dataloader, desc="Evaluating VPT-Deep"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if adapt:
            with torch.enable_grad():
                adapt_batch(model, inputs, optimizer, n_iter=1)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1_meter.update(acc1, inputs.size(0))
        top5_meter.update(acc5, inputs.size(0))
        
        all_outputs.append(outputs.cpu())
        all_targets.append(targets.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    ece = expected_calibration_error(all_outputs, all_targets)
    
    return {
        'top1_acc': top1_meter.avg,
        'top5_acc': top5_meter.avg,
        'ece': ece
    }


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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results/vpt_deep')
    parser.add_argument('--adapt', action='store_true')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Evaluating VPT-Deep on {args.dataset}")
    
    vit = create_vit_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    model = VPTDeepModel(vit, n_prompts=args.n_prompts, n_layers=12, embed_dim=768)
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
    
    start_time = time.time()
    results = evaluate_vpt_deep(model, dataloader, device, lr=args.lr, adapt=args.adapt)
    runtime = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Top-1 Accuracy: {results['top1_acc']:.2f}%")
    print(f"  Top-5 Accuracy: {results['top5_acc']:.2f}%")
    print(f"  ECE: {results['ece']:.4f}")
    print(f"  Runtime: {runtime:.2f}s")
    
    output = {
        'experiment': 'vpt_deep',
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
