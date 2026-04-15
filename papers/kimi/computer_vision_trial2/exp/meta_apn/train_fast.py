"""
Fast training for Meta-APN with reduced epochs for time-constrained experiments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from shared.models import wideresnet28_10, MetaAPN, get_prototype_distances
from shared.data_loader import load_cifar10, load_cifar100
from shared.augmentations import AUGMENTATION_OPS, apply_augmentation
from shared.utils import set_seed, save_meta_apn, compute_and_save_prototypes
from torchvision import transforms


def train_meta_apn_fast(dataset='cifar10', epochs=5, batch_size=128, lr=1e-3,
                        data_dir='./data', device='cuda', save_dir='models'):
    """Train Meta-APN with reduced epochs for faster experimentation."""
    set_seed(42)
    
    num_classes = 10 if dataset == 'cifar10' else 100
    model = wideresnet28_10(num_classes=num_classes).to(device)
    model.eval()
    
    # Compute or load prototypes
    prototypes_path = f'{save_dir}/prototypes_{dataset}.pt'
    if os.path.exists(prototypes_path):
        prototypes = torch.load(prototypes_path, map_location=device)
        print(f"Loaded prototypes from {prototypes_path}")
    else:
        prototypes = compute_and_save_prototypes(model, dataset, data_dir, device)
    
    # Initialize Meta-APN
    feature_dim = model.feature_dim
    meta_apn = MetaAPN(feature_dim, num_classes, num_operations=len(AUGMENTATION_OPS)).to(device)
    print(f"Meta-APN parameters: {meta_apn.count_parameters()}")
    
    # Load training data
    if dataset == 'cifar10':
        train_loader, _ = load_cifar10(data_dir, train=True, batch_size=batch_size)
    else:
        train_loader, _ = load_cifar100(data_dir, train=True, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(meta_apn.parameters(), lr=lr)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        meta_apn.train()
        total_loss = 0
        count = 0
        
        temperature = max(0.5, 1.0 - epoch * 0.1)
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            # Get features
            with torch.no_grad():
                features = model.get_features(images)
                distances = get_prototype_distances(features, prototypes, metric='cosine')
            
            # Get augmentation policy from Meta-APN
            policy_logits, severity_scale, num_aug_logits = meta_apn(features, distances)
            
            # Sample operations (use softmax instead of gumbel for speed)
            op_probs = F.softmax(policy_logits / temperature, dim=-1)
            
            # Apply top operation for each sample
            batch_loss = 0
            for i in range(min(4, images.size(0))):
                top_op = torch.argmax(op_probs[i])
                op_name = AUGMENTATION_OPS[top_op.item()]
                sev = max(1, min(5, int(3 * severity_scale[i].item())))
                
                aug_img = apply_augmentation(images[i], op_name, sev, dataset)
                if not isinstance(aug_img, torch.Tensor):
                    aug_img = transforms.ToTensor()(aug_img)
                aug_img = aug_img.unsqueeze(0).to(device)
                
                # Get prediction on augmented image
                output = model(aug_img)
                loss = F.cross_entropy(output, labels[i].unsqueeze(0))
                batch_loss += loss
            
            batch_loss = batch_loss / min(4, images.size(0))
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            count += 1
            
            # Limit iterations per epoch for speed
            if count >= 100:
                break
        
        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # Save final model
    save_meta_apn(meta_apn, dataset)
    print(f"Training complete! Final loss: {best_loss:.4f}")
    return meta_apn


if __name__ == '__main__':
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'cifar10'
    train_meta_apn_fast(dataset, epochs=5, data_dir='./data', device=device)
