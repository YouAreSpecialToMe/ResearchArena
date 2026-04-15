"""
Train Meta-Augmentation Policy Network (Meta-APN).

The Meta-APN is a lightweight network (~10K parameters) that predicts augmentation
parameters based on prototype similarity. It is trained offline on source data.
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
from shared.augmentations import AUGMENTATION_OPS, apply_augmentation, gumbel_softmax_sample
from shared.utils import set_seed, save_meta_apn, compute_and_save_prototypes
from torchvision import transforms


def train_meta_apn(dataset='cifar10', epochs=50, batch_size=128, lr=1e-3,
                   data_dir='./data', device='cuda', save_dir='models'):
    """
    Train Meta-APN on source data.
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        data_dir: Data directory
        device: Device
        save_dir: Directory to save trained model
    """
    set_seed(42)
    
    # Load source model
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
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        meta_apn.train()
        total_loss = 0
        correct = 0
        total = 0
        
        temperature = max(0.5, 1.0 - epoch * 0.01)  # Anneal from 1.0 to 0.5
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            # Get features
            with torch.no_grad():
                features = model.get_features(images)
                distances = get_prototype_distances(features, prototypes, metric='cosine')
            
            # Get augmentation policy from Meta-APN
            policy_logits, severity_scale, num_aug_logits = meta_apn(features, distances)
            
            # Sample operations using Gumbel-Softmax
            op_probs = gumbel_softmax_sample(policy_logits, temperature, hard=True)
            
            # For each image in batch, apply augmentations
            batch_loss = 0
            batch_correct = 0
            
            for i in range(min(4, images.size(0))):  # Subsample for efficiency
                # Select top operations
                topk_values, topk_indices = torch.topk(op_probs[i], k=4)
                
                # Apply augmentations
                aug_images = []
                for op_idx in topk_indices:
                    op_name = AUGMENTATION_OPS[op_idx.item()]
                    sev = max(1, min(5, int(3 * severity_scale[i].item())))
                    
                    aug_img = apply_augmentation(images[i], op_name, sev, dataset)
                    if not isinstance(aug_img, torch.Tensor):
                        aug_img = transforms.ToTensor()(aug_img)
                    aug_images.append(aug_img)
                
                aug_batch = torch.stack(aug_images).to(device)
                
                # Get predictions on augmented images
                outputs = model(aug_batch)
                
                # Proxy objective: minimize cross-entropy on correct class
                # This is a differentiable surrogate for accuracy
                target = labels[i].unsqueeze(0).expand(outputs.size(0))
                loss = F.cross_entropy(outputs, target)
                batch_loss += loss
                
                # Track accuracy
                _, predicted = outputs.max(1)
                batch_correct += predicted.eq(target).sum().item()
            
            batch_loss = batch_loss / min(4, images.size(0))
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            correct += batch_correct
            total += min(4, images.size(0)) * 4
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, Temp={temperature:.3f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_meta_apn(meta_apn, dataset)
    
    print(f"Training complete! Best loss: {best_loss:.4f}")
    return meta_apn


def train_both_datasets(data_dir='./data', device='cuda'):
    """Train Meta-APN for both CIFAR-10 and CIFAR-100"""
    print("=" * 60)
    print("Training Meta-APN for CIFAR-10")
    print("=" * 60)
    train_meta_apn('cifar10', epochs=30, data_dir=data_dir, device=device)
    
    print("\n" + "=" * 60)
    print("Training Meta-APN for CIFAR-100")
    print("=" * 60)
    train_meta_apn('cifar100', epochs=30, data_dir=data_dir, device=device)


if __name__ == '__main__':
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        train_meta_apn(dataset, epochs=30, data_dir='./data', device=device)
    else:
        train_both_datasets(data_dir='./data', device=device)
