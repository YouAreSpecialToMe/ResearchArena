"""
CCL-SC (Confidence-aware Contrastive Learning for Selective Classification) baseline.
Uses binary confidence thresholding (0.5) for sample selection.
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from shared.data_loader import get_cifar100_loaders
from shared.models import get_model, LinearClassifier
from shared.losses import CCL_SC_Loss
from shared.metrics import expected_calibration_error
from shared.utils import set_seed, save_results, AverageMeter


def train_ccl_sc_encoder(args):
    """Stage 1: Train encoder with CCL-SC loss."""
    set_seed(args['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loaders
    train_loader, test_loader, train_dataset = get_cifar100_loaders(
        root=args['data_root'],
        noise_rate=args['noise_rate'],
        noise_type=args['noise_type'],
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        random_state=args['seed']
    )
    
    # Model
    model = get_model(arch=args['arch'], num_classes=100, feature_dim=args['feature_dim']).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], 
                         momentum=args['momentum'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs_stage1'])
    
    # CCL-SC loss with binary threshold
    criterion = CCL_SC_Loss(temperature=args['temperature'], threshold=0.5)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(args['epochs_stage1']):
        model.train()
        loss_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'CCL-SC Epoch {epoch+1}/{args["epochs_stage1"]}')
        for batch in pbar:
            images, labels, _, _, _ = batch
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward to get features and confidences
            z, h = model(images, return_features=True)
            logits = model.get_classifier_logits(h)
            probs = torch.softmax(logits, dim=1)
            confidences, _ = torch.max(probs, dim=1)
            
            # CCL-SC loss with binary weighting
            loss = criterion(z, labels, confidences)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), labels.size(0))
            pbar.set_postfix({'loss': loss_meter.avg})
        
        scheduler.step()
    
    encoder_time = (time.time() - start_time) / 60
    
    return model, encoder_time


def train_linear_classifier(encoder, args):
    """Stage 2: Train linear classifier on frozen features."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loaders
    train_loader, test_loader, _ = get_cifar100_loaders(
        root=args['data_root'],
        noise_rate=args['noise_rate'],
        noise_type=args['noise_type'],
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        random_state=args['seed']
    )
    
    # Freeze encoder
    encoder.eval()
    
    # Get feature dimension
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        _, features = encoder(dummy_input, return_features=True)
        feature_dim = features.shape[1]
    
    # Linear classifier
    classifier = LinearClassifier(feature_dim, num_classes=100).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=args['lr_stage2'],
                         momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs_stage2'])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(args['epochs_stage2']):
        classifier.train()
        
        for batch in train_loader:
            images, labels, _, _, _ = batch
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                _, features = encoder(images, return_features=True)
            
            logits = classifier(features)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    # Evaluate
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            _, features = encoder(images, return_features=True)
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)
            confidences, predicted = torch.max(probs, dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    classifier_time = (time.time() - start_time) / 60
    
    import numpy as np
    
    accuracy = 100.0 * correct / total
    ece = expected_calibration_error(
        np.array(all_confidences),
        np.array(all_preds),
        np.array(all_labels)
    )
    
    return accuracy, ece, encoder_time + classifier_time


def main():
    args = {
        'data_root': './data',
        'noise_rate': 0.4,
        'noise_type': 'symmetric',
        'arch': 'resnet18',
        'feature_dim': 128,
        'temperature': 0.5,
        'batch_size': 256,
        'num_workers': 4,
        'epochs_stage1': 100,
        'epochs_stage2': 30,
        'lr': 0.5,
        'lr_stage2': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'seed': 42,
    }
    
    print("="*60)
    print("CCL-SC Baseline on CIFAR-100 with 40% Noise")
    print("="*60)
    
    # Stage 1: Train encoder
    print("\nStage 1: Training encoder with CCL-SC...")
    model, encoder_time = train_ccl_sc_encoder(args)
    
    # Stage 2: Train linear classifier
    print("\nStage 2: Training linear classifier...")
    accuracy, ece, total_time = train_linear_classifier(model, args)
    
    # Save results
    output = {
        'experiment': 'ccl_sc_baseline',
        'config': args,
        'metrics': {
            'accuracy': accuracy,
            'ece': ece
        },
        'runtime_minutes': total_time
    }
    
    os.makedirs('results', exist_ok=True)
    save_results(output, 'results/results.json')
    
    print("\n" + "="*60)
    print("Final Results:")
    print(f"  Test Accuracy: {accuracy:.2f}%")
    print(f"  ECE: {ece:.2f}%")
    print(f"  Runtime: {total_time:.1f} minutes")
    print("="*60)


if __name__ == '__main__':
    main()
