"""
Cross-Entropy baseline training script.
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
from shared.models import get_model
from shared.metrics import evaluate_model
from shared.utils import set_seed, save_results, AverageMeter


def train_ce(args):
    """Train with standard cross-entropy."""
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
    model = get_model(arch=args['arch'], num_classes=100).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], 
                         momentum=args['momentum'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(args['epochs']):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args["epochs"]}')
        for batch in pbar:
            images, labels, _, _, _ = batch
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            _, features = model(images, return_features=True)
            logits = model.get_classifier_logits(features)
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            _, predicted = torch.max(logits, 1)
            acc = (predicted == labels).float().mean().item()
            
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc, labels.size(0))
            
            pbar.set_postfix({'loss': loss_meter.avg, 'acc': acc_meter.avg * 100})
        
        scheduler.step()
        
        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == args['epochs'] - 1:
            results = evaluate_model(model, test_loader, device)
            print(f"Epoch {epoch+1}: Test Acc={results['accuracy']:.2f}%, ECE={results['ece']:.2f}%")
            
            if results['accuracy'] > best_acc:
                best_acc = results['accuracy']
    
    elapsed_time = (time.time() - start_time) / 60  # minutes
    
    # Final evaluation
    final_results = evaluate_model(model, test_loader, device)
    final_results['best_accuracy'] = best_acc
    final_results['runtime_minutes'] = elapsed_time
    
    return final_results


def main():
    args = {
        'data_root': './data',
        'noise_rate': 0.4,
        'noise_type': 'symmetric',
        'arch': 'resnet18',
        'batch_size': 128,
        'num_workers': 4,
        'epochs': 80,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'seed': 42,
    }
    
    print("="*60)
    print("Cross-Entropy Baseline on CIFAR-100 with 40% Noise")
    print("="*60)
    
    results = train_ce(args)
    
    # Save results
    output = {
        'experiment': 'ce_baseline',
        'config': args,
        'metrics': {
            'accuracy': results['accuracy'],
            'ece': results['ece'],
            'best_accuracy': results['best_accuracy']
        },
        'runtime_minutes': results['runtime_minutes']
    }
    
    os.makedirs('results', exist_ok=True)
    save_results(output, 'results/results.json')
    
    print("\n" + "="*60)
    print("Final Results:")
    print(f"  Test Accuracy: {results['accuracy']:.2f}%")
    print(f"  Best Accuracy: {results['best_accuracy']:.2f}%")
    print(f"  ECE: {results['ece']:.2f}%")
    print(f"  Runtime: {results['runtime_minutes']:.1f} minutes")
    print("="*60)


if __name__ == '__main__':
    main()
