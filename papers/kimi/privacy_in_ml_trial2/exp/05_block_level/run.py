"""
Block-level intervention ablation.
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared.models import get_model
from shared.data_loader import get_data_loaders
from shared.utils import set_seed, evaluate_model


def get_block_level_layers(arch):
    """Define blocks for different architectures."""
    if arch == 'resnet18':
        # 5 blocks: [conv1+bn1], [layer1], [layer2], [layer3], [layer4+fc]
        return {
            'block0': ['conv1', 'bn1'],
            'block1': ['layer1'],
            'block2': ['layer2'],
            'block3': ['layer3'],
            'block4': ['layer4', 'fc']
        }
    elif arch == 'vgg16':
        # 4 blocks: [conv1-4], [conv5-8], [conv9-12], [conv13+fc]
        return {
            'block0': ['features.0', 'features.1', 'features.3', 'features.4', 
                      'features.7', 'features.8', 'features.9', 'features.10'],
            'block1': ['features.14', 'features.15', 'features.16', 'features.17',
                      'features.21', 'features.22', 'features.23', 'features.24'],
            'block2': ['features.28', 'features.29', 'features.30', 'features.31',
                      'features.34', 'features.35', 'features.36', 'features.37'],
            'block3': ['classifier']
        }
    else:
        return {}


def compute_block_vulnerability(layer_scores, blocks):
    """Aggregate layer scores to block scores."""
    block_scores = {}
    for block_name, layer_names in blocks.items():
        scores = []
        for layer_name in layer_names:
            for ln, score in layer_scores.items():
                if layer_name in ln:
                    scores.append(score)
        if scores:
            block_scores[block_name] = sum(scores) / len(scores)
    return block_scores


def freeze_blocks(model, retrain_blocks, arch):
    """Freeze all blocks except retrain_blocks."""
    blocks = get_block_level_layers(arch)
    
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze selected blocks
    for name, param in model.named_parameters():
        for block_name in retrain_blocks:
            for layer_pattern in blocks.get(block_name, []):
                if layer_pattern in name:
                    param.requires_grad = True
                    print(f"  Unfreezing: {name}")
                    break
    
    return model


def train_block_level(model, train_loader, val_loader, device, epochs=20, lr=0.01, max_norm=1.0):
    """Train with block-level intervention."""
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f'Block-Level Epoch {epoch+1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return model, history, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--top_k_blocks', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./models')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vulnerability scores
    vuln_path = f'results/{args.dataset}_{args.arch}_seed{args.seed}_vulnerability.json'
    with open(vuln_path, 'r') as f:
        vuln_data = json.load(f)
    layer_scores = vuln_data['layer_scores']
    
    # Get blocks and compute block scores
    blocks = get_block_level_layers(args.arch)
    block_scores = compute_block_vulnerability(layer_scores, blocks)
    
    print("Block Vulnerability Scores:")
    for block, score in sorted(block_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {block}: {score:.4f}")
    
    # Select top-k blocks
    sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)
    retrain_blocks = [b for b, _ in sorted_blocks[:args.top_k_blocks]]
    print(f"\nSelected blocks for retraining: {retrain_blocks}")
    
    # Load model
    num_classes = 10 if args.dataset == 'cifar10' else (100 if args.dataset == 'cifar100' else 100)
    input_dim = 600 if args.dataset == 'purchase100' else None
    model = get_model(args.arch, num_classes, input_dim)
    model.load_state_dict(torch.load(args.model_path))
    
    # Freeze blocks
    model = freeze_blocks(model, retrain_blocks, args.arch)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Load data
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        args.dataset, args.data_dir, batch_size=128, num_workers=4, seed=args.seed
    )
    
    # Train
    print("\nTraining with block-level intervention...")
    model, history, best_val_acc = train_block_level(
        model, train_loader, val_loader, device, epochs=args.epochs
    )
    
    # Evaluate
    test_acc, _, _ = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy after Block-Level: {test_acc:.2f}%")
    
    # Save
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.arch}_seed{args.seed}_blocklevel.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    results = {
        'experiment': f'{args.dataset}_{args.arch}_seed{args.seed}_blocklevel',
        'config': vars(args),
        'retrain_blocks': retrain_blocks,
        'trainable_params': trainable_params,
        'metrics': {
            'test_accuracy': test_acc,
            'val_accuracy': best_val_acc
        },
        'history': history
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.dataset}_{args.arch}_seed{args.seed}_blocklevel.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
