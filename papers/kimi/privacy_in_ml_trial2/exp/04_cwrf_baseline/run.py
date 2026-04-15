"""
Weight-level baseline (CWRF-style): Selective weight rewinding.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared.models import get_model
from shared.data_loader import get_data_loaders
from shared.utils import set_seed, evaluate_model


def compute_weight_vulnerability(model, member_loader, non_member_loader, device, max_samples=50):
    """Compute per-weight vulnerability scores."""
    model = model.to(device)
    model.eval()
    
    weight_scores = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        member_grads = []
        non_member_grads = []
        
        count = 0
        for inputs, targets in member_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            for i in range(len(inputs)):
                if count >= max_samples:
                    break
                model.zero_grad()
                output = model(inputs[i:i+1])
                loss = nn.CrossEntropyLoss()(output, targets[i:i+1])
                loss.backward(retain_graph=True)
                
                if param.grad is not None:
                    member_grads.append(param.grad.abs().flatten().cpu().numpy())
                count += 1
            if count >= max_samples:
                break
        
        count = 0
        for inputs, targets in non_member_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            for i in range(len(inputs)):
                if count >= max_samples:
                    break
                model.zero_grad()
                output = model(inputs[i:i+1])
                loss = nn.CrossEntropyLoss()(output, targets[i:i+1])
                loss.backward(retain_graph=True)
                
                if param.grad is not None:
                    non_member_grads.append(param.grad.abs().flatten().cpu().numpy())
                count += 1
            if count >= max_samples:
                break
        
        if member_grads and non_member_grads:
            member_mean = np.mean(member_grads, axis=0)
            non_member_mean = np.mean(non_member_grads, axis=0)
            diff = np.abs(member_mean - non_member_mean)
            weight_scores[name] = diff
    
    return weight_scores


def select_top_k_weights(weight_scores, sparsity=0.005):
    """Select top-k% most vulnerable weights."""
    # Flatten all scores
    all_scores = []
    all_names = []
    all_indices = []
    
    for name, scores in weight_scores.items():
        flat_scores = scores.flatten()
        for idx, score in enumerate(flat_scores):
            all_scores.append(score)
            all_names.append(name)
            all_indices.append(idx)
    
    all_scores = np.array(all_scores)
    k = int(len(all_scores) * sparsity)
    
    # Get top-k indices
    top_k_indices = np.argsort(all_scores)[-k:]
    
    # Group by parameter name
    selected_weights = {}
    for idx in top_k_indices:
        name = all_names[idx]
        param_idx = all_indices[idx]
        if name not in selected_weights:
            selected_weights[name] = []
        selected_weights[name].append(param_idx)
    
    return selected_weights


def freeze_selected_weights(model, selected_weights):
    """Freeze selected vulnerable weights, train others."""
    # Create mask for each parameter
    for name, param in model.named_parameters():
        if name in selected_weights:
            # Create a mask where selected weights are frozen (False)
            mask = torch.ones_like(param, dtype=torch.bool)
            flat_mask = mask.flatten()
            for idx in selected_weights[name]:
                flat_mask[idx] = False
            mask = flat_mask.reshape(param.shape)
            
            # Register hook to zero out gradients for selected weights
            def make_hook(m):
                def hook(grad):
                    return grad * m.float()
                return hook
            
            param.register_hook(make_hook(mask))
    
    return model


def train_weight_level(model, train_loader, val_loader, device, epochs=20, 
                       lr=0.01, max_norm=1.0, lambda_ent=0.1):
    """Train with weight-level intervention."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f'CWRF Epoch {epoch+1}/{epochs}'):
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
    parser.add_argument('--sparsity', type=float, default=0.005, help='Fraction of weights to select')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./models')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    num_classes = 10 if args.dataset == 'cifar10' else (100 if args.dataset == 'cifar100' else 100)
    input_dim = 600 if args.dataset == 'purchase100' else None
    model = get_model(args.arch, num_classes, input_dim)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    
    # Load data
    train_loader, val_loader, test_loader, full_train = get_data_loaders(
        args.dataset, args.data_dir, batch_size=128, num_workers=4, seed=args.seed
    )
    
    # Create shadow datasets
    from torch.utils.data import Subset, DataLoader
    n_total = len(full_train)
    n_shadow = int(0.1 * n_total)
    
    g = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(n_total, generator=g)
    shadow_in_dataset = Subset(full_train, indices[:n_shadow].tolist())
    shadow_out_dataset = Subset(full_train, indices[n_shadow:2*n_shadow].tolist())
    shadow_in_loader = DataLoader(shadow_in_dataset, batch_size=128, shuffle=False)
    shadow_out_loader = DataLoader(shadow_out_dataset, batch_size=128, shuffle=False)
    
    # Compute weight vulnerability
    print("Computing weight-level vulnerability scores...")
    weight_scores = compute_weight_vulnerability(model, shadow_in_loader, shadow_out_loader, device)
    
    # Select top-k weights
    selected_weights = select_top_k_weights(weight_scores, sparsity=args.sparsity)
    total_selected = sum(len(v) for v in selected_weights.values())
    print(f"Selected {total_selected} weights ({args.sparsity*100:.2f}%)")
    
    # Rewind selected weights (reset to small random values)
    state_dict = model.state_dict()
    for name, param in model.named_parameters():
        if name in selected_weights:
            # Reset to small random values
            mask = torch.zeros_like(param)
            flat_mask = mask.flatten()
            for idx in selected_weights[name]:
                flat_mask[idx] = 1
            mask = flat_mask.reshape(param.shape)
            
            # Reset masked weights
            new_param = param * (1 - mask) + torch.randn_like(param) * 0.01 * mask
            state_dict[name] = new_param
    
    model.load_state_dict(state_dict)
    
    # Freeze selected weights
    model = freeze_selected_weights(model, selected_weights)
    
    # Train
    print("\nTraining with weight-level intervention...")
    model, history, best_val_acc = train_weight_level(
        model, train_loader, val_loader, device, epochs=args.epochs
    )
    
    # Evaluate
    test_acc, _, _ = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy after CWRF: {test_acc:.2f}%")
    
    # Save
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.arch}_seed{args.seed}_cwrf.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    results = {
        'experiment': f'{args.dataset}_{args.arch}_seed{args.seed}_cwrf',
        'config': vars(args),
        'selected_weights': total_selected,
        'metrics': {
            'test_accuracy': test_acc,
            'val_accuracy': best_val_acc
        },
        'history': history
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.dataset}_{args.arch}_seed{args.seed}_cwrf.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
