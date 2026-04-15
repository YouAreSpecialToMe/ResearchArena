"""
PRISM: Post-hoc Representation Intervention via Selective Memorization Mitigation.
Layer-level selective retraining.
"""
import os
import sys
import json
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared.models import get_model
from shared.data_loader import get_data_loaders
from shared.utils import set_seed, evaluate_model


def load_vulnerability_scores(dataset, arch, seed):
    """Load pre-computed vulnerability scores."""
    vuln_path = f'results/{dataset}_{arch}_seed{seed}_vulnerability.json'
    with open(vuln_path, 'r') as f:
        data = json.load(f)
    return data['layer_scores'], data['sorted_layers']


def get_layers_to_retrain(sorted_layers, top_k_percent=30):
    """Select top-k% most vulnerable layers for retraining."""
    n_layers = len(sorted_layers)
    n_retrain = max(1, int(n_layers * top_k_percent / 100))
    retrain_layers = [layer for layer, _ in sorted_layers[:n_retrain]]
    return retrain_layers


def freeze_layers(model, retrain_layers, model_type='resnet18'):
    """Freeze all layers except those in retrain_layers."""
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze selected layers
    for name, param in model.named_parameters():
        for layer_name in retrain_layers:
            if layer_name in name:
                param.requires_grad = True
                print(f"  Unfreezing: {name}")
                break
    
    return model


class EntropyLoss(nn.Module):
    """Entropy maximization loss."""
    def __init__(self, lambda_ent=0.1):
        super().__init__()
        self.lambda_ent = lambda_ent
    
    def forward(self, outputs, targets):
        # Standard CE loss
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Entropy maximization (negative entropy)
        probs = torch.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        entropy_loss = -entropy.mean()  # Negative to maximize entropy
        
        return ce_loss + self.lambda_ent * entropy_loss


class DistillationLoss(nn.Module):
    """Knowledge distillation loss."""
    def __init__(self, teacher_model, lambda_kd=1.0, temperature=4):
        super().__init__()
        self.teacher_model = teacher_model
        self.lambda_kd = lambda_kd
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_outputs, targets, inputs):
        # Standard CE loss
        ce_loss = self.ce_loss(student_outputs, targets)
        
        # Distillation loss
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        
        student_probs = torch.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_probs = torch.softmax(teacher_outputs / self.temperature, dim=1)
        
        kd_loss = nn.KLDivLoss(reduction='batchmean')(student_probs, teacher_probs)
        kd_loss = kd_loss * (self.temperature ** 2)
        
        return ce_loss + self.lambda_kd * kd_loss


def train_with_gradient_clipping(model, train_loader, val_loader, device, epochs=20, 
                                  lr=0.01, max_norm=1.0, lambda_ent=0.1, 
                                  teacher_model=None, lambda_kd=1.0):
    """Train model with gradient clipping and privacy-aware regularization."""
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Loss function
    if teacher_model is not None:
        criterion = DistillationLoss(teacher_model, lambda_kd=lambda_kd)
    else:
        criterion = EntropyLoss(lambda_ent=lambda_ent)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f'PRISM Epoch {epoch+1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if teacher_model is not None:
                loss = criterion(outputs, targets, inputs)
            else:
                loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return model, history, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--top_k_percent', type=float, default=30)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--lambda_ent', type=float, default=0.1)
    parser.add_argument('--lambda_kd', type=float, default=1.0)
    parser.add_argument('--use_kd', action='store_true')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./models')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load original model as teacher if using KD
    num_classes = 10 if args.dataset == 'cifar10' else (100 if args.dataset == 'cifar100' else 100)
    input_dim = 600 if args.dataset == 'purchase100' else None
    
    teacher_model = None
    if args.use_kd:
        teacher_model = get_model(args.arch, num_classes, input_dim)
        teacher_model.load_state_dict(torch.load(args.model_path))
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
    
    # Load model to apply PRISM
    model = get_model(args.arch, num_classes, input_dim)
    model.load_state_dict(torch.load(args.model_path))
    
    # Load vulnerability scores
    layer_scores, sorted_layers = load_vulnerability_scores(args.dataset, args.arch, args.seed)
    print(f"Layer vulnerability scores loaded. Layers: {list(layer_scores.keys())}")
    
    # Select layers to retrain
    retrain_layers = get_layers_to_retrain(sorted_layers, args.top_k_percent)
    print(f"\nSelected layers for retraining (top {args.top_k_percent}%): {retrain_layers}")
    
    # Freeze non-vulnerable layers
    model = freeze_layers(model, retrain_layers, args.arch)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Load data
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        args.dataset, args.data_dir, batch_size=128, num_workers=4, seed=args.seed
    )
    
    # Apply PRISM retraining
    print("\nApplying PRISM selective retraining...")
    model, history, best_val_acc = train_with_gradient_clipping(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, max_norm=args.max_norm,
        lambda_ent=args.lambda_ent, teacher_model=teacher_model, lambda_kd=args.lambda_kd
    )
    
    # Evaluate
    test_acc, test_probs, test_labels = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy after PRISM: {test_acc:.2f}%")
    
    # Save model
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.arch}_seed{args.seed}_prism.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    # Save results
    results = {
        'experiment': f'{args.dataset}_{args.arch}_seed{args.seed}_prism',
        'config': vars(args),
        'retrain_layers': retrain_layers,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'metrics': {
            'test_accuracy': test_acc,
            'val_accuracy': best_val_acc
        },
        'history': history
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.dataset}_{args.arch}_seed{args.seed}_prism.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved to {save_path}")
    print(f"Results saved to results/{args.dataset}_{args.arch}_seed{args.seed}_prism.json")


if __name__ == '__main__':
    main()
