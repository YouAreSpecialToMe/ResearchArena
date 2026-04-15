"""
Full fine-tuning baseline: retrain entire model with privacy-aware regularization.
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


def train_full_finetune(model, train_loader, val_loader, device, epochs=20, 
                         lr=0.001, max_norm=1.0, lambda_ent=0.1):
    """Full fine-tuning with privacy-aware regularization."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f'Full-FT Epoch {epoch+1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # CE loss
            ce_loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Entropy maximization
            probs = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            entropy_loss = -entropy.mean()
            
            loss = ce_loss + lambda_ent * entropy_loss
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
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
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
    
    # Load data
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        args.dataset, args.data_dir, batch_size=128, num_workers=4, seed=args.seed
    )
    
    # Train
    print("\nTraining with full fine-tuning...")
    model, history, best_val_acc = train_full_finetune(
        model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr
    )
    
    # Evaluate
    test_acc, _, _ = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy after Full FT: {test_acc:.2f}%")
    
    # Save
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.arch}_seed{args.seed}_fullft.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    results = {
        'experiment': f'{args.dataset}_{args.arch}_seed{args.seed}_fullft',
        'config': vars(args),
        'metrics': {
            'test_accuracy': test_acc,
            'val_accuracy': best_val_acc
        },
        'history': history
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.dataset}_{args.arch}_seed{args.seed}_fullft.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
