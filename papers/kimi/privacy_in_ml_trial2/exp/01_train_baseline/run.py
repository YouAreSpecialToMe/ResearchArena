"""
Train baseline models (no defense) for PRISM experiments.
"""
import os
import sys
import json
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared.models import get_model
from shared.data_loader import get_data_loaders
from shared.utils import set_seed, train_model, evaluate_model, save_results
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'purchase100'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'vgg16', 'simplecnn'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./models')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loaders
    print(f"Loading {args.dataset}...")
    num_classes = 10 if args.dataset == 'cifar10' else (100 if args.dataset == 'cifar100' else 100)
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        args.dataset, args.data_dir, args.batch_size, num_workers=4, seed=args.seed
    )
    
    # Model
    print(f"Creating {args.arch} model...")
    input_dim = 600 if args.dataset == 'purchase100' else None
    model = get_model(args.arch, num_classes, input_dim)
    
    # Train
    print("Training baseline model...")
    start_time = time.time()
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.arch}_seed{args.seed}_baseline.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    history, best_val_acc = train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, save_path=save_path
    )
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(save_path))
    test_acc, test_probs, test_labels = evaluate_model(model, test_loader, device)
    
    runtime = (time.time() - start_time) / 60  # minutes
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Runtime: {runtime:.1f} minutes")
    
    # Save results
    results = {
        'experiment': f'{args.dataset}_{args.arch}_seed{args.seed}_baseline',
        'config': vars(args),
        'metrics': {
            'test_accuracy': {'mean': test_acc, 'std': 0.0},
            'val_accuracy': {'mean': best_val_acc, 'std': 0.0}
        },
        'runtime_minutes': runtime,
        'history': history
    }
    
    os.makedirs('results', exist_ok=True)
    save_results(results, f'results/{args.dataset}_{args.arch}_seed{args.seed}_baseline.json')
    print(f"Results saved to results/{args.dataset}_{args.arch}_seed{args.seed}_baseline.json")


if __name__ == '__main__':
    main()
