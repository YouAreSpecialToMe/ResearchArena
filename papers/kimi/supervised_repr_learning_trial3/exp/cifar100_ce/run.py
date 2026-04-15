"""Cross-Entropy Baseline for CIFAR-100-LT."""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../shared'))
import torch
import torch.nn as nn
from models import create_cross_entropy_model
from data_loader import get_cifar_lt_dataloaders
from utils import set_seed, train_epoch_ce, evaluate_accuracy, get_optimizer, get_scheduler, save_results, save_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--imbalance_factor', type=int, default=100, choices=[10, 50, 100])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='resnet32')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    exp_name = f"{args.dataset}_ce_if{args.imbalance_factor}_seed{args.seed}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Loading {args.dataset.upper()}-LT (IF={args.imbalance_factor})...")
    train_loader, test_loader, cls_num_list = get_cifar_lt_dataloaders(
        dataset=args.dataset, imbalance_factor=args.imbalance_factor, batch_size=args.batch_size, seed=args.seed
    )
    num_classes = 100 if args.dataset == 'cifar100' else 10
    
    model = create_cross_entropy_model(args.model, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args.epochs, mode='cosine')
    
    print(f"Training for {args.epochs} epochs...")
    start_time = time.time()
    best_acc, best_balanced_acc = 0.0, 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_overall': [], 'test_balanced': [], 'test_many': [], 'test_medium': [], 'test_few': []}
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_ce(model, train_loader, criterion, optimizer, device, epoch)
        test_metrics = evaluate_accuracy(model, test_loader, device, num_classes)
        scheduler.step()
        
        for k, v in zip(history.keys(), [train_loss, train_acc, test_metrics['overall'], test_metrics['balanced'], 
                                          test_metrics['many_shot'], test_metrics['medium_shot'], test_metrics['few_shot']]):
            history[k].append(v)
        
        if test_metrics['overall'] > best_acc:
            best_acc = test_metrics['overall']
            save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(output_dir, 'best_model.pth'))
        if test_metrics['balanced'] > best_balanced_acc:
            best_balanced_acc = test_metrics['balanced']
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss={train_loss:.4f}, "
                  f"Overall={test_metrics['overall']:.2f}%, Balanced={test_metrics['balanced']:.2f}%, "
                  f"Many={test_metrics['many_shot']:.2f}%, Medium={test_metrics['medium_shot']:.2f}%, "
                  f"Few={test_metrics['few_shot']:.2f}%")
    
    training_time = time.time() - start_time
    results = {
        'experiment': exp_name, 'config': vars(args), 'best_overall_acc': best_acc,
        'best_balanced_acc': best_balanced_acc, 'final_metrics': test_metrics,
        'history': history, 'training_time_minutes': training_time / 60, 'cls_num_list': cls_num_list
    }
    save_results(results, os.path.join(output_dir, 'results.json'))
    print(f"Done! Best Overall: {best_acc:.2f}%, Balanced: {best_balanced_acc:.2f}%")
    return results

if __name__ == '__main__':
    main()
