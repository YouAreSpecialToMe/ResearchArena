"""ETF-SCL: Equiangular Tight Frame guided Supervised Contrastive Learning."""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../shared'))
import torch
from models import create_model
from data_loader import get_cifar_lt_contrastive_dataloaders, get_cifar_lt_dataloaders
from losses import ETFSCLLoss
from utils import (set_seed, train_epoch_contrastive, evaluate_accuracy, linear_evaluation,
                  get_optimizer, get_scheduler, save_results, save_checkpoint, compute_geometric_metrics, extract_features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--imbalance_factor', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--model', type=str, default='resnet32')
    parser.add_argument('--tau_base', type=float, default=0.1)
    parser.add_argument('--tau_0', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lambda_etf', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    exp_name = f"{args.dataset}_etfscl_if{args.imbalance_factor}_seed{args.seed}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Loading {args.dataset.upper()}-LT (IF={args.imbalance_factor}) with contrastive augmentation...")
    train_loader, test_loader, cls_num_list = get_cifar_lt_contrastive_dataloaders(
        dataset=args.dataset, imbalance_factor=args.imbalance_factor, batch_size=args.batch_size, seed=args.seed
    )
    train_loader_eval, _, _ = get_cifar_lt_dataloaders(
        dataset=args.dataset, imbalance_factor=args.imbalance_factor, batch_size=args.batch_size, seed=args.seed
    )
    
    num_classes = 100 if args.dataset == 'cifar100' else 10
    
    model, feature_dim = create_model(args.model, num_classes, args.projection_dim, use_projection=True)
    model = model.to(device)
    
    criterion = ETFSCLLoss(
        num_classes=num_classes, cls_num_list=cls_num_list, feature_dim=feature_dim,
        tau_base=args.tau_base, tau_0=args.tau_0, alpha=args.alpha, beta=args.beta,
        lambda_etf=args.lambda_etf, total_epochs=args.epochs, device=device
    )
    optimizer = get_optimizer(model, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer, args.epochs, mode='cosine')
    
    print(f"Stage 1: Training encoder with ETF-SCL for {args.epochs} epochs...")
    start_time = time.time()
    best_loss = float('inf')
    history = {'total_loss': [], 'scl_loss': [], 'etf_loss': []}
    
    for epoch in range(args.epochs):
        result = train_epoch_contrastive(model, train_loader, criterion, optimizer, device, epoch, 'etf_scl')
        scheduler.step()
        
        for k in ['loss', 'scl_loss', 'etf_loss']:
            if k in result:
                history[k if k != 'loss' else 'total_loss'].append(result[k])
        
        if result['loss'] < best_loss:
            best_loss = result['loss']
            save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(output_dir, 'best_encoder.pth'))
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Total={result['loss']:.4f}, "
                  f"SCL={result.get('scl_loss', 0):.4f}, ETF={result.get('etf_loss', 0):.4f}")
    
    print("\nStage 2: Linear evaluation...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_encoder.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    linear_acc = linear_evaluation(
        model.encoder, train_loader_eval, test_loader, device, feature_dim, num_classes, epochs=50, lr=0.1
    )
    
    test_metrics = evaluate_accuracy(model.encoder, test_loader, device, num_classes)
    test_metrics['linear_eval_acc'] = linear_acc
    
    # Compute geometric metrics
    print("\nComputing geometric metrics...")
    features, labels = extract_features(model.encoder, test_loader, device)
    geo_metrics = compute_geometric_metrics(features, labels, num_classes)
    
    training_time = time.time() - start_time
    print(f"\nLinear Evaluation Accuracy: {linear_acc:.2f}%")
    print(f"Test Overall: {test_metrics['overall']:.2f}%, Balanced: {test_metrics['balanced']:.2f}%")
    print(f"ETF Deviation: {geo_metrics['etf_deviation']:.4f}")
    
    results = {
        'experiment': exp_name, 'config': vars(args), 'linear_eval_acc': linear_acc,
        'best_overall_acc': test_metrics['overall'], 'best_balanced_acc': test_metrics['balanced'],
        'final_metrics': test_metrics, 'geometric_metrics': geo_metrics,
        'history': history, 'training_time_minutes': training_time / 60, 'cls_num_list': cls_num_list
    }
    save_results(results, os.path.join(output_dir, 'results.json'))
    return results

if __name__ == '__main__':
    main()
