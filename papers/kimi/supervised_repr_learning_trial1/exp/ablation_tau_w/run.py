"""
Ablation study: Weight temperature sensitivity.
Test different values of tau_w for FD-SCL.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
import argparse

from shared.data_loader import get_cifar100_loaders
from shared.models import create_model
from shared.losses import FeatureDiversitySCLLoss
from shared.metrics import linear_probe_accuracy
from shared.utils import set_seed, save_results, Timer


def train_encoder(model, train_loader, criterion, optimizer, device, epoch):
    """Train encoder with FD-SCL loss."""
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        images = torch.cat([images[0], images[1]], dim=0).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        projections = model(images)
        loss = criterion(projections, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--encoder_epochs', type=int, default=50)
    parser.add_argument('--classifier_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--weight_temperature', type=float, default=0.5)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--use_coarse', action='store_true')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Weight Temperature: {args.weight_temperature}")
    
    # Data loaders
    train_loader, test_loader, num_classes = get_cifar100_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        contrastive=True,
        use_coarse_labels=args.use_coarse
    )
    
    train_loader_eval, _, _ = get_cifar100_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        contrastive=False,
        use_coarse_labels=args.use_coarse
    )
    
    # Model
    model = create_model(num_classes=num_classes, use_projection_head=True, projection_dim=128)
    model = model.to(device)
    
    # Loss
    criterion = FeatureDiversitySCLLoss(
        temperature=args.temperature,
        weight_temperature=args.weight_temperature,
        return_weights=False
    )
    
    # Optimizer
    scaled_lr = args.lr * args.batch_size / 256
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=args.weight_decay)
    
    def lr_schedule(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        else:
            return 0.5 * (1 + torch.cos(torch.tensor((epoch - 5) / (args.encoder_epochs - 5) * 3.14159)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    
    # Training
    timer = Timer()
    timer.start()
    
    encoder_losses = []
    
    for epoch in range(args.encoder_epochs):
        loss = train_encoder(model, train_loader, criterion, optimizer, device, epoch)
        encoder_losses.append(loss)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.encoder_epochs} - Loss: {loss:.4f}")
    
    # Linear evaluation
    linear_acc = linear_probe_accuracy(
        model, train_loader_eval, test_loader, device,
        feature_dim=512, num_classes=num_classes,
        epochs=args.classifier_epochs, lr=1.0
    )
    
    total_time = timer.stop()
    
    # Save results
    results = {
        'experiment': 'ablation_tau_w',
        'weight_temperature': args.weight_temperature,
        'seed': args.seed,
        'linear_eval_acc': linear_acc,
        'encoder_losses': encoder_losses,
        'total_time_seconds': total_time,
        'config': vars(args)
    }
    
    os.makedirs('./results', exist_ok=True)
    save_results(results, f'./results/ablation_tau_w_{args.weight_temperature}_seed{args.seed}.json')
    
    print(f"\nWeight Temperature {args.weight_temperature}: Accuracy {linear_acc:.2f}%, Time {total_time:.1f}s")


if __name__ == '__main__':
    main()
