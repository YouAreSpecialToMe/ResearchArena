"""
Train SCL on CIFAR-100 with coarse labels (20 superclasses),
then evaluate on both coarse and fine labels.
This tests feature suppression by checking if the model retains
fine-grained features despite only being trained on coarse labels.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from shared.data_loader import get_cifar100_loaders, get_cifar100_fine_labels_only
from shared.models import create_model, LinearClassifier
from shared.losses import SupervisedContrastiveLoss
from shared.metrics import linear_probe_accuracy, extract_embeddings
from shared.utils import set_seed, save_results, save_checkpoint, Timer


def train_encoder(model, train_loader, criterion, optimizer, device, epoch):
    """Train encoder with contrastive loss."""
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


def evaluate_linear_probe(encoder, train_loader, test_loader, test_loader_fine, 
                          device, feature_dim, num_classes_coarse, num_classes_fine,
                          classifier_epochs, lr):
    """
    Train linear classifiers on frozen features for both coarse and fine labels.
    Returns both accuracies.
    """
    # Coarse label evaluation
    print("Evaluating on coarse labels...")
    coarse_acc = linear_probe_accuracy(
        encoder, train_loader, test_loader, device,
        feature_dim=feature_dim, num_classes=num_classes_coarse,
        epochs=classifier_epochs, lr=lr
    )
    print(f"Coarse label accuracy: {coarse_acc:.2f}%")
    
    # Fine label evaluation (key metric for feature suppression)
    print("Evaluating on fine labels (100 classes)...")
    fine_acc = linear_probe_accuracy(
        encoder, train_loader, test_loader_fine, device,
        feature_dim=feature_dim, num_classes=num_classes_fine,
        epochs=classifier_epochs, lr=lr
    )
    print(f"Fine label accuracy: {fine_acc:.2f}%")
    
    return coarse_acc, fine_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--encoder_epochs', type=int, default=150)
    parser.add_argument('--classifier_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--method', type=str, default='scl', choices=['scl', 'fdscl'])
    parser.add_argument('--weight_temperature', type=float, default=0.5)
    parser.add_argument('--activation_threshold', type=float, default=0.0)
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Method: {args.method.upper()}")
    
    # Data loaders with coarse labels for training
    train_loader, test_loader_coarse, num_classes_coarse = get_cifar100_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        contrastive=True,
        use_coarse_labels=True  # Train with 20 coarse labels
    )
    
    # For evaluation, we need non-contrastive loaders
    train_loader_eval, _, _ = get_cifar100_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        contrastive=False,
        use_coarse_labels=True
    )
    
    # Test loader with fine labels (100 classes) - key for feature suppression evaluation
    test_loader_fine, num_classes_fine = get_cifar100_fine_labels_only(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print(f"Coarse classes: {num_classes_coarse}, Fine classes: {num_classes_fine}")
    
    # Model
    model = create_model(num_classes=num_classes_coarse, use_projection_head=True, projection_dim=128)
    model = model.to(device)
    
    # Loss
    if args.method == 'scl':
        from shared.losses import SupervisedContrastiveLoss
        criterion = SupervisedContrastiveLoss(temperature=args.temperature)
    else:  # fdscl
        from shared.losses import FeatureDiversitySCLLoss
        criterion = FeatureDiversitySCLLoss(
            temperature=args.temperature,
            weight_temperature=args.weight_temperature,
            activation_threshold=args.activation_threshold,
            return_weights=False
        )
    
    # Optimizer
    scaled_lr = args.lr * args.batch_size / 256
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Warmup + Cosine annealing
    def lr_schedule(epoch):
        if epoch < 10:
            return (epoch + 1) / 10
        else:
            return 0.5 * (1 + torch.cos(torch.tensor((epoch - 10) / (args.encoder_epochs - 10) * 3.14159)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    
    # Training encoder
    timer = Timer()
    timer.start()
    
    print(f"\n=== Training {args.method.upper()} Encoder on Coarse Labels ===")
    encoder_losses = []
    
    for epoch in range(args.encoder_epochs):
        loss = train_encoder(model, train_loader, criterion, optimizer, device, epoch)
        encoder_losses.append(loss)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.encoder_epochs} - Loss: {loss:.4f}")
    
    encoder_time = timer.get_elapsed()
    
    # Save encoder
    save_checkpoint(
        model, optimizer, args.encoder_epochs,
        f"{args.save_dir}/{args.method}_coarse_seed{args.seed}.pth",
    )
    
    # Linear evaluation on both coarse and fine labels
    print(f"\n=== Linear Evaluation ===")
    coarse_acc, fine_acc = evaluate_linear_probe(
        model, train_loader_eval, test_loader_coarse, test_loader_fine,
        device, feature_dim=512, num_classes_coarse=num_classes_coarse, 
        num_classes_fine=num_classes_fine,
        classifier_epochs=args.classifier_epochs, lr=1.0
    )
    
    total_time = timer.stop()
    
    # Save results
    results = {
        'experiment': f'{args.method}_coarse_to_fine',
        'seed': args.seed,
        'method': args.method,
        'encoder_epochs': args.encoder_epochs,
        'coarse_label_accuracy': coarse_acc,
        'fine_label_accuracy': fine_acc,
        'fine_coarse_gap': fine_acc - coarse_acc,
        'encoder_losses': encoder_losses,
        'encoder_time_seconds': encoder_time,
        'total_time_seconds': total_time,
        'config': vars(args)
    }
    
    os.makedirs('./results', exist_ok=True)
    save_results(results, f'./results/{args.method}_coarse_to_fine_seed{args.seed}.json')
    
    print(f"\n{'='*50}")
    print(f"Results for {args.method.upper()} (Seed {args.seed}):")
    print(f"Coarse Label Accuracy: {coarse_acc:.2f}%")
    print(f"Fine Label Accuracy: {fine_acc:.2f}%")
    print(f"Training Time: {total_time:.1f}s")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
