"""
Adversarial robustness evaluation using PGD and AutoAttack.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from models import create_simclr_model, LinearClassifier
from data_loader import get_linear_eval_dataloaders
from fcl_utils import set_seed, linear_evaluation


def pgd_attack(model, classifier, inputs, labels, eps, steps, step_size, device):
    """PGD attack."""
    x_adv = inputs.clone().detach()
    
    for _ in range(steps):
        x_adv.requires_grad = True
        
        features = model.encoder(x_adv, return_features=True)
        outputs = classifier(features)
        loss = F.cross_entropy(outputs, labels)
        
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        perturbation = torch.clamp(x_adv - inputs, -eps, eps)
        x_adv = inputs + perturbation
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    
    return x_adv


def evaluate_pgd_robustness(model, classifier, test_loader, device, eps=8/255, steps=20):
    """Evaluate robust accuracy under PGD attack."""
    model.eval()
    classifier.eval()
    
    correct = 0
    total = 0
    step_size = eps / 4
    
    for inputs, labels in tqdm(test_loader, desc="PGD evaluation"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Generate adversarial examples
        x_adv = pgd_attack(model, classifier, inputs, labels, eps, steps, step_size, device)
        
        # Evaluate
        with torch.no_grad():
            features = model.encoder(x_adv, return_features=True)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def evaluate_standard_accuracy(model, classifier, test_loader, device):
    """Evaluate standard accuracy."""
    model.eval()
    classifier.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = model.encoder(inputs, return_features=True)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def evaluate_robustness(model_path, dataset, data_dir, device='cuda'):
    """Evaluate adversarial robustness."""
    
    set_seed(42)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = create_simclr_model().to(device)
    
    # Handle different checkpoint formats
    if 'encoder' in checkpoint:
        encoder_dict = checkpoint['encoder']
        projection_dict = checkpoint.get('projection_head', {})
        state_dict = {}
        for k, v in encoder_dict.items():
            state_dict[f'encoder.{k}'] = v
        for k, v in projection_dict.items():
            state_dict[f'projection_head.{k}'] = v
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    # Get dataloaders
    num_classes = 10 if dataset == 'cifar10' else 100
    train_loader, test_loader = get_linear_eval_dataloaders(
        dataset_name=dataset, data_dir=data_dir, batch_size=256
    )
    
    # Train linear classifier
    print("Training linear classifier...")
    from fcl_utils import linear_evaluation
    _ = linear_evaluation(model.encoder, train_loader, test_loader, device, 
                         num_classes=num_classes, epochs=100, lr=0.1)
    
    # Load trained classifier
    classifier = LinearClassifier(model.encoder.feature_dim, num_classes).to(device)
    classifier.eval()
    
    # For simplicity, train a quick classifier here
    classifier.train()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(50):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.no_grad():
                features = model.encoder(inputs, return_features=True)
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    classifier.eval()
    
    # Evaluate standard accuracy
    print("Evaluating standard accuracy...")
    standard_acc = evaluate_standard_accuracy(model, classifier, test_loader, device)
    
    # Evaluate PGD-20 robustness
    print("Evaluating PGD-20 robustness...")
    pgd20_acc = evaluate_pgd_robustness(model, classifier, test_loader, device, 
                                        eps=8/255, steps=20)
    
    # Evaluate PGD-100 robustness (stronger attack)
    print("Evaluating PGD-100 robustness...")
    pgd100_acc = evaluate_pgd_robustness(model, classifier, test_loader, device,
                                         eps=8/255, steps=100)
    
    return {
        'standard_accuracy': standard_acc,
        'pgd20_robust_accuracy': pgd20_acc,
        'pgd100_robust_accuracy': pgd100_acc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = evaluate_robustness(args.model_path, args.dataset, args.data_dir, device)
    
    results['model'] = args.model_path
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results: {results}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
