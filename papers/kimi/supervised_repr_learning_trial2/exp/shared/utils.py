"""
Utility functions for training and evaluation.
"""
import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from tqdm import tqdm


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


@torch.no_grad()
def extract_features(encoder, dataloader, device):
    """Extract features from encoder for all samples in dataloader."""
    encoder.eval()
    all_features = []
    all_labels = []
    
    for images, labels, _ in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        features = encoder.get_features(images)
        all_features.append(features.cpu())
        all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_labels


@torch.no_grad()
def knn_evaluate(encoder, train_loader, test_loader, device, k=20):
    """Evaluate encoder using k-NN classifier."""
    encoder.eval()
    
    # Extract features
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)
    
    # Normalize features
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    
    # Compute pairwise distances
    # Use batching to avoid OOM
    batch_size = 1000
    num_test = test_features.shape[0]
    correct = 0
    total = 0
    
    for i in range(0, num_test, batch_size):
        batch_test = test_features[i:i+batch_size].to(device)
        
        # Compute similarities
        similarities = torch.matmul(batch_test, train_features.T.to(device))
        
        # Get k nearest neighbors
        _, indices = similarities.topk(k, dim=1, largest=True)
        
        # Vote
        nearest_labels = train_labels[indices.cpu()]
        predictions = []
        for j in range(nearest_labels.shape[0]):
            labels = nearest_labels[j].numpy()
            # Majority vote
            unique, counts = np.unique(labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        
        predictions = torch.tensor(predictions)
        correct += (predictions == test_labels[i:i+batch_size]).sum().item()
        total += len(predictions)
    
    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def linear_evaluate(encoder, train_loader, test_loader, device, num_classes,
                    epochs=100, lr=0.1, momentum=0.9, weight_decay=0):
    """
    Evaluate encoder using linear classifier.
    Freeze encoder and train a linear classifier on top.
    """
    from shared.models import LinearClassifier
    import torch.optim as optim
    import torch.nn.functional as F
    
    encoder.eval()
    
    # Get feature dimension
    sample_images, _, _ = next(iter(train_loader))
    sample_images = sample_images.to(device)
    with torch.no_grad():
        sample_features = encoder.get_features(sample_images)
    feature_dim = sample_features.shape[1]
    
    # Create linear classifier
    classifier = LinearClassifier(feature_dim, num_classes).to(device)
    
    # Extract all features (more efficient than extracting every epoch)
    print("Extracting train features...")
    train_features, train_labels = extract_features(encoder, train_loader, device)
    print("Extracting test features...")
    test_features, test_labels = extract_features(encoder, test_loader, device)
    
    # Create dataset from features
    train_feat_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_feat_loader = torch.utils.data.DataLoader(
        train_feat_dataset, batch_size=256, shuffle=True, num_workers=4
    )
    
    test_feat_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_feat_loader = torch.utils.data.DataLoader(
        test_feat_dataset, batch_size=256, shuffle=False, num_workers=4
    )
    
    # Optimizer and scheduler
    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Train classifier
    best_acc = 0.0
    for epoch in range(epochs):
        classifier.train()
        train_loss = 0.0
        
        for features, labels in train_feat_loader:
            features, labels = features.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate
        classifier.eval()
        correct = 0
        total = 0
        
        for features, labels in test_feat_loader:
            features, labels = features.to(device), labels.to(device).long()
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)
        
        if (epoch + 1) % 20 == 0:
            print(f"Linear Eval Epoch {epoch+1}/{epochs}: Acc={acc:.2f}%, Best={best_acc:.2f}%")
    
    return best_acc


def save_results(results, path):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(path):
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


import torch.nn.functional as F
