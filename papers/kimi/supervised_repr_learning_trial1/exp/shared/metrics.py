"""
Evaluation metrics and utilities for feature diversity analysis.
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def compute_effective_rank(covariance_matrix):
    """Compute effective rank of a covariance matrix.
    
    Effective rank = exp(H(p)) where H is Shannon entropy and p_i = lambda_i / sum(lambda)
    
    Args:
        covariance_matrix: (D, D) covariance matrix
        
    Returns:
        effective_rank: Scalar effective rank
    """
    eigenvalues = torch.linalg.eigvalsh(covariance_matrix)
    eigenvalues = eigenvalues.clamp(min=1e-10)  # Avoid numerical issues
    
    # Normalize eigenvalues
    p = eigenvalues / eigenvalues.sum()
    
    # Compute entropy
    entropy = -(p * torch.log(p)).sum()
    
    # Effective rank
    effective_rank = torch.exp(entropy).item()
    return effective_rank


def compute_class_effective_ranks(embeddings, labels):
    """Compute average effective rank across all classes.
    
    Args:
        embeddings: (N, D) tensor of embeddings
        labels: (N,) tensor of class labels
        
    Returns:
        avg_effective_rank: Average effective rank across classes
        per_class_erank: Dict of effective rank per class
    """
    unique_labels = torch.unique(labels)
    eranks = []
    per_class_erank = {}
    
    for c in unique_labels:
        mask = labels == c
        class_embeddings = embeddings[mask]
        
        if len(class_embeddings) > 1:
            # Center the embeddings
            class_embeddings = class_embeddings - class_embeddings.mean(dim=0)
            
            # Compute covariance
            cov = torch.mm(class_embeddings.T, class_embeddings) / (len(class_embeddings) - 1)
            
            # Compute effective rank
            erank = compute_effective_rank(cov)
            eranks.append(erank)
            per_class_erank[int(c.item())] = erank
    
    avg_erank = np.mean(eranks) if eranks else 0.0
    return avg_erank, per_class_erank


def compute_participation_ratio(embeddings, labels):
    """Compute participation ratio as another diversity metric.
    
    PR = (sum(lambda_i))^2 / sum(lambda_i^2)
    
    Args:
        embeddings: (N, D) tensor of embeddings
        labels: (N,) tensor of class labels
        
    Returns:
        avg_pr: Average participation ratio across classes
    """
    unique_labels = torch.unique(labels)
    prs = []
    
    for c in unique_labels:
        mask = labels == c
        class_embeddings = embeddings[mask]
        
        if len(class_embeddings) > 1:
            class_embeddings = class_embeddings - class_embeddings.mean(dim=0)
            cov = torch.mm(class_embeddings.T, class_embeddings) / (len(class_embeddings) - 1)
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues.clamp(min=1e-10)
            
            pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
            prs.append(pr.item())
    
    return np.mean(prs) if prs else 0.0


def knn_evaluation(train_embeddings, train_labels, test_embeddings, test_labels, k=200):
    """Evaluate using k-NN classifier.
    
    Args:
        train_embeddings: (N_train, D) numpy array
        train_labels: (N_train,) numpy array
        test_embeddings: (N_test, D) numpy array
        test_labels: (N_test,) numpy array
        k: Number of neighbors
        
    Returns:
        accuracy: k-NN accuracy
    """
    # Normalize embeddings
    train_embeddings = F.normalize(train_embeddings, dim=1).numpy()
    test_embeddings = F.normalize(test_embeddings, dim=1).numpy()
    train_labels = train_labels.numpy()
    test_labels = test_labels.numpy()
    
    # k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=min(k, len(train_embeddings) - 1), 
                               metric='cosine', n_jobs=4)
    knn.fit(train_embeddings, train_labels)
    predictions = knn.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, predictions)
    
    return accuracy


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from a model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to use
        
    Returns:
        embeddings: (N, D) tensor of embeddings
        labels: (N,) tensor of labels
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, _ = batch  # Coarse label case
            else:
                images, labels = batch
            
            images = images.to(device)
            
            # Get embeddings (from backbone if contrastive model)
            if hasattr(model, 'backbone'):
                embeddings = model.backbone(images)
            else:
                embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return embeddings, labels


def evaluate_linear_classifier(classifier, encoder, dataloader, device):
    """Evaluate linear classifier on frozen features.
    
    Args:
        classifier: Linear classifier model
        encoder: Encoder model (for feature extraction)
        dataloader: DataLoader
        device: Device
        
    Returns:
        accuracy: Classification accuracy
    """
    classifier.eval()
    if encoder is not None:
        encoder.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Get features
            if encoder is not None:
                features = encoder.backbone(images) if hasattr(encoder, 'backbone') else encoder(images)
            else:
                features = images
            
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def compute_weight_rarity_correlation(weights, rarity_scores):
    """Compute correlation between weights and rarity scores.
    
    Args:
        weights: Array of pair weights
        rarity_scores: Array of rarity scores
        
    Returns:
        correlation: Pearson correlation coefficient
        p_value: p-value for correlation
    """
    if len(weights) < 2:
        return 0.0, 1.0
    
    corr, p_val = pearsonr(weights, rarity_scores)
    return corr, p_val


def compute_accuracy(model, dataloader, device):
    """Compute classification accuracy.
    
    Args:
        model: Model with final classifier
        dataloader: DataLoader
        device: Device
        
    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy
