"""
Evaluation metrics for classification and calibration.
"""
import numpy as np
import torch
import torch.nn.functional as F


def accuracy(outputs, targets):
    """Compute classification accuracy."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def expected_calibration_error(confidences, predictions, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        confidences: Max softmax probabilities (numpy array)
        predictions: Predicted class indices (numpy array)
        labels: True labels (numpy array)
        n_bins: Number of bins for calibration
    
    Returns:
        ece: Expected calibration error (percentage)
    """
    confidences = np.array(confidences)
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    correct = (predictions == labels).astype(float)
    
    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * in_bin.mean()
    
    return 100.0 * ece


def evaluate_model(model, loader, device):
    """
    Evaluate model on a dataset.
    
    Returns:
        dict with accuracy, ECE, and other metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_confidences = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:  # NoisyCIFAR100
                images, labels, _, _, _ = batch
            else:  # Standard dataset
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Get features and classify
            if hasattr(model, 'forward') and 'return_features' in model.forward.__code__.co_varnames:
                _, features = model(images, return_features=True)
                if hasattr(model, 'get_classifier_logits'):
                    logits = model.get_classifier_logits(features)
                else:
                    logits = model.classifier(features)
            else:
                logits = model(images)
            
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Compute metrics
    acc = 100.0 * (all_preds == all_labels).mean()
    ece = expected_calibration_error(all_confidences, all_preds, all_labels)
    
    return {
        'accuracy': acc,
        'ece': ece,
        'predictions': all_preds,
        'labels': all_labels,
        'confidences': all_confidences
    }


def linear_evaluation(encoder, train_loader, test_loader, device, 
                     num_classes=100, epochs=50, lr=0.1):
    """
    Linear evaluation protocol: train a linear classifier on frozen features.
    
    Returns:
        Test accuracy of linear classifier
    """
    encoder.eval()
    
    # Extract features
    def extract_features(loader):
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 5:
                    images, labels, _, _, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(device)
                _, features = encoder(images, return_features=True)
                features_list.append(features.cpu())
                labels_list.append(labels)
        
        return torch.cat(features_list), torch.cat(labels_list)
    
    train_features, train_labels = extract_features(train_loader)
    test_features, test_labels = extract_features(test_loader)
    
    # Train linear classifier
    feature_dim = train_features.shape[1]
    classifier = torch.nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create data loader for features
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader_features = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    for epoch in range(epochs):
        classifier.train()
        for features, labels in train_loader_features:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    # Evaluate
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        test_features = test_features.to(device)
        test_labels = test_labels.to(device)
        
        # Process in batches
        batch_size = 256
        for i in range(0, len(test_features), batch_size):
            features = test_features[i:i+batch_size]
            labels = test_labels[i:i+batch_size]
            
            outputs = classifier(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total
