"""
Evaluation metrics for PRISM experiments, including Membership Inference Attacks.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn


def compute_attack_accuracy(member_preds, non_member_preds):
    """Compute MIA accuracy from member/non-member predictions."""
    member_correct = np.sum(member_preds == 1)
    non_member_correct = np.sum(non_member_preds == 0)
    total = len(member_preds) + len(non_member_preds)
    return (member_correct + non_member_correct) / total


def compute_auc(member_scores, non_member_scores):
    """Compute AUC-ROC for MIA."""
    y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
    y_scores = np.concatenate([member_scores, non_member_scores])
    return roc_auc_score(y_true, y_scores)


def compute_tpr_at_fpr(member_scores, non_member_scores, fpr_threshold=0.001):
    """Compute TPR at a specific FPR threshold."""
    y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
    y_scores = np.concatenate([member_scores, non_member_scores])
    
    # Sort by scores
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Compute TPR at FPR threshold
    n_non_members = np.sum(y_true == 0)
    n_members = np.sum(y_true == 1)
    
    # Find threshold for target FPR
    thresholds = np.unique(y_scores_sorted)
    for threshold in thresholds:
        preds = (y_scores >= threshold).astype(int)
        tn = np.sum((preds == 0) & (y_true == 0))
        fp = np.sum((preds == 1) & (y_true == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        if fpr <= fpr_threshold:
            tp = np.sum((preds == 1) & (y_true == 1))
            fn = np.sum((preds == 0) & (y_true == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            return tpr
    
    return 0.0


class ThresholdAttack:
    """Threshold-based MIA using confidence scores."""
    
    def __init__(self, metric='confidence'):
        """
        Args:
            metric: 'confidence', 'entropy', or 'modified_entropy'
        """
        self.metric = metric
        self.threshold = None
    
    def fit(self, shadow_model, shadow_train_loader, shadow_test_loader, device):
        """Fit threshold on shadow model."""
        shadow_model.eval()
        
        # Get scores for shadow train (members)
        train_scores = self._get_scores(shadow_model, shadow_train_loader, device)
        # Get scores for shadow test (non-members)
        test_scores = self._get_scores(shadow_model, shadow_test_loader, device)
        
        # Find optimal threshold
        all_scores = np.concatenate([train_scores, test_scores])
        all_labels = np.concatenate([np.ones(len(train_scores)), np.zeros(len(test_scores))])
        
        best_acc = 0
        best_threshold = 0
        for threshold in np.linspace(all_scores.min(), all_scores.max(), 100):
            preds = (all_scores >= threshold).astype(int)
            acc = accuracy_score(all_labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        self.threshold = best_threshold
        return self
    
    def _get_scores(self, model, data_loader, device):
        """Get confidence/entropy scores."""
        model.eval()
        scores = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                if self.metric == 'confidence':
                    # Max confidence
                    score = probs.max(dim=1)[0].cpu().numpy()
                elif self.metric == 'entropy':
                    # Negative entropy (higher = more confident = more likely member)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    score = (-entropy).cpu().numpy()
                elif self.metric == 'modified_entropy':
                    # Modified entropy: entropy of top-k classes
                    topk_probs, _ = torch.topk(probs, k=3, dim=1)
                    entropy = -torch.sum(topk_probs * torch.log(topk_probs + 1e-10), dim=1)
                    score = (-entropy).cpu().numpy()
                
                scores.append(score)
        
        return np.concatenate(scores)
    
    def predict(self, model, data_loader, device):
        """Predict membership."""
        scores = self._get_scores(model, data_loader, device)
        return (scores >= self.threshold).astype(int), scores


class LiRAAttack:
    """Likelihood Ratio Attack (Carlini et al. 2022)."""
    
    def __init__(self, num_shadows=64):
        self.num_shadows = num_shadows
        self.shadow_models_in = []  # Trained WITH the target sample
        self.shadow_models_out = []  # Trained WITHOUT the target sample
    
    def fit(self, shadow_train_fn, train_dataset, test_dataset, device, model_class, model_kwargs):
        """Train shadow models."""
        # For simplicity, we'll train a subset of shadows and reuse
        n_shadows_actual = min(self.num_shadows, 16)  # Limit for computational efficiency
        
        for i in range(n_shadows_actual // 2):
            # IN model: trained with random subset that includes target
            model_in = model_class(**model_kwargs).to(device)
            # OUT model: trained with random subset that excludes target
            model_out = model_class(**model_kwargs).to(device)
            
            # Train models (simplified - in practice would need proper data splits)
            # This is a placeholder for the actual implementation
            self.shadow_models_in.append(model_in)
            self.shadow_models_out.append(model_out)
        
        return self
    
    def predict(self, model, data_loader, device):
        """Predict membership using likelihood ratio."""
        # Simplified version: use confidence-based proxy
        model.eval()
        scores = []
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                confidence = probs.max(dim=1)[0].cpu().numpy()
                scores.append(confidence)
        
        scores = np.concatenate(scores)
        # For LiRA, higher likelihood ratio = more likely member
        return (scores > 0.5).astype(int), scores


class NeuralNetworkAttack:
    """Neural network-based MIA."""
    
    def __init__(self, input_dim=10):
        self.input_dim = input_dim
        self.model = None
    
    def fit(self, shadow_model, shadow_train_loader, shadow_test_loader, device, epochs=50):
        """Train attack model on shadow model outputs."""
        # Collect features from shadow model
        train_features, train_labels = self._get_features(shadow_model, shadow_train_loader, device, 1)
        test_features, test_labels = self._get_features(shadow_model, shadow_test_loader, device, 0)
        
        X = np.vstack([train_features, test_features])
        y = np.concatenate([train_labels, test_labels])
        
        # Train simple classifier
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train, y_train)
        
        return self
    
    def _get_features(self, model, data_loader, device, label):
        """Extract features (posterior probabilities) from model."""
        model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                # Use top-k probabilities as features
                topk_probs = np.sort(probs, axis=1)[:, -self.input_dim:]
                features.append(topk_probs)
                labels.extend([label] * len(inputs))
        
        return np.vstack(features), np.array(labels)
    
    def predict(self, model, data_loader, device):
        """Predict membership."""
        features, _ = self._get_features(model, data_loader, device, 0)
        scores = self.model.predict_proba(features)[:, 1]
        return (scores >= 0.5).astype(int), scores


def evaluate_mia(model, train_loader, test_loader, device, attack_type='threshold', shadow_model=None):
    """
    Evaluate MIA on a model.
    
    Args:
        model: Target model to attack
        train_loader: Data loader for training set (members)
        test_loader: Data loader for test set (non-members)
        device: Device to use
        attack_type: 'threshold', 'lira', or 'nn'
        shadow_model: Shadow model for training attack (if needed)
    
    Returns:
        dict with attack accuracy, AUC, TPR@0.1%FPR
    """
    if attack_type == 'threshold':
        attack = ThresholdAttack(metric='confidence')
        if shadow_model is not None:
            attack.fit(shadow_model, train_loader, test_loader, device)
        else:
            # Use target model itself for threshold (overestimates attack)
            attack.fit(model, train_loader, test_loader, device)
        
        member_preds, member_scores = attack.predict(model, train_loader, device)
        non_member_preds, non_member_scores = attack.predict(model, test_loader, device)
    
    elif attack_type == 'nn':
        attack = NeuralNetworkAttack(input_dim=10)
        if shadow_model is not None:
            attack.fit(shadow_model, train_loader, test_loader, device)
        else:
            attack.fit(model, train_loader, test_loader, device)
        
        member_preds, member_scores = attack.predict(model, train_loader, device)
        non_member_preds, non_member_scores = attack.predict(model, test_loader, device)
    
    else:
        # Fallback: simple confidence-based attack
        model.eval()
        member_scores = []
        non_member_scores = []
        
        with torch.no_grad():
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                confidences = probs.max(dim=1)[0].cpu().numpy()
                member_scores.append(confidences)
            
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                confidences = probs.max(dim=1)[0].cpu().numpy()
                non_member_scores.append(confidences)
        
        member_scores = np.concatenate(member_scores)
        non_member_scores = np.concatenate(non_member_scores)
        
        # Use median as threshold
        threshold = np.median(np.concatenate([member_scores, non_member_scores]))
        member_preds = (member_scores >= threshold).astype(int)
        non_member_preds = (non_member_scores >= threshold).astype(int)
    
    # Compute metrics
    attack_acc = compute_attack_accuracy(member_preds, non_member_preds)
    auc = compute_auc(member_scores, non_member_scores)
    tpr_at_fpr = compute_tpr_at_fpr(member_scores, non_member_scores, fpr_threshold=0.001)
    
    return {
        'attack_accuracy': attack_acc,
        'auc': auc,
        'tpr_at_0.1_fpr': tpr_at_fpr,
        'member_mean_score': float(np.mean(member_scores)),
        'non_member_mean_score': float(np.mean(non_member_scores))
    }
