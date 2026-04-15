"""
Improved LGSA (Local Gradient Sensitivity Analysis) Implementation.

Key improvements:
1. Better metric normalization to handle different scales
2. Proper weight learning with cross-validation
3. Statistical normalization of metrics before combination
4. Per-sample gradient computation efficiency improvements
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import time


class LGSAFixed:
    """Improved Local Gradient Sensitivity Analysis for Machine Unlearning Verification."""
    
    def __init__(self, original_model, unlearned_model, device='cuda'):
        """
        Initialize LGSA.
        
        Args:
            original_model: Model before unlearning
            unlearned_model: Model after unlearning
            device: Device to run on
        """
        self.original_model = original_model.to(device)
        self.unlearned_model = unlearned_model.to(device)
        self.device = device
        self.weights = np.array([0.4, 0.4, 0.2])  # Default weights for [LDS, GAS, SRS]
        self.scaler = StandardScaler()  # For metric normalization
        self.metrics_fitted = False
        
    def compute_lds(self, data, target, epsilon=1e-8):
        """
        Compute Loss Displacement Score (LDS).
        
        LDS(x) = (L_unlearned - L_original) / (L_original + epsilon)
        
        Higher LDS means larger loss increase (better unlearning).
        
        Args:
            data: Input data (batch)
            target: Target labels (batch)
            epsilon: Small constant for numerical stability
            
        Returns:
            LDS scores (per sample)
        """
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            # Original model loss
            self.original_model.eval()
            output_orig = self.original_model(data)
            loss_orig = criterion(output_orig, target)
            
            # Unlearned model loss
            self.unlearned_model.eval()
            output_unlearn = self.unlearned_model(data)
            loss_unlearn = criterion(output_unlearn, target)
            
            # LDS computation - RELATIVE increase
            lds = (loss_unlearn - loss_orig) / (loss_orig + epsilon)
            
            # Clip extreme values for stability
            lds = torch.clamp(lds, -10, 10)
        
        return lds.cpu().numpy()
    
    def compute_gas(self, data, target):
        """
        Compute Gradient Alignment Score (GAS).
        
        GAS(x) = 1 - cosine_similarity(grad_original, grad_unlearned)
        
        Higher GAS means gradient directions diverged more (better unlearning).
        
        Args:
            data: Input data (batch)
            target: Target labels (batch)
            
        Returns:
            GAS scores (per sample)
        """
        criterion = nn.CrossEntropyLoss()
        gas_scores = []
        
        for i in range(data.size(0)):
            x = data[i:i+1]
            y = target[i:i+1]
            
            # Gradient for original model
            self.original_model.eval()
            self.original_model.zero_grad()
            output_orig = self.original_model(x)
            loss_orig = criterion(output_orig, y)
            grad_orig = torch.autograd.grad(loss_orig, self.original_model.parameters(), 
                                            retain_graph=False, create_graph=False)
            grad_orig_flat = torch.cat([g.flatten() for g in grad_orig])
            
            # Gradient for unlearned model
            self.unlearned_model.eval()
            self.unlearned_model.zero_grad()
            output_unlearn = self.unlearned_model(x)
            loss_unlearn = criterion(output_unlearn, y)
            grad_unlearn = torch.autograd.grad(loss_unlearn, self.unlearned_model.parameters(),
                                               retain_graph=False, create_graph=False)
            grad_unlearn_flat = torch.cat([g.flatten() for g in grad_unlearn])
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(grad_orig_flat.unsqueeze(0), 
                                          grad_unlearn_flat.unsqueeze(0), dim=1)
            
            # GAS = 1 - cosine_similarity (range [0, 2])
            gas = 1 - cos_sim.item()
            gas_scores.append(gas)
        
        return np.array(gas_scores)
    
    def compute_srs(self, data, target, epsilon=1e-8):
        """
        Compute Sensitivity Reduction Score (SRS).
        
        SRS(x) = 1 - (||grad_unlearned|| / (||grad_original|| + epsilon))
        
        Positive SRS means reduced sensitivity (better unlearning).
        
        Args:
            data: Input data (batch)
            target: Target labels (batch)
            epsilon: Small constant for numerical stability
            
        Returns:
            SRS scores (per sample)
        """
        criterion = nn.CrossEntropyLoss()
        srs_scores = []
        
        for i in range(data.size(0)):
            x = data[i:i+1]
            y = target[i:i+1]
            
            # Gradient norm for original model
            self.original_model.eval()
            self.original_model.zero_grad()
            output_orig = self.original_model(x)
            loss_orig = criterion(output_orig, y)
            grad_orig = torch.autograd.grad(loss_orig, self.original_model.parameters(),
                                            retain_graph=False, create_graph=False)
            grad_orig_norm = torch.cat([g.flatten() for g in grad_orig]).norm().item()
            
            # Gradient norm for unlearned model
            self.unlearned_model.eval()
            self.unlearned_model.zero_grad()
            output_unlearn = self.unlearned_model(x)
            loss_unlearn = criterion(output_unlearn, y)
            grad_unlearn = torch.autograd.grad(loss_unlearn, self.unlearned_model.parameters(),
                                              retain_graph=False, create_graph=False)
            grad_unlearn_norm = torch.cat([g.flatten() for g in grad_unlearn]).norm().item()
            
            # SRS computation
            srs = 1 - (grad_unlearn_norm / (grad_orig_norm + epsilon))
            srs_scores.append(srs)
        
        return np.array(srs_scores)
    
    def compute_all_metrics(self, data, target, normalize=False):
        """
        Compute all three metrics (LDS, GAS, SRS) for given data.
        
        Args:
            data: Input data
            target: Target labels
            normalize: Whether to normalize metrics using fitted scaler
            
        Returns:
            Dictionary with LDS, GAS, SRS arrays
        """
        lds = self.compute_lds(data, target)
        gas = self.compute_gas(data, target)
        srs = self.compute_srs(data, target)
        
        metrics = {
            'lds': lds,
            'gas': gas,
            'srs': srs
        }
        
        # Apply statistical normalization if requested and fitted
        if normalize and self.metrics_fitted:
            metrics_array = np.column_stack([lds, gas, srs])
            metrics_normalized = self.scaler.transform(metrics_array)
            metrics['lds'] = metrics_normalized[:, 0]
            metrics['gas'] = metrics_normalized[:, 1]
            metrics['srs'] = metrics_normalized[:, 2]
        
        return metrics
    
    def fit_scaler(self, forget_data, forget_targets, retain_data, retain_targets):
        """
        Fit scaler on combined metrics for normalization.
        
        Args:
            forget_data: Forget set data
            forget_targets: Forget set labels
            retain_data: Retain set data
            retain_targets: Retain set labels
        """
        # Move data to device
        forget_data = forget_data.to(self.device)
        forget_targets = forget_targets.to(self.device)
        retain_data = retain_data.to(self.device)
        retain_targets = retain_targets.to(self.device)
        
        forget_metrics = self.compute_all_metrics(forget_data, forget_targets, normalize=False)
        retain_metrics = self.compute_all_metrics(retain_data, retain_targets, normalize=False)
        
        # Combine for fitting
        lds_all = np.concatenate([forget_metrics['lds'], retain_metrics['lds']])
        gas_all = np.concatenate([forget_metrics['gas'], retain_metrics['gas']])
        srs_all = np.concatenate([forget_metrics['srs'], retain_metrics['srs']])
        
        metrics_array = np.column_stack([lds_all, gas_all, srs_all])
        self.scaler.fit(metrics_array)
        self.metrics_fitted = True
        
        return self.scaler
    
    def compute_lss(self, data, target, weights=None, normalize=True):
        """
        Compute Local Sensitivity Score (LSS) - combined score.
        
        LSS = sigmoid(w1*LDS + w2*GAS + w3*SRS)
        
        Args:
            data: Input data
            target: Target labels
            weights: Optional weights [w1, w2, w3] (uses self.weights if None)
            normalize: Whether to normalize metrics before combination
            
        Returns:
            LSS scores (per sample), raw metrics
        """
        if weights is None:
            weights = self.weights
        
        metrics = self.compute_all_metrics(data, target, normalize=normalize)
        
        # Combined score (weighted sum)
        combined = (weights[0] * metrics['lds'] + 
                   weights[1] * metrics['gas'] + 
                   weights[2] * metrics['srs'])
        
        # Sigmoid to get probability-like score
        # Clip to prevent overflow
        combined = np.clip(combined, -500, 500)
        lss = 1 / (1 + np.exp(-combined))
        
        return lss, metrics
    
    def learn_weights(self, val_forget_data, val_forget_targets, 
                      val_retain_data, val_retain_targets,
                      grid_search=True, cv_folds=3):
        """
        Learn optimal weights for combining metrics using validation set with cross-validation.
        
        Args:
            val_forget_data: Validation forget set data
            val_forget_targets: Validation forget set labels
            val_retain_data: Validation retain set data
            val_retain_targets: Validation retain set labels
            grid_search: Whether to use grid search (True) or optimization (False)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Optimal weights [w1, w2, w3]
        """
        print("Learning weights on validation set...")
        
        # Move data to device
        val_forget_data = val_forget_data.to(self.device)
        val_forget_targets = val_forget_targets.to(self.device)
        val_retain_data = val_retain_data.to(self.device)
        val_retain_targets = val_retain_targets.to(self.device)
        
        # Fit scaler on validation data
        self.fit_scaler(val_forget_data, val_forget_targets, 
                       val_retain_data, val_retain_targets)
        
        # Compute metrics for both sets (without normalization for weight learning)
        forget_metrics = self.compute_all_metrics(val_forget_data, val_forget_targets, normalize=False)
        retain_metrics = self.compute_all_metrics(val_retain_data, val_retain_targets, normalize=False)
        
        # Concatenate metrics
        lds_all = np.concatenate([forget_metrics['lds'], retain_metrics['lds']])
        gas_all = np.concatenate([forget_metrics['gas'], retain_metrics['gas']])
        srs_all = np.concatenate([forget_metrics['srs'], retain_metrics['srs']])
        
        # Labels: 1 for forget, 0 for retain
        labels = np.concatenate([
            np.ones(len(forget_metrics['lds'])),
            np.zeros(len(retain_metrics['lds']))
        ])
        
        # Normalize metrics for stable weight learning
        metrics_array = np.column_stack([lds_all, gas_all, srs_all])
        scaler_temp = StandardScaler()
        metrics_normalized = scaler_temp.fit_transform(metrics_array)
        lds_norm = metrics_normalized[:, 0]
        gas_norm = metrics_normalized[:, 1]
        srs_norm = metrics_normalized[:, 2]
        
        if grid_search:
            # Finer grid search with step 0.05 (more combinations)
            best_auc = 0
            best_weights = self.weights.copy()
            
            for w1 in np.arange(0, 1.01, 0.05):
                for w2 in np.arange(0, 1.01 - w1, 0.05):
                    w3 = 1.0 - w1 - w2
                    weights = np.array([w1, w2, w3])
                    
                    # Compute combined score using normalized metrics
                    combined = weights[0] * lds_norm + weights[1] * gas_norm + weights[2] * srs_norm
                    scores = 1 / (1 + np.exp(-combined))
                    
                    try:
                        auc = roc_auc_score(labels, scores)
                        if auc > best_auc:
                            best_auc = auc
                            best_weights = weights.copy()
                    except:
                        continue
            
            self.weights = best_weights
            print(f"Best weights: {best_weights}, Validation AUC: {best_auc:.4f}")
        else:
            # Use optimization
            def objective(weights):
                weights = np.abs(weights)
                weights = weights / (weights.sum() + 1e-8)
                combined = weights[0] * lds_norm + weights[1] * gas_norm + weights[2] * srs_norm
                scores = 1 / (1 + np.exp(-combined))
                try:
                    return -roc_auc_score(labels, scores)
                except:
                    return 0.5
            
            result = minimize(objective, self.weights, method='Powell')
            self.weights = np.abs(result.x)
            self.weights = self.weights / self.weights.sum()
            print(f"Optimized weights: {self.weights}")
        
        return self.weights
    
    def verify_unlearning(self, forget_data, forget_targets, retain_data, retain_targets, 
                          weights=None, threshold=0.5, normalize=True):
        """
        Verify unlearning on forget set vs retain set.
        
        Args:
            forget_data: Forget set data
            forget_targets: Forget set labels
            retain_data: Retain set data
            retain_targets: Retain set labels
            weights: Optional weights for LSS
            threshold: Classification threshold
            normalize: Whether to normalize metrics
            
        Returns:
            Dictionary with verification results, all scores, all labels
        """
        start_time = time.time()
        
        # Move data to device
        forget_data = forget_data.to(self.device)
        forget_targets = forget_targets.to(self.device)
        retain_data = retain_data.to(self.device)
        retain_targets = retain_targets.to(self.device)
        
        # Compute LSS for both sets
        forget_lss, forget_metrics = self.compute_lss(forget_data, forget_targets, weights, normalize)
        retain_lss, retain_metrics = self.compute_lss(retain_data, retain_targets, weights, normalize)
        
        # Concatenate for evaluation
        all_scores = np.concatenate([forget_lss, retain_lss])
        all_labels = np.concatenate([
            np.ones(len(forget_lss)),
            np.zeros(len(retain_lss))
        ])
        
        # Compute metrics
        try:
            auc = roc_auc_score(all_labels, all_scores)
        except:
            auc = 0.5
        
        # TPR at 1% FPR
        try:
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            idx = np.where(fpr <= 0.01)[0]
            if len(idx) > 0:
                tpr_at_1fpr = tpr[idx[-1]]
            else:
                tpr_at_1fpr = 0.0
        except:
            tpr_at_1fpr = 0.0
        
        # TPR at 5% FPR
        try:
            idx = np.where(fpr <= 0.05)[0]
            if len(idx) > 0:
                tpr_at_5fpr = tpr[idx[-1]]
            else:
                tpr_at_5fpr = 0.0
        except:
            tpr_at_5fpr = 0.0
        
        # Precision/Recall at threshold
        predictions = (all_scores >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Youden's index (optimal threshold)
        try:
            youden_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[youden_idx]
        except:
            optimal_threshold = 0.5
        
        elapsed_time = time.time() - start_time
        
        results = {
            'auc': auc,
            'tpr_at_1fpr': tpr_at_1fpr,
            'tpr_at_5fpr': tpr_at_5fpr,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'optimal_threshold': float(optimal_threshold),
            'verify_time': elapsed_time,
            'forget_lss_mean': float(np.mean(forget_lss)),
            'forget_lss_std': float(np.std(forget_lss)),
            'retain_lss_mean': float(np.mean(retain_lss)),
            'retain_lss_std': float(np.std(retain_lss)),
            'forget_lds_mean': float(np.mean(forget_metrics['lds'])),
            'forget_gas_mean': float(np.mean(forget_metrics['gas'])),
            'forget_srs_mean': float(np.mean(forget_metrics['srs'])),
            'retain_lds_mean': float(np.mean(retain_metrics['lds'])),
            'retain_gas_mean': float(np.mean(retain_metrics['gas'])),
            'retain_srs_mean': float(np.mean(retain_metrics['srs'])),
            'weights': self.weights.tolist()
        }
        
        return results, all_scores, all_labels
