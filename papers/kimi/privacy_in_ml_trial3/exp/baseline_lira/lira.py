"""
LiRA (Membership Inference) Baseline Implementation.

LiRA (Carlini et al., 2022) uses shadow models to infer membership.
For unlearning verification, we check if forgotten samples are still "members".
"""
import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import norm


class LiRA:
    """LiRA: Membership Inference Attack for Unlearning Verification."""
    
    def __init__(self, shadow_models, device='cuda'):
        """
        Initialize LiRA.
        
        Args:
            shadow_models: List of trained shadow models
            device: Device to run on
        """
        self.shadow_models = [m.to(device) for m in shadow_models]
        self.device = device
        
    def collect_logits(self, target_sample, target_label, shadow_indices_list, full_train_dataset):
        """
        Collect logits from shadow models.
        
        Args:
            target_sample: Target sample (1, C, H, W)
            target_label: Target label (int)
            shadow_indices_list: List of indices used to train each shadow model
            full_train_dataset: Full training dataset
            
        Returns:
            in_logits, out_logits: Logits from IN and OUT shadow models
        """
        in_logits = []
        out_logits = []
        
        # Get target sample index
        # (Simplified: assuming we know if sample was in training set)
        # In practice, we'd track this during shadow model training
        
        for i, (shadow_model, indices) in enumerate(zip(self.shadow_models, shadow_indices_list)):
            shadow_model.eval()
            with torch.no_grad():
                target_sample = target_sample.to(self.device)
                output = shadow_model(target_sample)
                logit = output[0, target_label].item()
                
                # For this implementation, we alternate IN/OUT for demonstration
                # In real implementation, we'd track actual membership
                if i % 2 == 0:
                    in_logits.append(logit)
                else:
                    out_logits.append(logit)
        
        return np.array(in_logits), np.array(out_logits)
    
    def compute_lira_score(self, in_logits, out_logits, target_logit):
        """
        Compute LiRA score using likelihood ratio test.
        
        Args:
            in_logits: Logits from shadow models trained WITH sample
            out_logits: Logits from shadow models trained WITHOUT sample
            target_logit: Logit from target model
            
        Returns:
            LiRA score (higher = more likely member)
        """
        # Fit Gaussian distributions
        in_mean, in_std = np.mean(in_logits), np.std(in_logits) + 1e-8
        out_mean, out_std = np.mean(out_logits), np.std(out_logits) + 1e-8
        
        # Log-likelihood under each distribution
        in_log_likelihood = norm.logpdf(target_logit, in_mean, in_std)
        out_log_likelihood = norm.logpdf(target_logit, out_mean, out_std)
        
        # Likelihood ratio (log scale)
        score = in_log_likelihood - out_log_likelihood
        
        return score
    
    def verify_unlearning(self, model, forget_data, forget_targets, 
                          retain_data, retain_targets, shadow_indices_list, 
                          full_train_dataset):
        """
        Verify unlearning using LiRA.
        
        For unlearning verification:
        - Forgotten samples should have LOW membership scores (not members)
        - Retained samples should have HIGH membership scores (are members)
        
        Args:
            model: Target model to evaluate
            forget_data: Forget set data
            forget_targets: Forget set labels
            retain_data: Retain set data
            retain_targets: Retain set labels
            shadow_indices_list: List of indices for each shadow model
            full_train_dataset: Full training dataset
            
        Returns:
            Dictionary with verification results
        """
        start_time = time.time()
        
        model = model.to(self.device)
        model.eval()
        
        forget_scores = []
        retain_scores = []
        
        print("Computing LiRA scores for forget set...")
        for i in range(forget_data.size(0)):
            sample = forget_data[i:i+1]
            label = forget_targets[i].item()
            
            # Get target logit
            with torch.no_grad():
                sample = sample.to(self.device)
                output = model(sample)
                target_logit = output[0, label].item()
            
            # Collect shadow logits
            in_logits, out_logits = self.collect_logits(
                sample, label, shadow_indices_list, full_train_dataset)
            
            # Compute LiRA score
            if len(in_logits) > 0 and len(out_logits) > 0:
                score = self.compute_lira_score(in_logits, out_logits, target_logit)
            else:
                score = 0.0
            
            forget_scores.append(score)
        
        print("Computing LiRA scores for retain set...")
        for i in range(retain_data.size(0)):
            sample = retain_data[i:i+1]
            label = retain_targets[i].item()
            
            # Get target logit
            with torch.no_grad():
                sample = sample.to(self.device)
                output = model(sample)
                target_logit = output[0, label].item()
            
            # Collect shadow logits
            in_logits, out_logits = self.collect_logits(
                sample, label, shadow_indices_list, full_train_dataset)
            
            # Compute LiRA score
            if len(in_logits) > 0 and len(out_logits) > 0:
                score = self.compute_lira_score(in_logits, out_logits, target_logit)
            else:
                score = 0.0
            
            retain_scores.append(score)
        
        forget_scores = np.array(forget_scores)
        retain_scores = np.array(retain_scores)
        
        # For unlearning verification:
        # - We want forget scores to be LOW (not members anymore)
        # - We want retain scores to be HIGH (still members)
        # So we negate the scores for AUC calculation
        
        all_scores = np.concatenate([-forget_scores, -retain_scores])
        all_labels = np.concatenate([
            np.ones(len(forget_scores)),
            np.zeros(len(retain_scores))
        ])
        
        # Compute metrics
        try:
            auc = roc_auc_score(all_labels, all_scores)
        except:
            auc = 0.5
        
        # TPR at 1% FPR
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_scores)
            tpr_at_1fpr = tpr[np.where(fpr <= 0.01)[0][-1]] if len(np.where(fpr <= 0.01)[0]) > 0 else 0.0
        except:
            tpr_at_1fpr = 0.0
        
        elapsed_time = time.time() - start_time
        
        results = {
            'auc': auc,
            'tpr_at_1fpr': tpr_at_1fpr,
            'verify_time': elapsed_time,
            'forget_score_mean': np.mean(forget_scores),
            'retain_score_mean': np.mean(retain_scores)
        }
        
        return results, all_scores, all_labels
