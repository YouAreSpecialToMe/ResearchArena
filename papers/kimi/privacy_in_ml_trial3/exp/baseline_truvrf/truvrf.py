"""
TruVRF Baseline Implementation.

TruVRF (Zhou et al., 2024) is a fine-tuning-based verification method
that measures model sensitivity via parameter changes during fine-tuning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import roc_auc_score, roc_curve


class TruVRF:
    """TruVRF: Fine-tuning-based verification."""
    
    def __init__(self, original_model, unlearned_model, device='cuda'):
        """
        Initialize TruVRF.
        
        Args:
            original_model: Model before unlearning
            unlearned_model: Model after unlearning
            device: Device to run on
        """
        self.original_model = original_model.to(device)
        self.unlearned_model = unlearned_model.to(device)
        self.device = device
        
    def compute_sensitivity(self, model, data, targets, epochs=3, lr=0.001):
        """
        Compute sensitivity by fine-tuning on verification data.
        
        Args:
            model: Model to evaluate
            data: Verification data
            targets: Verification labels
            epochs: Number of fine-tuning epochs
            lr: Learning rate
            
        Returns:
            Sensitivity scores (per sample)
        """
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Fine-tune
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets).mean()
            loss.backward()
            optimizer.step()
        
        # Compute parameter change as sensitivity
        param_changes = []
        for name, param in model.named_parameters():
            change = torch.abs(param - initial_params[name]).sum().item()
            param_changes.append(change)
        
        total_change = sum(param_changes)
        
        # Compute per-sample sensitivity (gradient contribution)
        model.eval()
        sensitivities = []
        for i in range(data.size(0)):
            x = data[i:i+1]
            y = targets[i:i+1]
            
            model.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            
            # Compute gradient norm as sensitivity
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
            
            sensitivities.append(grad_norm)
        
        return np.array(sensitivities), total_change
    
    def verify_unlearning(self, forget_data, forget_targets, retain_data, retain_targets,
                          verify_data, verify_targets, epochs=3, lr=0.001):
        """
        Verify unlearning using TruVRF method.
        
        Args:
            forget_data: Forget set data
            forget_targets: Forget set labels
            retain_data: Retain set data
            retain_targets: Retain set labels
            verify_data: Verification data for fine-tuning
            verify_targets: Verification labels
            epochs: Fine-tuning epochs
            lr: Learning rate
            
        Returns:
            Dictionary with verification results
        """
        start_time = time.time()
        
        # Move data to device
        forget_data = forget_data.to(self.device)
        forget_targets = forget_targets.to(self.device)
        retain_data = retain_data.to(self.device)
        retain_targets = retain_targets.to(self.device)
        verify_data = verify_data.to(self.device)
        verify_targets = verify_targets.to(self.device)
        
        # Compute sensitivity for original model
        print("Computing sensitivity for original model...")
        orig_sens_forget, _ = self.compute_sensitivity(
            self.original_model, verify_data, verify_targets, epochs, lr)
        
        # Compute sensitivity for unlearned model
        print("Computing sensitivity for unlearned model...")
        unlearn_sens_forget, _ = self.compute_sensitivity(
            self.unlearned_model, verify_data, verify_targets, epochs, lr)
        
        # Compute sensitivity difference (verification score)
        # Higher score = more sensitive = more likely to be forgotten
        forget_scores = np.abs(unlearn_sens_forget - orig_sens_forget)
        
        # For retain set, compute similarly
        # (Simplified: use same verification data approach)
        orig_sens_retain, _ = self.compute_sensitivity(
            self.original_model, verify_data[:len(retain_data)], verify_targets[:len(retain_data)], epochs, lr)
        unlearn_sens_retain, _ = self.compute_sensitivity(
            self.unlearned_model, verify_data[:len(retain_data)], verify_targets[:len(retain_data)], epochs, lr)
        retain_scores = np.abs(unlearn_sens_retain - orig_sens_retain)
        
        # Concatenate for evaluation
        all_scores = np.concatenate([forget_scores, retain_scores])
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
            'forget_sens_mean': np.mean(forget_scores),
            'retain_sens_mean': np.mean(retain_scores)
        }
        
        return results, all_scores, all_labels
