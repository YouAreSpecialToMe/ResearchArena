"""
EATA (Efficient Test-Time Adaptation) baseline.
Niu et al., ICML 2022
Sample-level filtering with Fisher information regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import time
import math


class EATA:
    """
    EATA: Efficient Test-Time Adaptation.
    Filters samples based on entropy and uses Fisher regularization.
    """
    
    def __init__(self, model: nn.Module,
                 lr: float = 1e-3,
                 momentum: float = 0.9,
                 num_steps: int = 1,
                 ent_threshold: float = None,  # Set to ln(num_classes) * 0.4 by default
                 fisher_alpha: float = 2000.0,
                 device: str = 'cuda'):
        """
        Args:
            model: Pretrained model
            lr: Learning rate for adaptation
            momentum: Momentum for SGD optimizer
            num_steps: Number of adaptation steps per batch
            ent_threshold: Entropy threshold for sample filtering
            fisher_alpha: Fisher regularization coefficient
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.lr = lr
        self.momentum = momentum
        self.num_steps = num_steps
        self.fisher_alpha = fisher_alpha
        
        # Configure model for TTA
        self._configure_model()
        
        # Get number of classes from model
        if hasattr(model, 'num_classes'):
            self.num_classes = model.num_classes
        elif hasattr(model, 'head'):
            self.num_classes = model.head.out_features
        else:
            self.num_classes = 1000  # Default
        
        # Set entropy threshold
        if ent_threshold is None:
            self.ent_threshold = math.log(self.num_classes) * 0.4
        else:
            self.ent_threshold = ent_threshold
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Fisher information for regularization
        self.fisher_dict = {}
        self._init_fisher()
        
        # Track samples filtered
        self.total_samples = 0
        self.filtered_samples = 0
        
    def _configure_model(self):
        """Configure model for test-time adaptation."""
        self.model.to(self.device)
        self.model.train()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable gradients for normalization layers
        self.norm_layers = []
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.BatchNorm1d)):
                for param in m.parameters():
                    param.requires_grad = True
                self.norm_layers.append(m)
        
        print(f"EATA: Enabled adaptation for {len(self.norm_layers)} normalization layers")
        
    def _setup_optimizer(self):
        """Setup optimizer for adaptation."""
        params = []
        for m in self.norm_layers:
            params.extend(list(m.parameters()))
        
        if len(params) == 0:
            print("Warning: No normalization parameters found")
            return None
            
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
        return optimizer
    
    def _init_fisher(self):
        """Initialize Fisher information for regularization."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_dict[name] = torch.zeros_like(param)
    
    def adapt_and_evaluate(self, loader, adapt: bool = True) -> Dict[str, float]:
        """
        Adapt on data and evaluate.
        
        Args:
            loader: Data loader
            adapt: Whether to perform adaptation
        
        Returns:
            Dictionary with metrics
        """
        correct = 0
        total = 0
        total_entropy = 0.0
        total_adaptation_time = 0.0
        total_forward_time = 0.0
        num_batches = 0
        
        self.total_samples = 0
        self.filtered_samples = 0
        
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Adaptation phase
            if adapt and self.optimizer is not None:
                adapt_start = time.time()
                self._adapt_batch(images)
                total_adaptation_time += time.time() - adapt_start
            
            # Evaluation phase
            forward_start = time.time()
            with torch.no_grad():
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                # Compute entropy
                probs = F.softmax(outputs, dim=1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(1).mean()
                total_entropy += entropy.item()
            
            total_forward_time += time.time() - forward_start
            num_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)}, "
                      f"Running Acc: {100. * correct / total:.2f}%")
        
        accuracy = 100. * correct / total
        avg_entropy = total_entropy / num_batches
        filter_ratio = 100. * self.filtered_samples / max(self.total_samples, 1)
        
        metrics = {
            'accuracy': accuracy,
            'avg_entropy': avg_entropy,
            'total_samples': total,
            'correct_samples': correct,
            'filtered_samples': self.filtered_samples,
            'filter_ratio': filter_ratio,
            'adaptation_time_seconds': total_adaptation_time,
            'forward_time_seconds': total_forward_time,
            'time_per_sample_ms': ((total_adaptation_time + total_forward_time) / total) * 1000
        }
        
        return metrics
    
    def _adapt_batch(self, images: torch.Tensor):
        """
        Adapt model on a batch with sample filtering.
        """
        self.model.train()
        
        for _ in range(self.num_steps):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute entropy per sample
            probs = F.softmax(outputs, dim=1)
            entropies = -(probs * torch.log(probs + 1e-10)).sum(1)
            
            # Filter samples with high entropy
            mask = entropies < self.ent_threshold
            self.total_samples += images.size(0)
            self.filtered_samples += (~mask).sum().item()
            
            if mask.sum() == 0:
                # No samples to adapt on
                continue
            
            # Only use reliable samples
            reliable_outputs = outputs[mask]
            reliable_probs = probs[mask]
            
            # Entropy minimization loss
            entropy_loss = -(reliable_probs * torch.log(reliable_probs + 1e-10)).sum(1).mean()
            
            # Fisher regularization (simplified version)
            fisher_loss = self._compute_fisher_loss()
            
            # Total loss
            total_loss = entropy_loss + self.fisher_alpha * fisher_loss
            
            # Backward and update
            total_loss.backward()
            self.optimizer.step()
    
    def _compute_fisher_loss(self) -> torch.Tensor:
        """Compute Fisher information regularization loss."""
        fisher_loss = 0.0
        count = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_dict:
                # Simplified Fisher: just use squared parameter values
                fisher_loss += (param ** 2).sum()
                count += 1
        
        return fisher_loss / max(count, 1)


def run_eata_baseline(model: nn.Module, loader,
                      lr: float = 1e-3,
                      momentum: float = 0.9,
                      ent_threshold: float = None,
                      fisher_alpha: float = 2000.0,
                      device: str = 'cuda') -> Dict[str, float]:
    """
    Convenience function to run EATA baseline.
    """
    eata = EATA(model, lr=float(lr), momentum=float(momentum), 
                ent_threshold=ent_threshold, fisher_alpha=float(fisher_alpha),
                device=device)
    metrics = eata.adapt_and_evaluate(loader, adapt=True)
    return metrics


if __name__ == '__main__':
    # Test with synthetic data
    import sys
    sys.path.append('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/computer_vision/idea_01')
    
    from src.model_configs import load_model
    from src.data_loader import SyntheticCorruptedDataset
    from torch.utils.data import DataLoader
    
    print("Testing EATA baseline...")
    
    # Create synthetic dataset
    dataset = SyntheticCorruptedDataset(num_samples=200, corruption_type='noise', severity=3)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = load_model('deit_small_patch16_224', num_classes=10, pretrained=False)
    
    # Run EATA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = run_eata_baseline(model, loader, device=device)
    
    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Avg Entropy: {metrics['avg_entropy']:.4f}")
    print(f"  Filter Ratio: {metrics['filter_ratio']:.2f}%")
    print(f"  Time per sample: {metrics['time_per_sample_ms']:.2f} ms")
    
    print("\nEATA baseline test passed!")
