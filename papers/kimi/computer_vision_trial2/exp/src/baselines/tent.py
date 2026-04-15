"""
TENT (Test Entropy Minimization) baseline.
Wang et al., ICLR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import time


class TENT:
    """
    TENT: Fully Test-Time Adaptation by Entropy Minimization.
    Adapts all normalization layer parameters by minimizing entropy.
    """
    
    def __init__(self, model: nn.Module, 
                 lr: float = 1e-3,
                 momentum: float = 0.9,
                 num_steps: int = 1,
                 device: str = 'cuda'):
        """
        Args:
            model: Pretrained model
            lr: Learning rate for adaptation
            momentum: Momentum for SGD optimizer
            num_steps: Number of adaptation steps per batch
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.lr = lr
        self.momentum = momentum
        self.num_steps = num_steps
        
        # Configure model for TTA
        self._configure_model()
        
        # Setup optimizer for normalization layers
        self.optimizer = self._setup_optimizer()
        
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
        
        print(f"TENT: Enabled adaptation for {len(self.norm_layers)} normalization layers")
        
    def _setup_optimizer(self):
        """Setup optimizer for adaptation."""
        params = []
        for m in self.norm_layers:
            params.extend(list(m.parameters()))
        
        if len(params) == 0:
            print("Warning: No normalization parameters found for adaptation")
            return None
            
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
        return optimizer
    
    def adapt_and_evaluate(self, loader, adapt: bool = True) -> Dict[str, float]:
        """
        Adapt on data and evaluate.
        
        Args:
            loader: Data loader
            adapt: Whether to perform adaptation (if False, just evaluate)
        
        Returns:
            Dictionary with metrics
        """
        correct = 0
        total = 0
        total_entropy = 0.0
        total_adaptation_time = 0.0
        total_forward_time = 0.0
        num_batches = 0
        
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
        
        metrics = {
            'accuracy': accuracy,
            'avg_entropy': avg_entropy,
            'total_samples': total,
            'correct_samples': correct,
            'adaptation_time_seconds': total_adaptation_time,
            'forward_time_seconds': total_forward_time,
            'time_per_sample_ms': ((total_adaptation_time + total_forward_time) / total) * 1000
        }
        
        return metrics
    
    def _adapt_batch(self, images: torch.Tensor):
        """
        Adapt model on a batch using entropy minimization.
        """
        self.model.train()
        
        for _ in range(self.num_steps):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Entropy loss
            probs = F.softmax(outputs, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(1).mean()
            
            # Backward and update
            entropy.backward()
            self.optimizer.step()
    
    def reset(self):
        """Reset model to initial state."""
        # This would require saving initial state
        pass


def run_tent_baseline(model: nn.Module, loader, 
                      lr: float = 1e-3,
                      momentum: float = 0.9,
                      device: str = 'cuda') -> Dict[str, float]:
    """
    Convenience function to run TENT baseline.
    
    Args:
        model: Pretrained model
        loader: Data loader
        lr: Learning rate
        momentum: Momentum
        device: Device to run on
    
    Returns:
        Dictionary with metrics
    """
    tent = TENT(model, lr=float(lr), momentum=float(momentum), device=device)
    metrics = tent.adapt_and_evaluate(loader, adapt=True)
    return metrics


if __name__ == '__main__':
    # Test with synthetic data
    import sys
    sys.path.append('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/computer_vision/idea_01')
    
    from src.model_configs import load_model
    from src.data_loader import SyntheticCorruptedDataset
    from torch.utils.data import DataLoader
    
    print("Testing TENT baseline...")
    
    # Create synthetic dataset
    dataset = SyntheticCorruptedDataset(num_samples=200, corruption_type='noise', severity=3)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = load_model('deit_small_patch16_224', num_classes=10, pretrained=False)
    
    # Run TENT
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = run_tent_baseline(model, loader, device=device)
    
    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Avg Entropy: {metrics['avg_entropy']:.4f}")
    print(f"  Time per sample: {metrics['time_per_sample_ms']:.2f} ms")
    
    print("\nTENT baseline test passed!")
