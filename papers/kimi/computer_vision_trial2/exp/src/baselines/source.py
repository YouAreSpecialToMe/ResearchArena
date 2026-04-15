"""
Source baseline: No adaptation, evaluate pretrained model directly.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import time


class Source:
    """
    Source baseline - no adaptation.
    Simply evaluates the pretrained model on test data.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: Pretrained model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        """
        Evaluate model on data loader.
        
        Returns:
            Dictionary with metrics (accuracy, loss, etc.)
        """
        correct = 0
        total = 0
        total_loss = 0.0
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Compute loss (for reference)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)}, "
                      f"Running Acc: {100. * correct / total:.2f}%")
        
        elapsed_time = time.time() - start_time
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / total
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'total_samples': total,
            'correct_samples': correct,
            'time_seconds': elapsed_time,
            'time_per_sample_ms': (elapsed_time / total) * 1000
        }
        
        return metrics


def run_source_baseline(model: nn.Module, loader, device: str = 'cuda') -> Dict[str, float]:
    """
    Convenience function to run source baseline.
    
    Args:
        model: Pretrained model
        loader: Data loader
        device: Device to run on
    
    Returns:
        Dictionary with metrics
    """
    source = Source(model, device)
    metrics = source.evaluate(loader)
    return metrics


if __name__ == '__main__':
    # Test with synthetic data
    import sys
    sys.path.append('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/computer_vision/idea_01')
    
    from src.model_configs import load_model
    from src.data_loader import SyntheticCorruptedDataset
    from torch.utils.data import DataLoader
    
    print("Testing Source baseline...")
    
    # Create synthetic dataset
    dataset = SyntheticCorruptedDataset(num_samples=200, corruption_type='noise', severity=3)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = load_model('deit_small_patch16_224', num_classes=10, pretrained=False)
    
    # Run evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = run_source_baseline(model, loader, device)
    
    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Time per sample: {metrics['time_per_sample_ms']:.2f} ms")
    
    print("\nSource baseline test passed!")
