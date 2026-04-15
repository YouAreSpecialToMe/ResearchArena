"""
Utility functions for evaluation and metrics.
"""
import torch
import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[float]:
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def evaluate_model(model, dataloader, device='cuda', adapt_method=None) -> Dict:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to run on
        adapt_method: Optional adaptation method (TENT, VPA, SPT-TTA, etc.)
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    correct = 0
    total = 0
    total_loss = 0.0
    all_entropies = []
    
    start_time = time.time()
    
    with torch.set_grad_enabled(adapt_method is not None):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            if adapt_method is not None:
                # Adapt and predict
                if hasattr(adapt_method, 'adapt_and_predict'):
                    outputs = adapt_method.adapt_and_predict(images)
                elif hasattr(adapt_method, 'adapt_sequential'):
                    outputs, _ = adapt_method.adapt_sequential(images, device)
                else:
                    raise ValueError("Unknown adaptation method")
            else:
                # Standard inference
                with torch.no_grad():
                    outputs = model(images)
            
            # Compute accuracy
            acc = accuracy(outputs, labels, topk=(1,))[0]
            correct += acc * labels.size(0) / 100.0
            total += labels.size(0)
            
            # Compute entropy
            probs = torch.softmax(outputs, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            all_entropies.extend(entropy.cpu().tolist())
    
    end_time = time.time()
    
    metrics = {
        'accuracy': 100.0 * correct / total,
        'total_samples': total,
        'avg_entropy': np.mean(all_entropies),
        'std_entropy': np.std(all_entropies),
        'inference_time': end_time - start_time,
        'time_per_image': (end_time - start_time) / total
    }
    
    return metrics


def evaluate_on_corruptions(model, data_dir: str, corruptions: List[str], 
                            severity: int = 3, batch_size: int = 1,
                            device='cuda', adapt_method=None) -> Dict:
    """
    Evaluate model on multiple corruption types.
    
    Args:
        model: Model to evaluate
        data_dir: Root directory for ImageNet-C
        corruptions: List of corruption types
        severity: Corruption severity (1-5)
        batch_size: Batch size for evaluation
        device: Device to run on
        adapt_method: Optional adaptation method
    
    Returns:
        Dictionary with per-corruption and average metrics
    """
    from data_loader import get_imagenet_c_loader
    
    results = {}
    all_accuracies = []
    
    for corruption in corruptions:
        print(f"Evaluating on {corruption} (severity {severity})...")
        
        try:
            loader = get_imagenet_c_loader(
                data_dir, corruption, severity, 
                batch_size=batch_size, num_workers=4
            )
            
            if len(loader.dataset) == 0:
                print(f"  Warning: No data for {corruption}")
                continue
            
            metrics = evaluate_model(model, loader, device, adapt_method)
            results[corruption] = metrics
            all_accuracies.append(metrics['accuracy'])
            
            print(f"  Accuracy: {metrics['accuracy']:.2f}%")
            print(f"  Time: {metrics['time_per_image']*1000:.2f} ms/image")
        except Exception as e:
            print(f"  Error evaluating {corruption}: {e}")
            results[corruption] = {'error': str(e)}
    
    # Compute average metrics
    if all_accuracies:
        results['average'] = {
            'accuracy': np.mean(all_accuracies),
            'std': np.std(all_accuracies),
            'min': np.min(all_accuracies),
            'max': np.max(all_accuracies)
        }
    
    return results


def save_results(results: Dict, filepath: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(results_dict: Dict[str, Dict], 
                             output_path: str = 'figures/accuracy_comparison.png'):
    """
    Plot accuracy comparison across methods.
    
    Args:
        results_dict: Dictionary mapping method name to results
        output_path: Path to save figure
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    methods = list(results_dict.keys())
    corruptions = [c for c in list(results_dict[methods[0]].keys()) 
                   if c not in ['average', 'total']]
    
    # Extract accuracies
    data = []
    for method in methods:
        for corruption in corruptions:
            if corruption in results_dict[method] and 'accuracy' in results_dict[method][corruption]:
                data.append({
                    'Method': method,
                    'Corruption': corruption,
                    'Accuracy': results_dict[method][corruption]['accuracy']
                })
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x='Corruption', y='Accuracy', hue='Method')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison Across Corruption Types')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {output_path}")


def plot_ablation_results(results: Dict, output_path: str):
    """Plot ablation study results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    variants = list(results.keys())
    accuracies = [results[v]['average']['accuracy'] for v in variants if 'average' in results[v]]
    variants = [v for v in variants if 'average' in results[v]]
    
    ax.bar(variants, accuracies)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Ablation Study Results')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_gradient_conflict(model, x: torch.Tensor) -> float:
    """
    Compute gradient conflict between layers.
    
    Args:
        model: Model with prompts
        x: Input tensor
    
    Returns:
        Average pairwise gradient cosine similarity (negative = conflict)
    """
    model.eval()
    
    # Forward pass
    logits = model(x)
    loss = torch.softmax(logits, dim=-1).mean()  # Dummy loss for gradient computation
    
    # Get gradients for each layer's prompts
    gradients = []
    for prompt in model.prompts:
        grad = torch.autograd.grad(loss, prompt, retain_graph=True)[0]
        gradients.append(grad.flatten())
    
    # Compute pairwise cosine similarity
    n = len(gradients)
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            sim = torch.nn.functional.cosine_similarity(
                gradients[i].unsqueeze(0),
                gradients[j].unsqueeze(0)
            ).item()
            similarities.append(sim)
    
    return np.mean(similarities)


def aggregate_seeds_results(results_list: List[Dict]) -> Dict:
    """
    Aggregate results from multiple random seeds.
    
    Args:
        results_list: List of results dictionaries from different seeds
    
    Returns:
        Aggregated results with mean and std
    """
    if not results_list:
        return {}
    
    aggregated = {}
    
    # Get all keys (corruptions, etc.)
    keys = results_list[0].keys()
    
    for key in keys:
        if key == 'config':
            aggregated[key] = results_list[0][key]
            continue
        
        # Collect metrics across seeds
        metrics_keys = results_list[0][key].keys()
        aggregated[key] = {}
        
        for metric in metrics_keys:
            values = [r[key][metric] for r in results_list if key in r and metric in r[key]]
            if values and isinstance(values[0], (int, float)):
                aggregated[key][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
    
    return aggregated


def print_results_table(results: Dict, title: str = "Results"):
    """Print results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    if 'average' in results:
        avg = results['average']
        print(f"Average Accuracy: {avg.get('accuracy', 0):.2f}%")
        print(f"Std Dev: {avg.get('std', 0):.2f}%")
        print(f"Min: {avg.get('min', 0):.2f}%")
        print(f"Max: {avg.get('max', 0):.2f}%")
    
    print(f"\nPer-Corruption Accuracy:")
    print(f"{'Corruption':<25} {'Accuracy':>10} {'Time (ms)':>12}")
    print("-" * 60)
    
    for key, val in results.items():
        if key in ['average', 'config', 'total']:
            continue
        if isinstance(val, dict) and 'accuracy' in val:
            acc = val['accuracy']
            time_ms = val.get('time_per_image', 0) * 1000
            print(f"{key:<25} {acc:>10.2f} {time_ms:>12.2f}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    print("Seed set to 42")
    
    # Test accuracy function
    output = torch.randn(10, 1000)
    target = torch.randint(0, 1000, (10,))
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    print(f"Random accuracy: Top-1={acc1:.2f}%, Top-5={acc5:.2f}%")
