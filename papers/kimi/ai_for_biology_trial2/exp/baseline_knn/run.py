"""
Baseline 1: Simple k-NN Classifier for Cell Type Annotation
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/exp')

import numpy as np
import scanpy as sc
import json
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from shared.data_loader import load_processed_data
from shared.metrics import compute_classification_metrics
from shared.utils import set_seed, save_results


def evaluate_knn(adata, splits, k=15, seed=42):
    """
    Evaluate k-NN classifier.
    """
    set_seed(seed)
    
    # Prepare data
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    y = adata.obs['cell_type'].values
    
    # Split data
    X_train = X[splits['train']]
    y_train = y[splits['train']]
    X_test = X[splits['test']]
    y_test = y[splits['test']]
    
    # Train k-NN
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(X_train, y_train)
    
    # Predict
    y_pred = knn.predict(X_test)
    
    # Compute metrics
    metrics = compute_classification_metrics(y_test, y_pred)
    
    return metrics, y_test, y_pred


def evaluate_zero_shot(adata, splits, held_out_types, k=15, seed=42):
    """
    Evaluate zero-shot performance on held-out cell types.
    """
    set_seed(seed)
    
    # Prepare data
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    y = adata.obs['cell_type'].values
    
    # Filter out held-out types from training
    is_held_out = np.isin(y, held_out_types)
    train_mask = np.isin(np.arange(len(y)), splits['train']) & (~is_held_out)
    test_mask = np.isin(np.arange(len(y)), splits['test'])
    
    # Split data
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Train k-NN
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(X_train, y_train)
    
    # Predict
    y_pred = knn.predict(X_test)
    
    # Compute metrics on all test cells
    all_metrics = compute_classification_metrics(y_test, y_pred)
    
    # Compute metrics on zero-shot cells only
    zero_shot_mask = np.isin(y_test, held_out_types)
    if zero_shot_mask.sum() > 0:
        zero_shot_metrics = compute_classification_metrics(
            y_test[zero_shot_mask], y_pred[zero_shot_mask]
        )
    else:
        zero_shot_metrics = {'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0}
    
    # Compute metrics on seen cells
    seen_mask = ~zero_shot_mask
    if seen_mask.sum() > 0:
        seen_metrics = compute_classification_metrics(
            y_test[seen_mask], y_pred[seen_mask]
        )
    else:
        seen_metrics = {'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0}
    
    return {
        'all': all_metrics,
        'zero_shot': zero_shot_metrics,
        'seen': seen_metrics,
        'generalization_gap': seen_metrics['accuracy'] - zero_shot_metrics['accuracy']
    }


def main():
    print("=" * 60)
    print("Baseline 1: k-NN Classifier")
    print("=" * 60)
    
    data_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/data')
    output_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    adata = sc.read_h5ad(data_dir / 'pbmc3k_processed.h5ad')
    
    # Load zero-shot splits
    with open(data_dir / 'zero_shot_splits.json', 'r') as f:
        zero_shot_info = json.load(f)
    
    seeds = [42, 123, 456]
    k_values = [5, 15, 30]
    
    all_results = []
    
    for seed in seeds:
        print(f"\n[Seed {seed}]")
        
        # Load splits
        with open(data_dir / f'pbmc3k_splits_seed{seed}.json', 'r') as f:
            splits = json.load(f)
        
        held_out_types = zero_shot_info[f'seed_{seed}']['held_out_types']
        print(f"  Held-out types: {held_out_types}")
        
        # Test different k values
        best_k = 15
        best_acc = 0
        
        for k in k_values:
            metrics, _, _ = evaluate_knn(adata, splits, k=k, seed=seed)
            print(f"  k={k}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")
            
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                best_k = k
        
        print(f"  Best k: {best_k}")
        
        # Evaluate with best k
        test_metrics, _, _ = evaluate_knn(adata, splits, k=best_k, seed=seed)
        zero_shot_results = evaluate_zero_shot(adata, splits, held_out_types, k=best_k, seed=seed)
        
        print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Zero-shot accuracy: {zero_shot_results['zero_shot']['accuracy']:.4f}")
        print(f"  Generalization gap: {zero_shot_results['generalization_gap']:.4f}")
        
        all_results.append({
            'seed': seed,
            'k': best_k,
            'test_accuracy': test_metrics['accuracy'],
            'test_macro_f1': test_metrics['macro_f1'],
            'zero_shot_accuracy': zero_shot_results['zero_shot']['accuracy'],
            'zero_shot_macro_f1': zero_shot_results['zero_shot']['macro_f1'],
            'seen_accuracy': zero_shot_results['seen']['accuracy'],
            'generalization_gap': zero_shot_results['generalization_gap']
        })
    
    # Compute mean and std
    mean_results = {
        'test_accuracy': np.mean([r['test_accuracy'] for r in all_results]),
        'test_accuracy_std': np.std([r['test_accuracy'] for r in all_results]),
        'test_macro_f1': np.mean([r['test_macro_f1'] for r in all_results]),
        'test_macro_f1_std': np.std([r['test_macro_f1'] for r in all_results]),
        'zero_shot_accuracy': np.mean([r['zero_shot_accuracy'] for r in all_results]),
        'zero_shot_accuracy_std': np.std([r['zero_shot_accuracy'] for r in all_results]),
        'zero_shot_macro_f1': np.mean([r['zero_shot_macro_f1'] for r in all_results]),
        'zero_shot_macro_f1_std': np.std([r['zero_shot_macro_f1'] for r in all_results]),
        'generalization_gap': np.mean([r['generalization_gap'] for r in all_results]),
        'generalization_gap_std': np.std([r['generalization_gap'] for r in all_results]),
    }
    
    final_results = {
        'method': 'k-NN',
        'description': 'Simple k-nearest neighbors baseline',
        'seeds': seeds,
        'per_seed_results': all_results,
        'mean': mean_results,
        'config': {'k_values_tested': k_values}
    }
    
    save_results(final_results, output_dir / 'baseline_knn.json')
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Test Accuracy: {mean_results['test_accuracy']:.4f} ± {mean_results['test_accuracy_std']:.4f}")
    print(f"Zero-shot Accuracy: {mean_results['zero_shot_accuracy']:.4f} ± {mean_results['zero_shot_accuracy_std']:.4f}")
    print(f"Generalization Gap: {mean_results['generalization_gap']:.4f} ± {mean_results['generalization_gap_std']:.4f}")


if __name__ == '__main__':
    main()
