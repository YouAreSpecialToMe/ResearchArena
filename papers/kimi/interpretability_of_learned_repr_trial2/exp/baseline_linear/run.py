"""
Linear Probing Baseline Experiment.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json

from shared.utils import set_seed, save_results
from shared.metrics import linear_probe_accuracy


def run_linear_probe(
    features_path: str,
    seed: int = 42,
    output_dir: str = None
):
    """Run linear probing on extracted features."""
    set_seed(seed)
    
    # Load features
    data = np.load(features_path)
    features = data['features']
    labels = data['labels']
    
    print(f"  Features: {features.shape}, Labels: {labels.shape}")
    
    # Run with different C values
    best_acc = 0
    best_C = None
    
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=seed, stratify=labels
        )
        
        clf = LogisticRegression(C=C, max_iter=1000, multi_class='multinomial')
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        
        if acc > best_acc:
            best_acc = acc
            best_C = C
    
    # Final evaluation with best C
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    
    clf = LogisticRegression(C=best_C, max_iter=1000, multi_class='multinomial')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    from sklearn.metrics import accuracy_score, f1_score
    
    results = {
        'experiment': 'linear_probe',
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        'best_C': float(best_C),
        'config': {'seed': seed, 'feature_dim': features.shape[1], 'n_classes': len(np.unique(labels))}
    }
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_results(results, os.path.join(output_dir, f'results_seed{seed}.json'))
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='exp/baseline_linear/results')
    args = parser.parse_args()
    
    results = run_linear_probe(args.features, args.seed, args.output_dir)
    print(f"  Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
