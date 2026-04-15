"""
Main LayoutLearner Experiment: Train and Evaluate XGBoost Model.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
import os
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import set_seed, save_json, get_project_paths
from metrics import compute_metrics, compute_metrics_with_ci


def load_data():
    """Load train and test data."""
    paths = get_project_paths()
    train_df = pd.read_csv(f"{paths['data']}/processed/train.csv")
    test_df = pd.read_csv(f"{paths['data']}/processed/test.csv")
    
    feature_cols = [c for c in train_df.columns if c not in ['benchmark_name', 'label']]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                params: Dict = None) -> xgb.XGBClassifier:
    """Train XGBoost model with given parameters."""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42
        }
    
    model = xgb.XGBClassifier(**params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    return model, train_time


def cross_validate(X: np.ndarray, y: np.ndarray, params: Dict, n_folds: int = 5) -> Dict:
    """Perform cross-validation."""
    model = xgb.XGBClassifier(**params)
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Use AUC as primary metric
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    return {
        'auc_mean': float(np.mean(auc_scores)),
        'auc_std': float(np.std(auc_scores)),
        'accuracy_mean': float(np.mean(acc_scores)),
        'accuracy_std': float(np.std(acc_scores)),
        'fold_aucs': auc_scores.tolist(),
        'fold_accs': acc_scores.tolist()
    }


def hyperparameter_search(X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """Simple grid search for hyperparameters."""
    print("\nPerforming hyperparameter search...")
    
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    
    best_auc = 0
    best_params = None
    
    for max_depth in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            params = {
                'n_estimators': 100,
                'max_depth': max_depth,
                'learning_rate': lr,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'random_state': 42
            }
            
            cv_results = cross_validate(X_train, y_train, params, n_folds=3)
            mean_auc = cv_results['auc_mean']
            
            print(f"  max_depth={max_depth}, lr={lr}: AUC={mean_auc:.3f}")
            
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = params
    
    print(f"\nBest params (AUC={best_auc:.3f}):")
    print(f"  max_depth={best_params['max_depth']}")
    print(f"  learning_rate={best_params['learning_rate']}")
    
    return best_params


def evaluate_with_seed(X_train, X_test, y_train, y_test, feature_cols, seed: int) -> Dict:
    """Evaluate model with specific random seed."""
    set_seed(seed)
    
    # Use fixed best params (from hyperparameter search or default)
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': seed
    }
    
    # Train model
    model, train_time = train_model(X_train, y_train, params)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = {
        feature_cols[i]: float(importance[i])
        for i in range(len(feature_cols))
    }
    
    return {
        'seed': seed,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'accuracy': metrics['accuracy'],
        'roc_auc': metrics.get('roc_auc', 0.0),
        'train_time': train_time,
        'feature_importance': feature_importance
    }


def main():
    print("=" * 60)
    print("Main Experiment: LayoutLearner XGBoost Model")
    print("=" * 60)
    
    paths = get_project_paths()
    
    # Load data
    print("\nLoading data...")
    X_train, X_test, y_train, y_test, feature_cols = load_data()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Features: {len(feature_cols)}")
    
    # Cross-validation with default params
    print("\n5-Fold Cross-Validation...")
    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42
    }
    cv_results = cross_validate(X_train, y_train, default_params, n_folds=5)
    print(f"  CV AUC: {cv_results['auc_mean']:.3f} ± {cv_results['auc_std']:.3f}")
    print(f"  CV Acc: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
    
    # Hyperparameter search
    best_params = hyperparameter_search(X_train, y_train)
    
    # Final training with best params
    print("\nTraining final model with best hyperparameters...")
    model, train_time = train_model(X_train, y_train, best_params)
    print(f"  Training time: {train_time:.2f}s")
    
    # Save model
    model.save_model(f"{paths['models']}/layout_learner_xgboost.json")
    print(f"  Model saved to: {paths['models']}/layout_learner_xgboost.json")
    
    # Evaluate on test set with multiple seeds
    print("\nEvaluating on test set (3 seeds)...")
    seeds = [42, 123, 456]
    results = []
    
    for seed in seeds:
        result = evaluate_with_seed(X_train, X_test, y_train, y_test, feature_cols, seed)
        results.append(result)
        print(f"\nSeed {seed}:")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall:    {result['recall']:.3f}")
        print(f"  F1-Score:  {result['f1_score']:.3f}")
        print(f"  ROC-AUC:   {result['roc_auc']:.3f}")
    
    # Aggregate results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'feature_importance'} 
                               for r in results])
    
    aggregated = {
        'experiment': 'layoutlearner_main',
        'seeds': results,
        'mean': {
            'precision': float(results_df['precision'].mean()),
            'recall': float(results_df['recall'].mean()),
            'f1_score': float(results_df['f1_score'].mean()),
            'accuracy': float(results_df['accuracy'].mean()),
            'roc_auc': float(results_df['roc_auc'].mean())
        },
        'std': {
            'precision': float(results_df['precision'].std()),
            'recall': float(results_df['recall'].std()),
            'f1_score': float(results_df['f1_score'].std()),
            'accuracy': float(results_df['accuracy'].std()),
            'roc_auc': float(results_df['roc_auc'].std())
        },
        'cv_results': cv_results,
        'best_params': best_params,
        'model_size_mb': os.path.getsize(f"{paths['models']}/layout_learner_xgboost.json") / (1024 * 1024)
    }
    
    # Save feature importance from first seed
    aggregated['feature_importance'] = results[0]['feature_importance']
    
    # Save results
    exp_dir = paths['exp']
    save_json(aggregated, f"{exp_dir}/layoutlearner_main/results.json")
    results_df.to_csv(f"{exp_dir}/layoutlearner_main/results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("Aggregated Results (mean ± std):")
    print("=" * 60)
    for metric in ['precision', 'recall', 'f1_score', 'accuracy', 'roc_auc']:
        print(f"  {metric:12}: {aggregated['mean'][metric]:.3f} ± {aggregated['std'][metric]:.3f}")
    
    # Check success criterion: within 20% of profile-guided (F1=1.0)
    achieved_f1 = aggregated['mean']['f1_score']
    threshold_f1 = 0.8  # Within 20% of 1.0
    print(f"\nSuccess Criterion: F1 >= {threshold_f1} (within 20% of oracle)")
    print(f"  Achieved: {achieved_f1:.3f} - {'PASS' if achieved_f1 >= threshold_f1 else 'FAIL'}")
    
    print(f"\nResults saved to: {exp_dir}/layoutlearner_main/")
    
    return aggregated


if __name__ == '__main__':
    main()
