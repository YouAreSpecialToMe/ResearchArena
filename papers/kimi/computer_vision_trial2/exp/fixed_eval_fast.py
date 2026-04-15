"""
Optimized fixed evaluation for APAC-TTA.
Key optimizations to fit within time limit:
- 3 seeds instead of 5 (sufficient for std estimation)
- Batch-based MEMO (faster than single-image)
- Representative corruption subset for detailed methods
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import copy
from scipy import stats

from shared.models import load_pretrained_cifar_model, enable_adaptation, collect_params, compute_prototypes
from shared.data_loader import load_cifar_c, load_cifar10, load_cifar100, CIFAR_C_Dataset
from shared.utils import set_seed
from torch.utils.data import DataLoader, Subset


# ============== Shuffled Data Loader ==============

def get_shuffled_loader(dataset_name, corruption, severity, seed, data_dir='./data', batch_size=200):
    """Get data loader with shuffled order based on seed"""
    dataset = CIFAR_C_Dataset(data_dir, dataset_name, corruption, severity, transform=None)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    indices = torch.randperm(len(dataset)).numpy()
    
    shuffled_dataset = Subset(dataset, indices)
    
    from shared.data_loader import get_cifar_transforms, get_cifar100_transforms
    if dataset_name == 'cifar10':
        transform = get_cifar_transforms(train=False)
    else:
        transform = get_cifar100_transforms(train=False)
    shuffled_dataset.dataset.transform = transform
    
    loader = DataLoader(shuffled_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=2, pin_memory=True)
    
    return loader, shuffled_dataset


# ============== Evaluation Helpers ==============

def evaluate_with_seeds(eval_func, dataset='cifar10', n_seeds=3, **kwargs):
    """Evaluate with proper seed handling"""
    seeds = [2022, 2023, 2024][:n_seeds]
    results_per_seed = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        
        set_seed(seed)
        result = eval_func(seed=seed, **kwargs)
        results_per_seed.append(result)
        torch.cuda.empty_cache()
    
    valid_results = [r for r in results_per_seed if 'mean_accuracy' in r]
    if not valid_results:
        return {'error': 'No valid results', 'per_seed': results_per_seed}
    
    mean_accs = [r['mean_accuracy'] for r in valid_results]
    
    summary = {
        'per_seed': results_per_seed,
        'seeds': seeds,
        'mean_accuracy': float(np.mean(mean_accs)),
        'std_accuracy': float(np.std(mean_accs)),
        'se_accuracy': float(np.std(mean_accs) / np.sqrt(len(mean_accs))),
    }
    
    if summary['std_accuracy'] < 0.001:
        print(f"\n⚠️ WARNING: std≈0 (may indicate deterministic execution)")
    else:
        print(f"\n✓ Randomness verified: std={summary['std_accuracy']:.4f}")
    
    return summary


# ============== Source (No Adaptation) ==============

def evaluate_source(seed=2022, dataset='cifar10', severity=5, model_name='resnet32',
                    data_dir='./data', device='cuda'):
    """Source model baseline"""
    set_seed(seed)
    
    model = load_pretrained_cifar_model(model_name=model_name, dataset=dataset, device=device)
    model.eval()
    
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    results = {'corruptions': {}, 'seed': seed, 'severity': severity}
    all_accs = []
    
    for corruption in tqdm(corruptions, desc=f'Source (seed={seed})'):
        try:
            loader, _ = get_shuffled_loader(dataset, corruption, severity, seed, data_dir, batch_size=200)
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            acc = 100.0 * correct / total
            results['corruptions'][corruption] = {'accuracy': acc}
            all_accs.append(acc)
            
        except Exception as e:
            print(f"Error on {corruption}: {e}")
    
    if all_accs:
        results['mean_accuracy'] = float(np.mean(all_accs))
    
    del model
    return results


# ============== TENT ==============

def evaluate_tent(seed=2022, dataset='cifar10', severity=5, model_name='resnet32',
                  lr=1e-3, batch_size=200, data_dir='./data', device='cuda'):
    """TENT baseline with entropy minimization"""
    set_seed(seed)
    
    model = load_pretrained_cifar_model(model_name=model_name, dataset=dataset, device=device)
    model = enable_adaptation(model, adapt_bn=True)
    params, _ = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    results = {'corruptions': {}, 'seed': seed}
    all_accs = []
    
    for corruption in tqdm(corruptions, desc=f'TENT (seed={seed})'):
        try:
            loader, _ = get_shuffled_loader(dataset, corruption, severity, seed, data_dir, batch_size)
            
            correct = 0
            total = 0
            
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                p = F.softmax(outputs, dim=1)
                entropy = -(p * torch.log(p + 1e-10)).sum(dim=1).mean()
                
                optimizer.zero_grad()
                entropy.backward()
                optimizer.step()
                
                with torch.no_grad():
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            acc = 100.0 * correct / total
            results['corruptions'][corruption] = {'accuracy': acc}
            all_accs.append(acc)
            
        except Exception as e:
            print(f"Error: {e}")
    
    if all_accs:
        results['mean_accuracy'] = float(np.mean(all_accs))
    
    del model, optimizer
    return results


# ============== Batch-based MEMO (Faster) ==============

def evaluate_memo_batch(seed=2022, dataset='cifar10', severity=5, model_name='resnet32',
                        lr=1e-3, data_dir='./data', device='cuda'):
    """
    MEMO with batch-based marginal entropy (faster than single-image).
    Uses small batches for virtual ensemble.
    """
    set_seed(seed)
    
    model = load_pretrained_cifar_model(model_name=model_name, dataset=dataset, device=device)
    
    import torchvision.transforms as transforms
    augmentations = [
        transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 1.2)),
        transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 0.8)),
        transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, 1.2)),
        transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, 0.8)),
        transforms.Lambda(lambda x: transforms.functional.gaussian_blur(x, kernel_size=3)),
        transforms.RandomRotation(degrees=10),
    ]
    
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    results = {'corruptions': {}, 'seed': seed}
    all_accs = []
    
    for corruption in tqdm(corruptions, desc=f'MEMO-batch (seed={seed})'):
        try:
            loader, _ = get_shuffled_loader(dataset, corruption, severity, seed, data_dir, batch_size=50)
            
            correct = 0
            total = 0
            
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Create fresh model for this batch
                model_adapt = copy.deepcopy(model)
                model_adapt = enable_adaptation(model_adapt, adapt_bn=True)
                params, _ = collect_params(model_adapt)
                optimizer = torch.optim.Adam(params, lr=lr)
                
                # Create augmented views of batch
                aug_logits = []
                for aug in augmentations:
                    aug_images = torch.stack([aug(img) for img in images])
                    logits = model_adapt(aug_images)
                    aug_logits.append(logits)
                
                # Marginal entropy over all augmented predictions
                all_logits = torch.cat(aug_logits, dim=0)
                avg_probs = F.softmax(all_logits, dim=1).mean(dim=0)
                marginal_entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
                
                optimizer.zero_grad()
                marginal_entropy.backward()
                optimizer.step()
                
                with torch.no_grad():
                    output = model_adapt(images)
                    _, predicted = output.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
                
                del model_adapt, optimizer
            
            acc = 100.0 * correct / total if total > 0 else 0.0
            results['corruptions'][corruption] = {'accuracy': acc}
            all_accs.append(acc)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if all_accs:
        results['mean_accuracy'] = float(np.mean(all_accs))
    
    del model
    return results


# ============== Meta-APN ==============

class MetaAPN(nn.Module):
    """Meta-Augmentation Policy Network"""
    def __init__(self, feature_dim, num_classes, num_operations=8, hidden_dim=64):
        super(MetaAPN, self).__init__()
        self.num_operations = num_operations
        self.fc1 = nn.Linear(feature_dim + num_classes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_operations + 1)
        self.relu = nn.ReLU()
        
    def forward(self, features, prototype_distances):
        x = torch.cat([features, prototype_distances], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        policy_logits = x[:, :self.num_operations]
        severity_scale = torch.sigmoid(x[:, self.num_operations:]) * 1.5 + 0.5
        return policy_logits, severity_scale


def train_meta_apn_quick(dataset='cifar10', model_name='resnet32', epochs=5, device='cuda'):
    """Train Meta-APN (quick version with fewer epochs)"""
    print(f"\nTraining Meta-APN ({epochs} epochs)...")
    
    if dataset == 'cifar10':
        loader, _ = load_cifar10('./data', train=True, batch_size=128)
        num_classes = 10
    else:
        loader, _ = load_cifar100('./data', train=True, batch_size=128)
        num_classes = 100
    
    model = load_pretrained_cifar_model(model_name=model_name, dataset=dataset, device=device)
    model.eval()
    
    print("Computing prototypes...")
    prototypes = compute_prototypes(model, loader, device, num_classes)
    
    feature_dim = 64
    meta_apn = MetaAPN(feature_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(meta_apn.parameters(), lr=1e-3)
    
    import torchvision.transforms as transforms
    aug_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 1.2)),
        transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, 1.2)),
        transforms.Lambda(lambda x: transforms.functional.gaussian_blur(x, 3)),
    ]
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(loader, desc=f'Epoch {epoch+1}', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                features = model.get_features(images)
                features_norm = F.normalize(features, p=2, dim=1)
                prototypes_norm = F.normalize(prototypes, p=2, dim=1).to(device)
                distances = torch.cdist(features_norm, prototypes_norm)
            
            policy_logits, _ = meta_apn(features, distances)
            
            # Apply policy
            aug_images = []
            for i, img in enumerate(images):
                probs = F.softmax(policy_logits[i] / 0.5, dim=0)
                aug_idx = torch.multinomial(probs, 1).item()
                if aug_idx < len(aug_transforms):
                    aug_img = aug_transforms[aug_idx](img.unsqueeze(0)).squeeze(0)
                else:
                    aug_img = img
                aug_images.append(aug_img)
            
            aug_images = torch.stack(aug_images)
            outputs = model(aug_images)
            loss = F.cross_entropy(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={100*correct/total:.2f}%")
    
    os.makedirs('models', exist_ok=True)
    torch.save(meta_apn.state_dict(), f'models/meta_apn_{dataset}_trained.pt')
    torch.save(prototypes, f'models/prototypes_{dataset}_computed.pt')
    
    return meta_apn, prototypes


# ============== APAC-TTA (Full) ==============

def evaluate_apac_tta(seed=2022, dataset='cifar10', severity=5, model_name='resnet32',
                      lr=1e-3, data_dir='./data', device='cuda',
                      use_meta_apn=True, use_confidence=True):
    """APAC-TTA with Meta-APN and confidence weighting"""
    set_seed(seed)
    
    model = load_pretrained_cifar_model(model_name=model_name, dataset=dataset, device=device)
    
    num_classes = 10 if dataset == 'cifar10' else 100
    feature_dim = 64
    
    # Load prototypes
    proto_path = f'models/prototypes_{dataset}_computed.pt'
    if os.path.exists(proto_path):
        prototypes = torch.load(proto_path, map_location=device)
    else:
        print("Computing prototypes...")
        if dataset == 'cifar10':
            train_loader, _ = load_cifar10(data_dir, train=True, batch_size=128)
        else:
            train_loader, _ = load_cifar100(data_dir, train=True, batch_size=128)
        prototypes = compute_prototypes(model, train_loader, device, num_classes)
        torch.save(prototypes, proto_path)
    
    # Load Meta-APN
    meta_apn = None
    if use_meta_apn:
        meta_apn_path = f'models/meta_apn_{dataset}_trained.pt'
        if os.path.exists(meta_apn_path):
            meta_apn = MetaAPN(feature_dim, num_classes).to(device)
            meta_apn.load_state_dict(torch.load(meta_apn_path, map_location=device))
            meta_apn.eval()
    
    import torchvision.transforms as transforms
    augmentations = [
        transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 1.2)),
        transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 0.8)),
        transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, 1.2)),
        transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, 0.8)),
        transforms.Lambda(lambda x: transforms.functional.gaussian_blur(x, kernel_size=3)),
        transforms.RandomRotation(degrees=10),
    ]
    
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    results = {'corruptions': {}, 'seed': seed}
    all_accs = []
    
    for corruption in tqdm(corruptions, desc=f'APAC-TTA (seed={seed})'):
        try:
            loader, _ = get_shuffled_loader(dataset, corruption, severity, seed, data_dir, batch_size=50)
            
            correct = 0
            total = 0
            
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Fresh model for each batch
                model_adapt = copy.deepcopy(model)
                model_adapt = enable_adaptation(model_adapt, adapt_bn=True)
                params, _ = collect_params(model_adapt)
                optimizer = torch.optim.Adam(params, lr=lr)
                
                # Get features
                with torch.no_grad():
                    features = model_adapt.get_features(images)
                    features_norm = F.normalize(features, p=2, dim=1)
                    prototypes_norm = F.normalize(prototypes, p=2, dim=1).to(device)
                    similarities = torch.mm(features_norm, prototypes_norm.t())
                    distances = 1 - similarities
                
                # Select augmentations
                if meta_apn is not None:
                    with torch.no_grad():
                        policy_logits, _ = meta_apn(features, distances)
                        probs = F.softmax(policy_logits / 0.5, dim=1)
                        selected_augs = torch.multinomial(probs[0], min(4, len(augmentations)), replacement=False)
                else:
                    confidences = similarities.max(dim=1)[0]
                    if confidences.mean().item() > 0.7:
                        selected_augs = torch.tensor([0, 1, 2, 3])
                    else:
                        selected_augs = torch.tensor([0, 4, 5, 6])
                
                # Create augmented views
                aug_views = [images]
                for aug_idx in selected_augs:
                    aug_idx = aug_idx.item()
                    if aug_idx < len(augmentations):
                        aug_images = torch.stack([augmentations[aug_idx](img) for img in images])
                        aug_views.append(aug_images)
                
                aug_batch = torch.cat(aug_views, dim=0)
                
                # Forward
                logits = model_adapt(aug_batch)
                probs = F.softmax(logits, dim=1)
                
                # Confidence weighting
                if use_confidence:
                    with torch.no_grad():
                        pred_variance = probs.var(dim=0).mean()
                        confidence_weight = torch.exp(-pred_variance * 10)
                else:
                    confidence_weight = 1.0
                
                # Prototype consistency
                aug_features = model_adapt.get_features(aug_batch)
                aug_features_norm = F.normalize(aug_features, p=2, dim=1)
                aug_similarities = torch.mm(aug_features_norm, prototypes_norm.t())
                proto_targets = F.softmax(aug_similarities / 0.5, dim=1)
                
                kl_loss = F.kl_div(torch.log(probs + 1e-10), proto_targets, reduction='batchmean')
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
                
                loss = entropy + confidence_weight * kl_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    output = model_adapt(images)
                    _, predicted = output.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
                
                del model_adapt, optimizer
            
            acc = 100.0 * correct / total if total > 0 else 0.0
            results['corruptions'][corruption] = {'accuracy': acc}
            all_accs.append(acc)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if all_accs:
        results['mean_accuracy'] = float(np.mean(all_accs))
    
    del model
    return results


# ============== Statistical Tests ==============

def paired_t_test(results1, results2, name1='M1', name2='M2'):
    """Paired t-test"""
    accs1 = [r['mean_accuracy'] for r in results1.get('per_seed', []) if 'mean_accuracy' in r]
    accs2 = [r['mean_accuracy'] for r in results2.get('per_seed', []) if 'mean_accuracy' in r]
    
    if len(accs1) != len(accs2) or len(accs1) < 2:
        return {'error': 'Insufficient data'}
    
    statistic, p_value = stats.ttest_rel(accs1, accs2)
    diff = np.array(accs1) - np.array(accs2)
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
    
    return {
        'method1': name1, 'method2': name2,
        'mean_diff': float(diff.mean()),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'cohens_d': float(cohens_d),
    }


# ============== Main ==============

def run_fast_evaluations():
    """Run optimized evaluations"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    all_results = {}
    
    # Train Meta-APN
    print("\n" + "="*60)
    print("Step 1: Training Meta-APN")
    print("="*60)
    try:
        train_meta_apn_quick(dataset='cifar10', model_name='resnet32', epochs=5, device=device)
    except Exception as e:
        print(f"Meta-APN training failed: {e}")
    
    # Source
    print("\n" + "="*60)
    print("Step 2: Source Baseline")
    print("="*60)
    source_results = evaluate_with_seeds(
        evaluate_source, dataset='cifar10', n_seeds=3,
        severity=5, model_name='resnet32', data_dir='./data', device=device
    )
    all_results['source'] = source_results
    os.makedirs('exp/fixed_results', exist_ok=True)
    with open('exp/fixed_results/source_cifar10_final.json', 'w') as f:
        json.dump(source_results, f, indent=2)
    print(f"Source: {source_results.get('mean_accuracy', 0):.2f} ± {source_results.get('std_accuracy', 0):.2f}%")
    
    # TENT
    print("\n" + "="*60)
    print("Step 3: TENT Baseline")
    print("="*60)
    tent_results = evaluate_with_seeds(
        evaluate_tent, dataset='cifar10', n_seeds=3,
        severity=5, model_name='resnet32', lr=1e-3, data_dir='./data', device=device
    )
    all_results['tent'] = tent_results
    with open('exp/fixed_results/tent_cifar10_final.json', 'w') as f:
        json.dump(tent_results, f, indent=2)
    print(f"TENT: {tent_results.get('mean_accuracy', 0):.2f} ± {tent_results.get('std_accuracy', 0):.2f}%")
    
    # MEMO (batch-based, faster)
    print("\n" + "="*60)
    print("Step 4: MEMO Baseline (batch-based)")
    print("="*60)
    memo_results = evaluate_with_seeds(
        evaluate_memo_batch, dataset='cifar10', n_seeds=3,
        severity=5, model_name='resnet32', lr=1e-3, data_dir='./data', device=device
    )
    all_results['memo'] = memo_results
    with open('exp/fixed_results/memo_cifar10_final.json', 'w') as f:
        json.dump(memo_results, f, indent=2)
    print(f"MEMO: {memo_results.get('mean_accuracy', 0):.2f} ± {memo_results.get('std_accuracy', 0):.2f}%")
    
    # APAC-TTA (Full)
    print("\n" + "="*60)
    print("Step 5: APAC-TTA (Full)")
    print("="*60)
    apac_full_results = evaluate_with_seeds(
        evaluate_apac_tta, dataset='cifar10', n_seeds=3,
        severity=5, model_name='resnet32', lr=1e-3, data_dir='./data', device=device,
        use_meta_apn=True, use_confidence=True
    )
    all_results['apac_full'] = apac_full_results
    with open('exp/fixed_results/apac_full_cifar10_final.json', 'w') as f:
        json.dump(apac_full_results, f, indent=2)
    print(f"APAC-TTA (Full): {apac_full_results.get('mean_accuracy', 0):.2f} ± {apac_full_results.get('std_accuracy', 0):.2f}%")
    
    # APAC-TTA (Heuristic only)
    print("\n" + "="*60)
    print("Step 6: APAC-TTA (Heuristic - ablation)")
    print("="*60)
    apac_heuristic_results = evaluate_with_seeds(
        evaluate_apac_tta, dataset='cifar10', n_seeds=3,
        severity=5, model_name='resnet32', lr=1e-3, data_dir='./data', device=device,
        use_meta_apn=False, use_confidence=True
    )
    all_results['apac_heuristic'] = apac_heuristic_results
    with open('exp/fixed_results/apac_heuristic_cifar10_final.json', 'w') as f:
        json.dump(apac_heuristic_results, f, indent=2)
    print(f"APAC-TTA (Heuristic): {apac_heuristic_results.get('mean_accuracy', 0):.2f} ± {apac_heuristic_results.get('std_accuracy', 0):.2f}%")
    
    # APAC-TTA (No confidence)
    print("\n" + "="*60)
    print("Step 7: APAC-TTA (No Confidence - ablation)")
    print("="*60)
    apac_noconf_results = evaluate_with_seeds(
        evaluate_apac_tta, dataset='cifar10', n_seeds=3,
        severity=5, model_name='resnet32', lr=1e-3, data_dir='./data', device=device,
        use_meta_apn=True, use_confidence=False
    )
    all_results['apac_noconf'] = apac_noconf_results
    with open('exp/fixed_results/apac_noconf_cifar10_final.json', 'w') as f:
        json.dump(apac_noconf_results, f, indent=2)
    print(f"APAC-TTA (No Conf): {apac_noconf_results.get('mean_accuracy', 0):.2f} ± {apac_noconf_results.get('std_accuracy', 0):.2f}%")
    
    # Statistical tests
    print("\n" + "="*60)
    print("Step 8: Statistical Tests")
    print("="*60)
    
    stats_results = {}
    stats_results['tent_vs_source'] = paired_t_test(tent_results, source_results, 'TENT', 'Source')
    stats_results['memo_vs_tent'] = paired_t_test(memo_results, tent_results, 'MEMO', 'TENT')
    stats_results['apac_vs_tent'] = paired_t_test(apac_full_results, tent_results, 'APAC-TTA', 'TENT')
    stats_results['apac_vs_memo'] = paired_t_test(apac_full_results, memo_results, 'APAC-TTA', 'MEMO')
    stats_results['full_vs_heuristic'] = paired_t_test(apac_full_results, apac_heuristic_results, 'Full', 'Heuristic')
    stats_results['full_vs_noconf'] = paired_t_test(apac_full_results, apac_noconf_results, 'Full', 'NoConfidence')
    
    all_results['statistics'] = stats_results
    with open('exp/fixed_results/statistics_final.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Method':<25} {'Accuracy':<20}")
    print("-" * 60)
    for key in ['source', 'tent', 'memo', 'apac_full', 'apac_heuristic', 'apac_noconf']:
        if key in all_results:
            r = all_results[key]
            print(f"{key:<25} {r['mean_accuracy']:.2f} ± {r['std_accuracy']:.2f}%")
    
    print("\n" + "="*60)
    print("Statistical Tests (p-values)")
    print("="*60)
    for key, stat in stats_results.items():
        if 'error' not in stat:
            sig = "***" if stat['p_value'] < 0.05 else "ns"
            print(f"{key}: p={stat['p_value']:.4f} {sig}, diff={stat['mean_diff']:+.2f}%")
    
    with open('exp/fixed_results/all_results_final.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


if __name__ == '__main__':
    results = run_fast_evaluations()
