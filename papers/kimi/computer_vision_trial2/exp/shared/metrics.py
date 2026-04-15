"""
Evaluation metrics for TTA experiments.
"""

import torch
import numpy as np
from scipy import stats


def accuracy(outputs, targets):
    """Compute top-1 accuracy"""
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def top_k_accuracy(outputs, targets, k=5):
    """Compute top-k accuracy"""
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / targets.size(0)).item()


def compute_mce(errors, alexnet_errors):
    """
    Compute mean Corruption Error (mCE).
    
    Args:
        errors: Dict mapping corruption name to error rate
        alexnet_errors: Dict mapping corruption name to AlexNet baseline error
    
    Returns:
        mCE as percentage
    """
    ce_values = []
    for corruption, error in errors.items():
        if corruption in alexnet_errors:
            # Corruption Error = error / AlexNet_error * 100
            ce = (error / alexnet_errors[corruption]) * 100
            ce_values.append(ce)
    
    return np.mean(ce_values) if ce_values else 0


# AlexNet baseline errors for CIFAR-10-C (from Hendrycks & Dietterich)
ALEXNET_CIFAR10_C_ERRORS = {
    'gaussian_noise': 88.2,
    'shot_noise': 80.9,
    'impulse_noise': 78.0,
    'defocus_blur': 67.4,
    'glass_blur': 81.4,
    'motion_blur': 66.2,
    'zoom_blur': 68.8,
    'snow': 67.6,
    'frost': 70.9,
    'fog': 68.4,
    'brightness': 53.2,
    'contrast': 76.1,
    'elastic_transform': 68.8,
    'pixelate': 71.1,
    'jpeg_compression': 58.8,
}

# AlexNet baseline errors for CIFAR-100-C
ALEXNET_CIFAR100_C_ERRORS = {
    'gaussian_noise': 93.2,
    'shot_noise': 90.5,
    'impulse_noise': 88.6,
    'defocus_blur': 82.5,
    'glass_blur': 89.7,
    'motion_blur': 80.9,
    'zoom_blur': 83.1,
    'snow': 81.9,
    'frost': 84.5,
    'fog': 80.5,
    'brightness': 64.9,
    'contrast': 86.7,
    'elastic_transform': 82.2,
    'pixelate': 82.4,
    'jpeg_compression': 70.8,
}

# AlexNet baseline errors for ImageNet-C
ALEXNET_IMAGENET_C_ERRORS = {
    'gaussian_noise': 89.6,
    'shot_noise': 85.2,
    'impulse_noise': 87.2,
    'defocus_blur': 72.2,
    'glass_blur': 80.3,
    'motion_blur': 69.9,
    'zoom_blur': 74.5,
    'snow': 68.6,
    'frost': 68.0,
    'fog': 61.3,
    'brightness': 46.9,
    'contrast': 68.1,
    'elastic_transform': 65.9,
    'pixelate': 71.8,
    'jpeg_compression': 54.1,
}


def get_alexnet_errors(dataset='cifar10'):
    """Get AlexNet baseline errors for a dataset"""
    if dataset == 'cifar10':
        return ALEXNET_CIFAR10_C_ERRORS
    elif dataset == 'cifar100':
        return ALEXNET_CIFAR100_C_ERRORS
    elif dataset == 'imagenet':
        return ALEXNET_IMAGENET_C_ERRORS
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def paired_ttest(baseline_acc, method_acc):
    """
    Perform paired t-test to compare baseline and method.
    
    Args:
        baseline_acc: List of accuracies for baseline (per image or per run)
        method_acc: List of accuracies for method
    
    Returns:
        t_statistic, p_value
    """
    baseline_acc = np.array(baseline_acc)
    method_acc = np.array(method_acc)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(method_acc, baseline_acc)
    
    return t_stat, p_value


def cohens_d(baseline_acc, method_acc):
    """
    Compute Cohen's d for effect size.
    
    Args:
        baseline_acc: List of accuracies
        method_acc: List of accuracies
    
    Returns:
        Cohen's d
    """
    baseline_acc = np.array(baseline_acc)
    method_acc = np.array(method_acc)
    
    diff = method_acc - baseline_acc
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    return mean_diff / std_diff if std_diff > 0 else 0


def compute_statistics(results_dict):
    """
    Compute mean and standard error from results.
    
    Args:
        results_dict: Dict with lists of results per seed
    
    Returns:
        Dict with mean and std_error
    """
    stats_dict = {}
    for key, values in results_dict.items():
        values = np.array(values)
        stats_dict[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)),
            'se': float(np.std(values, ddof=1) / np.sqrt(len(values))),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    return stats_dict


def compute_confidence_scores(features, prototypes, augmentations, model, device='cuda'):
    """
    Compute confidence scores for prototypes based on consistency across augmentations.
    
    Args:
        features: Original feature vector [feature_dim]
        prototypes: Class prototypes [num_classes, feature_dim]
        augmentations: List of augmented images
        model: Model to extract features
        device: Device
    
    Returns:
        confidence_scores: [num_classes] confidence for each prototype
    """
    num_classes = prototypes.size(0)
    similarities = []
    
    # Get features for all augmented views
    with torch.no_grad():
        aug_features = []
        for aug_img in augmentations:
            if isinstance(aug_img, torch.Tensor):
                aug_img = aug_img.unsqueeze(0).to(device)
            feat = model.get_features(aug_img)
            aug_features.append(feat)
        
        aug_features = torch.cat(aug_features, dim=0)  # [num_aug, feature_dim]
        
        # Compute cosine similarity to each prototype
        aug_features_norm = torch.nn.functional.normalize(aug_features, p=2, dim=1)
        prototypes_norm = torch.nn.functional.normalize(prototypes, p=2, dim=1)
        
        # [num_aug, num_classes]
        sim_matrix = torch.mm(aug_features_norm, prototypes_norm.t())
        
        # For each prototype, compute confidence based on consistency
        confidence_scores = []
        for c in range(num_classes):
            sims = sim_matrix[:, c]
            # Consistency = 1 / (1 + variance) * mean_similarity
            consistency = 1.0 / (1.0 + torch.var(sims))
            mean_sim = torch.mean(sims)
            confidence = consistency * mean_sim
            confidence_scores.append(confidence)
    
    return torch.stack(confidence_scores)


def js_divergence(p, q):
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    Args:
        p: [B, C] probability distribution
        q: [B, C] probability distribution
    
    Returns:
        JS divergence
    """
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log(p / (m + 1e-10) + 1e-10), dim=-1)
    kl_qm = torch.sum(q * torch.log(q / (m + 1e-10) + 1e-10), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
