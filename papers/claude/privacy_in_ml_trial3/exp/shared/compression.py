"""Model compression: pruning utilities with proper mask preservation.

Key fix from v1: Uses torch.nn.utils.prune hooks to maintain sparsity
masks during fine-tuning. prune.remove() is only called after fine-tuning
via finalize_pruning().
"""

import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from collections import defaultdict


def magnitude_prune(model, sparsity, structured=False, keep_masks=True):
    """Apply global magnitude-based pruning.

    Args:
        model: PyTorch model
        sparsity: fraction of weights to prune (0.0 to 1.0)
        structured: if True, apply structured (filter-level) pruning on Conv2d
        keep_masks: if True, keep pruning hooks active (for fine-tuning).
                    Call finalize_pruning() after fine-tuning.

    Returns:
        pruned model (deepcopy)
    """
    model = copy.deepcopy(model)

    if structured:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=sparsity, n=1, dim=0)
                if not keep_masks:
                    prune.remove(module, 'weight')
    else:
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))

        if len(parameters_to_prune) > 0:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity,
            )
            if not keep_masks:
                for module, param_name in parameters_to_prune:
                    prune.remove(module, param_name)

    return model


def fisher_prune(model, sparsity, data_loader, device, n_samples=1000, keep_masks=True):
    """Prune based on global Fisher information (not subgroup-aware)."""
    model = copy.deepcopy(model)
    fisher_scores = compute_fisher_importance(model, data_loader, device, n_samples)
    return _prune_by_scores(model, fisher_scores, sparsity, keep_masks=keep_masks)


def fairprune_dp(model, sparsity, data_loader, device, n_samples=2000, alpha=0.5,
                 normalize=True, keep_masks=True):
    """FairPrune-DP: prune using worst-group-aware Fisher information.

    Improved version with per-subgroup normalization to handle scale differences
    between subgroups (especially important for DP models where minority Fisher
    estimates are noisier).

    Importance: s_i^fair = alpha * min_g(norm_s_i^g) + (1 - alpha) * mean_g(norm_s_i^g)
    where norm_s_i^g = s_i^g / mean(s^g) normalizes each subgroup's Fisher to unit mean.

    Args:
        normalize: if True, normalize per-subgroup Fisher before aggregation
        alpha: weight on worst-group (min) vs mean. 0.5 = equal blend.
    """
    model = copy.deepcopy(model)
    subgroup_fisher = compute_subgroup_fisher(model, data_loader, device, n_samples)

    subgroup_keys = list(subgroup_fisher.keys())

    # Compute per-subgroup global mean for normalization
    subgroup_global_means = {}
    if normalize:
        for sg in subgroup_keys:
            all_scores = []
            for key, scores in subgroup_fisher[sg].items():
                all_scores.append(scores.flatten())
            if all_scores:
                cat = torch.cat(all_scores)
                subgroup_global_means[sg] = cat.mean().item() + 1e-12
            else:
                subgroup_global_means[sg] = 1e-12

    fair_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            key = name + ".weight"
            group_scores = []
            for sg in subgroup_keys:
                raw = subgroup_fisher[sg].get(key, torch.zeros_like(module.weight))
                if normalize:
                    raw = raw / subgroup_global_means[sg]
                group_scores.append(raw)
            if group_scores:
                stacked = torch.stack(group_scores)
                min_scores = stacked.min(dim=0)[0]
                mean_scores = stacked.mean(dim=0)
                fair_scores[key] = alpha * min_scores + (1 - alpha) * mean_scores
            else:
                fair_scores[key] = torch.zeros_like(module.weight)

    return _prune_by_scores(model, fair_scores, sparsity, keep_masks=keep_masks)


def fairprune_dp_hard_min(model, sparsity, data_loader, device, n_samples=2000,
                           normalize=True, keep_masks=True):
    """FairPrune-DP with hard min criterion (for ablation comparison)."""
    return fairprune_dp(model, sparsity, data_loader, device, n_samples,
                        alpha=1.0, normalize=normalize, keep_masks=keep_masks)


def mean_fisher_prune(model, sparsity, data_loader, device, n_samples=1000, keep_masks=True):
    """Prune using mean-subgroup Fisher information (ablation baseline)."""
    model = copy.deepcopy(model)
    subgroup_fisher = compute_subgroup_fisher(model, data_loader, device, n_samples)

    mean_scores = {}
    subgroup_keys = list(subgroup_fisher.keys())
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            key = name + ".weight"
            group_scores = [subgroup_fisher[sg].get(key, torch.zeros_like(module.weight))
                           for sg in subgroup_keys]
            if group_scores:
                stacked = torch.stack(group_scores)
                mean_scores[key] = stacked.mean(dim=0)
            else:
                mean_scores[key] = torch.zeros_like(module.weight)

    return _prune_by_scores(model, mean_scores, sparsity, keep_masks=keep_masks)


def finalize_pruning(model):
    """Remove pruning hooks, making the mask permanent.

    Call this AFTER fine-tuning to permanently zero pruned weights.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
    return model


def compute_fisher_importance(model, data_loader, device, n_samples=1000):
    """Compute per-weight Fisher information over all samples."""
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    fisher = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            fisher[name + ".weight"] = torch.zeros_like(module.weight)

    count = 0
    for batch in data_loader:
        if count >= n_samples:
            break
        images, labels, subgroups = batch
        batch_size = min(len(images), n_samples - count)
        images = images[:batch_size].to(device)
        labels_t = torch.tensor(labels[:batch_size], dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels[:batch_size].to(device)

        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_t)
        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                key = name + ".weight"
                if module.weight.grad is not None:
                    fisher[key] += (module.weight.grad ** 2).detach() * batch_size

        count += batch_size

    for key in fisher:
        fisher[key] /= max(count, 1)

    return fisher


def compute_subgroup_fisher(model, data_loader, device, n_samples=1000):
    """Compute per-weight, per-subgroup Fisher information.

    Returns: dict mapping subgroup_id -> dict mapping param_name -> importance tensor
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    subgroup_data = defaultdict(lambda: {"images": [], "labels": []})
    count = 0
    for batch in data_loader:
        if count >= n_samples:
            break
        images, labels, subgroups = batch
        if isinstance(labels, torch.Tensor):
            labels_np = labels.numpy()
        else:
            labels_np = np.array(labels)
        if isinstance(subgroups, torch.Tensor):
            subgroups_np = subgroups.numpy()
        else:
            subgroups_np = np.array(subgroups)

        for i in range(len(images)):
            if count >= n_samples:
                break
            sg = int(subgroups_np[i])
            subgroup_data[sg]["images"].append(images[i])
            subgroup_data[sg]["labels"].append(int(labels_np[i]))
            count += 1

    result = {}
    for sg, data in subgroup_data.items():
        fisher = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                fisher[name + ".weight"] = torch.zeros_like(module.weight)

        imgs = torch.stack(data["images"]).to(device)
        labs = torch.tensor(data["labels"], dtype=torch.long).to(device)
        n = len(imgs)

        bs = 64
        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch_imgs = imgs[start:end]
            batch_labs = labs[start:end]

            model.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labs)
            loss.backward()

            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    key = name + ".weight"
                    if module.weight.grad is not None:
                        fisher[key] += (module.weight.grad ** 2).detach() * (end - start)

        for key in fisher:
            fisher[key] /= max(n, 1)

        result[sg] = fisher

    return result


def _prune_by_scores(model, importance_scores, sparsity, keep_masks=True):
    """Prune model weights based on importance scores using prune.custom_from_mask.

    This preserves masks via forward hooks so fine-tuning maintains sparsity.
    """
    all_scores = []
    param_info = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            key = name + ".weight"
            if key in importance_scores:
                all_scores.append(importance_scores[key].flatten())
                param_info.append((module, name, key))

    if not all_scores:
        return model

    all_scores_cat = torch.cat(all_scores)
    threshold = torch.quantile(all_scores_cat.float(), sparsity)

    for module, name, key in param_info:
        mask = (importance_scores[key] > threshold).float()
        prune.custom_from_mask(module, name='weight', mask=mask)
        if not keep_masks:
            prune.remove(module, 'weight')

    return model


def get_sparsity(model):
    """Compute actual sparsity of a model."""
    total = 0
    zeros = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            if hasattr(module, 'weight_mask'):
                w = module.weight_orig * module.weight_mask
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / max(total, 1)


def get_weight_stats_by_subgroup_relevance(model, subgroup_fisher, minority_subgroups):
    """Classify weights as minority-relevant or majority-relevant based on Fisher info."""
    all_params = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            key = name + ".weight"
            all_params[key] = module.weight.data.abs().flatten().cpu().numpy()

    minority_fisher_max = None
    majority_fisher_max = None

    for sg, fisher_dict in subgroup_fisher.items():
        for key, scores in fisher_dict.items():
            flat = scores.flatten().cpu()
            if sg in minority_subgroups:
                if minority_fisher_max is None:
                    minority_fisher_max = {key: flat}
                elif key in minority_fisher_max:
                    minority_fisher_max[key] = torch.max(minority_fisher_max[key], flat)
                else:
                    minority_fisher_max[key] = flat
            else:
                if majority_fisher_max is None:
                    majority_fisher_max = {key: flat}
                elif key in majority_fisher_max:
                    majority_fisher_max[key] = torch.max(majority_fisher_max[key], flat)
                else:
                    majority_fisher_max[key] = flat

    if minority_fisher_max is None or majority_fisher_max is None:
        return None

    minority_mags = []
    majority_mags = []

    for key, weights in all_params.items():
        if key in minority_fisher_max and key in majority_fisher_max:
            min_f = minority_fisher_max[key].numpy()
            maj_f = majority_fisher_max[key].numpy()
            median_min = np.median(min_f)
            is_minority_relevant = min_f > median_min
            minority_mags.extend(weights[is_minority_relevant].tolist())
            majority_mags.extend(weights[~is_minority_relevant].tolist())

    return {
        "minority_relevant_magnitude_mean": float(np.mean(minority_mags)) if minority_mags else 0,
        "minority_relevant_magnitude_std": float(np.std(minority_mags)) if minority_mags else 0,
        "majority_relevant_magnitude_mean": float(np.mean(majority_mags)) if majority_mags else 0,
        "majority_relevant_magnitude_std": float(np.std(majority_mags)) if majority_mags else 0,
        "minority_relevant_count": len(minority_mags),
        "majority_relevant_count": len(majority_mags),
    }


def get_pruning_overlap_with_minority(model_standard, model_dp, sparsity, subgroup_fisher_std, subgroup_fisher_dp, minority_subgroups):
    """Compute what fraction of pruned weights are minority-relevant, for standard vs DP models."""
    results = {}
    for label, model, sg_fisher in [("standard", model_standard, subgroup_fisher_std),
                                      ("dp", model_dp, subgroup_fisher_dp)]:
        pruned_model = magnitude_prune(model, sparsity, keep_masks=False)

        pruned_mask_all = []
        minority_relevant_all = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                key = name + ".weight"
                orig_w = module.weight.data.abs().flatten()
                pruned_module = dict(pruned_model.named_modules())[name]
                pruned_w = pruned_module.weight.data.abs().flatten()
                is_pruned = (pruned_w == 0) & (orig_w != 0)
                pruned_mask_all.append(is_pruned.cpu())

                minority_fisher_scores = []
                for sg, fisher_dict in sg_fisher.items():
                    if sg in minority_subgroups and key in fisher_dict:
                        minority_fisher_scores.append(fisher_dict[key].flatten().cpu())

                if minority_fisher_scores:
                    max_minority_fisher = torch.stack(minority_fisher_scores).max(dim=0)[0]
                    median_f = max_minority_fisher.median()
                    is_minority_rel = max_minority_fisher > median_f
                else:
                    is_minority_rel = torch.zeros(orig_w.shape[0], dtype=torch.bool)

                minority_relevant_all.append(is_minority_rel)

        pruned_mask = torch.cat(pruned_mask_all)
        minority_rel = torch.cat(minority_relevant_all)

        n_pruned = pruned_mask.sum().item()
        n_pruned_minority = (pruned_mask & minority_rel).sum().item()

        results[label] = {
            "total_pruned": int(n_pruned),
            "minority_relevant_pruned": int(n_pruned_minority),
            "fraction_minority_relevant": n_pruned_minority / max(n_pruned, 1),
        }

    return results
