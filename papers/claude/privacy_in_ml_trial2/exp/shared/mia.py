import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from .config import DEVICE


def compute_losses(model, data_loader, device=DEVICE):
    """Compute per-sample CE losses."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_losses = []
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            loss = criterion(out, y)
            all_losses.append(loss.cpu().numpy())
    return np.concatenate(all_losses)


def loss_threshold_attack(member_losses, nonmember_losses):
    """Loss-threshold MIA: members have lower loss (higher score = -loss)."""
    scores = np.concatenate([-member_losses, -nonmember_losses])
    labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(nonmember_losses))])
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5
    return auc


def calibrated_attack(member_losses, nonmember_losses, member_ref_losses, nonmember_ref_losses):
    """Calibrated MIA (Watson et al., 2022): score = -(loss - mean_ref_loss)."""
    member_scores = -(member_losses - member_ref_losses)
    nonmember_scores = -(nonmember_losses - nonmember_ref_losses)
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5
    return auc


def lira_attack(member_losses, nonmember_losses, ref_member_losses_list, ref_nonmember_losses_list):
    """Simplified LiRA: Gaussian-based likelihood ratio using reference models.

    ref_member_losses_list: list of arrays, one per ref model, losses on member samples
    ref_nonmember_losses_list: list of arrays, one per ref model, losses on nonmember samples
    """
    # Stack reference losses: shape (K, N)
    ref_member = np.stack(ref_member_losses_list, axis=0)  # (K, n_members)
    ref_nonmember = np.stack(ref_nonmember_losses_list, axis=0)  # (K, n_nonmembers)

    # For members: estimate p(loss|in) and p(loss|out) using reference models
    # Reference models that included the sample → "in" distribution
    # Reference models that excluded it → "out" distribution
    # Simplified: use mean/std of ref losses as proxy for out-distribution
    ref_all = np.concatenate([ref_member, ref_nonmember], axis=1)  # (K, N_total)
    ref_mean = ref_all.mean(axis=0)  # per-sample across ref models — not right for LiRA

    # Proper simplified LiRA: for each sample, use ref model losses as the null distribution
    # Score = how much lower the target model loss is compared to ref model losses
    member_ref_mean = ref_member.mean(axis=0)
    member_ref_std = ref_member.std(axis=0) + 1e-8
    nonmember_ref_mean = ref_nonmember.mean(axis=0)
    nonmember_ref_std = ref_nonmember.std(axis=0) + 1e-8

    # Z-score: negative means target loss is lower than ref (more memorized)
    member_scores = -(member_losses - member_ref_mean) / member_ref_std
    nonmember_scores = -(nonmember_losses - nonmember_ref_mean) / nonmember_ref_std

    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5
    return auc


def run_all_attacks(member_losses, nonmember_losses,
                    member_ref_mean_losses, nonmember_ref_mean_losses,
                    member_ref_losses_list, nonmember_ref_losses_list):
    """Run all MIA attacks, return dict of AUCs."""
    loss_auc = loss_threshold_attack(member_losses, nonmember_losses)
    cal_auc = calibrated_attack(member_losses, nonmember_losses,
                                member_ref_mean_losses, nonmember_ref_mean_losses)
    lira_auc = lira_attack(member_losses, nonmember_losses,
                           member_ref_losses_list, nonmember_ref_losses_list)
    best_auc = max(loss_auc, cal_auc, lira_auc)
    return {
        'loss_auc': float(loss_auc),
        'calibrated_auc': float(cal_auc),
        'lira_auc': float(lira_auc),
        'best_auc': float(best_auc),
    }


def stratified_mia(member_losses, nonmember_losses,
                   member_ref_mean, nonmember_ref_mean,
                   member_ref_list, nonmember_ref_list,
                   member_quintiles, nonmember_quintiles,
                   n_strata=5):
    """Run MIA per difficulty stratum. Returns per-stratum and aggregate results."""
    results = {}
    for q in range(n_strata):
        m_mask = member_quintiles == q
        nm_mask = nonmember_quintiles == q
        n_m = m_mask.sum()
        n_nm = nm_mask.sum()

        if n_m < 5 or n_nm < 5:
            results[f'q{q+1}'] = {'best_auc': 0.5, 'loss_auc': 0.5,
                                   'calibrated_auc': 0.5, 'lira_auc': 0.5, 'n_members': int(n_m), 'n_nonmembers': int(n_nm)}
            continue

        m_losses = member_losses[m_mask]
        nm_losses = nonmember_losses[nm_mask]
        m_ref_mean = member_ref_mean[m_mask]
        nm_ref_mean = nonmember_ref_mean[nm_mask]
        m_ref_list = [r[m_mask] for r in member_ref_list]
        nm_ref_list = [r[nm_mask] for r in nonmember_ref_list]

        aucs = run_all_attacks(m_losses, nm_losses, m_ref_mean, nm_ref_mean,
                               m_ref_list, nm_ref_list)
        aucs['n_members'] = int(n_m)
        aucs['n_nonmembers'] = int(n_nm)
        results[f'q{q+1}'] = aucs

    # Compute summary metrics
    agg = run_all_attacks(member_losses, nonmember_losses,
                          member_ref_mean, nonmember_ref_mean,
                          member_ref_list, nonmember_ref_list)
    results['aggregate'] = agg

    best_aucs = [results[f'q{q+1}']['best_auc'] for q in range(n_strata)]
    results['wq_auc'] = float(best_aucs[-1])  # Worst quintile = hardest = last
    results['dg'] = float(best_aucs[-1] - best_aucs[0])  # Difficulty gap
    results['max_spread'] = float(max(best_aucs) - min(best_aucs))

    return results
