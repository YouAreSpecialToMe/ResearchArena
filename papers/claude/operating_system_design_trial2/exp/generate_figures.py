#!/usr/bin/env python3
"""Generate publication-quality figures and aggregated results.json."""

import sys
import os
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Consistent styling
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colorblind-friendly palette (Set2)
COLORS = {
    'GreedyByRank': '#66c2a5',
    'FIFO': '#fc8d62',
    'Random': '#8da0cb',
    'BKSDensity': '#e78ac3',
    'BKSThreshold': '#a6d854',
    'BKSDecay': '#ffd92f',
    'BKSNoDensity': '#b3b3b3',
    'BKSNoThreshold': '#e5c494',
    'BKSNoDecay': '#1b9e77',
    'equal-share': '#66c2a5',
    'greedy-global': '#fc8d62',
    'BKS-global': '#a6d854',
    'BKS-fair': '#e78ac3',
    'TPP': '#66c2a5',
    'Colloid': '#fc8d62',
    'ALTO': '#8da0cb',
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(FIGURES_DIR, 'supplementary'), exist_ok=True)


def load_results(exp_name):
    path = os.path.join(RESULTS_DIR, exp_name, 'results.json')
    with open(path) as f:
        return json.load(f)


def aggregate(data, group_keys, value_key):
    """Aggregate value_key by group_keys, returning mean and std."""
    groups = defaultdict(list)
    for row in data:
        key = tuple(row[k] for k in group_keys)
        groups[key].append(row[value_key])
    result = {}
    for key, values in groups.items():
        result[key] = {'mean': np.mean(values), 'std': np.std(values), 'n': len(values)}
    return result


# ============================
# FIGURE 1: Motivation
# ============================
def fig1_motivation(exp7_data):
    """Bar chart showing adversarial epoch: GreedyByRank vs BKS."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))

    # Use R=512, adv_frac=1.0 data
    frac1_data = [r for r in exp7_data if r['scenario'] == 'adv_fraction'
                  and r['adversarial_fraction'] == 1.0]
    agg = aggregate(frac1_data, ['scheduler'], 'total_benefit')

    scheds = ['GreedyByRank', 'BKSDensity', 'BKSThreshold']
    labels = ['Greedy-by-Rank', 'BKS-Density', 'BKS-Threshold']
    colors = [COLORS[s] for s in scheds]

    means = [agg.get((s,), {'mean': 0})['mean'] for s in scheds]
    stds = [agg.get((s,), {'std': 0})['std'] for s in scheds]

    bars = ax.bar(range(len(scheds)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(scheds)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Total Migration Benefit')
    ax.set_title('Adversarial Workload (R=512)')

    # Annotate ratio
    if means[0] > 0:
        ratio = means[1] / means[0]
        ax.annotate(f'{ratio:.0f}x', xy=(1, means[1]), fontsize=10,
                   ha='center', va='bottom', fontweight='bold')

    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_motivation.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_motivation.png'))
    plt.close()
    print("  fig1_motivation saved")


# ============================
# FIGURE 2: Main results
# ============================
def fig2_main_results(exp1_data):
    """Grouped bar chart: normalized latency across archetypes + BKS improvement vs bandwidth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    archetypes = ['mcf-like', 'lbm-like', 'xalancbmk-like', 'omnetpp-like',
                  'bwaves-like', 'cactuBSSN-like']
    scheds = ['GreedyByRank', 'FIFO', 'Random', 'BKSDensity', 'BKSThreshold']
    bw = 0.2

    # Filter to archetype traces at bw=0.2
    bw_data = [r for r in exp1_data if r['bandwidth_pct'] == bw
               and r['trace'] in archetypes]

    # Panel 1: Normalized latency per archetype
    x = np.arange(len(archetypes))
    width = 0.15
    for i, sched in enumerate(scheds):
        sched_data = [r for r in bw_data if r['scheduler'] == sched]
        agg = aggregate(sched_data, ['trace'], 'total_latency')
        means = [agg.get((a,), {'mean': 0})['mean'] for a in archetypes]
        stds = [agg.get((a,), {'std': 0})['std'] for a in archetypes]
        # Normalize by GreedyByRank
        greedy_agg = aggregate([r for r in bw_data if r['scheduler'] == 'GreedyByRank'],
                               ['trace'], 'total_latency')
        norms = []
        norm_stds = []
        for a_idx, a in enumerate(archetypes):
            gm = greedy_agg.get((a,), {'mean': 1})['mean']
            norms.append(means[a_idx] / gm if gm > 0 else 1.0)
            norm_stds.append(stds[a_idx] / gm if gm > 0 else 0.0)
        ax1.bar(x + (i - 2) * width, norms, width, yerr=norm_stds, capsize=2,
                color=COLORS[sched], label=sched.replace('BKS', 'BKS-'), linewidth=0.3, edgecolor='black')

    ax1.set_xticks(x)
    ax1.set_xticklabels([a.replace('-like', '') for a in archetypes], rotation=30, ha='right')
    ax1.set_ylabel('Normalized Latency\n(vs GreedyByRank)')
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5)
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_title('Latency at 20% Bandwidth')

    # Panel 2: BKS improvement vs bandwidth budget
    bw_values = [0.1, 0.2, 0.3, 0.5, 1.0]
    improvements = {'BKSDensity': [], 'BKSThreshold': []}
    imp_stds = {'BKSDensity': [], 'BKSThreshold': []}
    for bw_val in bw_values:
        for bks in ['BKSDensity', 'BKSThreshold']:
            bks_data = [r for r in exp1_data if r['scheduler'] == bks
                       and r['bandwidth_pct'] == bw_val and r['trace'] in archetypes]
            greedy_data = [r for r in exp1_data if r['scheduler'] == 'GreedyByRank'
                          and r['bandwidth_pct'] == bw_val and r['trace'] in archetypes]
            # Compute per-seed improvement
            imp_vals = []
            for seed_val in set(r['seed'] for r in bks_data):
                bks_lat = np.mean([r['total_latency'] for r in bks_data if r['seed'] == seed_val])
                gr_lat = np.mean([r['total_latency'] for r in greedy_data if r['seed'] == seed_val])
                if gr_lat > 0:
                    imp_vals.append((gr_lat - bks_lat) / gr_lat * 100)
            improvements[bks].append(np.mean(imp_vals) if imp_vals else 0)
            imp_stds[bks].append(np.std(imp_vals) if imp_vals else 0)

    for bks in ['BKSDensity', 'BKSThreshold']:
        label = bks.replace('BKS', 'BKS-')
        ax2.errorbar([f'{int(b*100)}%' for b in bw_values], improvements[bks],
                    yerr=imp_stds[bks], marker='o', capsize=3,
                    color=COLORS[bks], label=label)
    ax2.set_xlabel('Bandwidth Budget')
    ax2.set_ylabel('Improvement over\nGreedyByRank (%)')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.legend()
    ax2.set_title('BKS Improvement vs Bandwidth')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_main_results.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_main_results.png'))
    plt.close()
    print("  fig2_main_results saved")


# ============================
# FIGURE 3: Heterogeneity
# ============================
def fig3_heterogeneity(exp2_data):
    """Three-panel plot: performance gap for uniform-4KB, uniform-2MB, mixed."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)
    variants = ['uniform-4KB', 'uniform-2MB', 'mixed']
    archetypes = ['mcf-like', 'lbm-like', 'xalancbmk-like', 'omnetpp-like',
                  'bwaves-like', 'cactuBSSN-like']

    for vi, variant in enumerate(variants):
        ax = axes[vi]
        v_data = [r for r in exp2_data if r['page_size_variant'] == variant]

        improvements = []
        imp_stds = []
        for arch in archetypes:
            bks_vals = [r['total_latency'] for r in v_data
                       if r['scheduler'] == 'BKSThreshold' and r['trace'] == arch]
            gr_vals = [r['total_latency'] for r in v_data
                      if r['scheduler'] == 'GreedyByRank' and r['trace'] == arch]
            if gr_vals and bks_vals:
                # Pair by seed
                bks_by_seed = {r['seed']: r['total_latency'] for r in v_data
                              if r['scheduler'] == 'BKSThreshold' and r['trace'] == arch}
                gr_by_seed = {r['seed']: r['total_latency'] for r in v_data
                             if r['scheduler'] == 'GreedyByRank' and r['trace'] == arch}
                imps = []
                for s in bks_by_seed:
                    if s in gr_by_seed and gr_by_seed[s] > 0:
                        imps.append((gr_by_seed[s] - bks_by_seed[s]) / gr_by_seed[s] * 100)
                improvements.append(np.mean(imps) if imps else 0)
                imp_stds.append(np.std(imps) if imps else 0)
            else:
                improvements.append(0)
                imp_stds.append(0)

        colors = [COLORS['BKSThreshold']] * len(archetypes)
        x = np.arange(len(archetypes))
        ax.bar(x, improvements, yerr=imp_stds, capsize=3, color=colors,
               edgecolor='black', linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('-like', '') for a in archetypes], rotation=45, ha='right', fontsize=7)
        ax.set_title(variant)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        if vi == 0:
            ax.set_ylabel('BKS-Threshold Improvement\nover GreedyByRank (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_heterogeneity.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_heterogeneity.png'))
    plt.close()
    print("  fig3_heterogeneity saved")


# ============================
# FIGURE 4: Multi-tenant
# ============================
def fig4_multitenant(exp3_data):
    """Two subplots: total latency and Jain's fairness index."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    policies = ['equal-share', 'greedy-global', 'BKS-global', 'BKS-fair']
    tenant_counts = [2, 4, 8]

    x = np.arange(len(tenant_counts))
    width = 0.2

    for pi, policy in enumerate(policies):
        lats = []
        lat_stds = []
        jains = []
        jain_stds = []
        for nt in tenant_counts:
            vals = [r['system_latency'] for r in exp3_data
                   if r['num_tenants'] == nt and r['policy'] == policy]
            jvals = [r['jain_fairness'] for r in exp3_data
                    if r['num_tenants'] == nt and r['policy'] == policy]
            lats.append(np.mean(vals) if vals else 0)
            lat_stds.append(np.std(vals) if vals else 0)
            jains.append(np.mean(jvals) if jvals else 0)
            jain_stds.append(np.std(jvals) if jvals else 0)

        label = policy
        ax1.bar(x + (pi - 1.5) * width, np.array(lats) / 1e12, width, yerr=np.array(lat_stds) / 1e12,
                capsize=2, color=COLORS[policy], label=label, edgecolor='black', linewidth=0.3)
        ax2.bar(x + (pi - 1.5) * width, jains, width, yerr=jain_stds,
                capsize=2, color=COLORS[policy], label=label, edgecolor='black', linewidth=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{n}T' for n in tenant_counts])
    ax1.set_ylabel('System Latency (x10^12 ns)')
    ax1.set_title('Total System Latency')
    ax1.legend(fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{n}T' for n in tenant_counts])
    ax2.set_ylabel("Jain's Fairness Index")
    ax2.set_title('Fairness')
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_multitenant.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_multitenant.png'))
    plt.close()
    print("  fig4_multitenant saved")


# ============================
# FIGURE 5: Ablation
# ============================
def fig5_ablation(exp4_data):
    """Grouped bar chart: ablation variants relative to BKS-Full."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    archetypes = ['mcf-like', 'lbm-like', 'xalancbmk-like', 'omnetpp-like',
                  'bwaves-like', 'cactuBSSN-like']
    ablation_schedulers = ['BKSDecay', 'BKSNoDecay', 'BKSNoThreshold',
                           'BKSNoDensity', 'BKSDensity']
    labels = ['BKS-Full\n(Decay)', 'BKS-\nNoDecay', 'BKS-\nNoThreshold',
              'BKS-\nNoDensity', 'BKS-\nDensityOnly']
    bw = 0.2

    bw_data = [r for r in exp4_data if r['bandwidth_pct'] == bw]
    x = np.arange(len(archetypes))
    width = 0.15

    for si, sched in enumerate(ablation_schedulers):
        # Compute latency relative to BKSDecay (full)
        rel_lats = []
        rel_stds = []
        for arch in archetypes:
            full_vals = [r['total_latency'] for r in bw_data
                        if r['scheduler'] == 'BKSDecay' and r['trace'] == arch]
            sched_vals = [r['total_latency'] for r in bw_data
                         if r['scheduler'] == sched and r['trace'] == arch]
            if full_vals and sched_vals:
                full_mean = np.mean(full_vals)
                rel = np.mean(sched_vals) / full_mean if full_mean > 0 else 1.0
                rel_std = np.std(sched_vals) / full_mean if full_mean > 0 else 0.0
            else:
                rel, rel_std = 1.0, 0.0
            rel_lats.append(rel)
            rel_stds.append(rel_std)

        ax.bar(x + (si - 2) * width, rel_lats, width, yerr=rel_stds, capsize=2,
               color=COLORS.get(sched, '#cccccc'), label=labels[si],
               edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('-like', '') for a in archetypes], rotation=30, ha='right')
    ax.set_ylabel('Latency Relative to BKS-Full')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.set_title('Ablation Study (BW=20%)')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_ablation.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_ablation.png'))
    plt.close()
    print("  fig5_ablation saved")


# ============================
# FIGURE 6: Sensitivity
# ============================
def fig6_sensitivity(exp5_data):
    """2x2 grid: BKS improvement vs epoch_length, latency_ratio, fast_fraction, bandwidth."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    sweeps = [
        ('epoch_length', 'Epoch Length', 'Epoch Length (accesses)'),
        ('latency_ratio', 'Latency Ratio', 'Slow/Fast Latency Ratio'),
        ('fast_fraction', 'Fast Tier Fraction', 'Fast Tier Capacity Fraction'),
    ]

    for idx, (sweep_name, title, xlabel) in enumerate(sweeps):
        ax = axes[idx // 2][idx % 2]
        sweep_data = [r for r in exp5_data if r['sweep'] == sweep_name]
        if not sweep_data:
            continue

        param_values = sorted(set(r['param_value'] for r in sweep_data))

        for arch in ['mcf-like', 'lbm-like']:
            improvements = []
            imp_stds = []
            for pv in param_values:
                bks_by_seed = {}
                gr_by_seed = {}
                for r in sweep_data:
                    if r['trace'] == arch and r['param_value'] == pv:
                        if r['scheduler'] == 'BKSThreshold':
                            bks_by_seed[r['seed']] = r['total_latency']
                        elif r['scheduler'] == 'GreedyByRank':
                            gr_by_seed[r['seed']] = r['total_latency']
                imps = []
                for s in bks_by_seed:
                    if s in gr_by_seed and gr_by_seed[s] > 0:
                        imps.append((gr_by_seed[s] - bks_by_seed[s]) / gr_by_seed[s] * 100)
                improvements.append(np.mean(imps) if imps else 0)
                imp_stds.append(np.std(imps) if imps else 0)

            ax.errorbar(range(len(param_values)), improvements, yerr=imp_stds,
                       marker='o', capsize=3, label=arch.replace('-like', ''))

        ax.set_xticks(range(len(param_values)))
        if sweep_name == 'epoch_length':
            ax.set_xticklabels([f'{int(v/1000)}K' for v in param_values], fontsize=8)
        elif sweep_name == 'latency_ratio':
            ax.set_xticklabels([f'{v:.1f}x' for v in param_values], fontsize=8)
        else:
            ax.set_xticklabels([f'{v:.0%}' for v in param_values], fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('BKS Improvement (%)')
        ax.set_title(title)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=7)

    # Panel 4: Bandwidth (from exp1 data, loaded separately)
    ax = axes[1][1]
    ax.set_title('Bandwidth Budget')
    ax.set_xlabel('Bandwidth Budget (%)')
    ax.set_ylabel('BKS Improvement (%)')
    ax.text(0.5, 0.5, 'See Fig 2 panel 2', transform=ax.transAxes,
           ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_sensitivity.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_sensitivity.png'))
    plt.close()
    print("  fig6_sensitivity saved")


# ============================
# FIGURE 7: Composability
# ============================
def fig7_composability(exp6_data):
    """Grouped bar chart: latency for GreedyByRank vs BKSThreshold x 3 rankers."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    rankers = ['TPP', 'Colloid', 'ALTO']
    comp_schedulers = ['GreedyByRank', 'BKSThreshold']

    x = np.arange(len(rankers))
    width = 0.35

    for si, sched in enumerate(comp_schedulers):
        means = []
        stds_vals = []
        for ranker in rankers:
            vals = [r['total_latency'] for r in exp6_data
                   if r['ranker'] == ranker and r['scheduler'] == sched]
            means.append(np.mean(vals) if vals else 0)
            stds_vals.append(np.std(vals) if vals else 0)

        # Normalize by max for readability
        max_val = max(means) if means else 1
        norm_means = [m / max_val for m in means]
        norm_stds = [s / max_val for s in stds_vals]

        ax.bar(x + (si - 0.5) * width, norm_means, width, yerr=norm_stds, capsize=3,
               color=COLORS[sched], label=sched.replace('BKS', 'BKS-'),
               edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(rankers)
    ax.set_ylabel('Normalized Latency')
    ax.set_title('Composability: BKS with Different Rankers')
    ax.legend()
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5)

    # Add improvement annotations
    for ri, ranker in enumerate(rankers):
        gr_vals = [r['total_latency'] for r in exp6_data
                  if r['ranker'] == ranker and r['scheduler'] == 'GreedyByRank']
        bks_vals = [r['total_latency'] for r in exp6_data
                   if r['ranker'] == ranker and r['scheduler'] == 'BKSThreshold']
        if gr_vals and bks_vals:
            imp = (np.mean(gr_vals) - np.mean(bks_vals)) / np.mean(gr_vals) * 100
            ax.annotate(f'{imp:.1f}%', xy=(ri, 0.02), fontsize=8,
                       ha='center', va='bottom', color='green' if imp > 0 else 'red')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_composability.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_composability.png'))
    plt.close()
    print("  fig7_composability saved")


# ============================
# FIGURE 8: Competitive ratio
# ============================
def fig8_competitive_ratio(exp7_data):
    """Two subplots: CR vs adversarial fraction, CR vs R."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Panel 1: CR vs adversarial fraction
    adv_fracs = [0.1, 0.5, 1.0]
    for sched in ['GreedyByRank', 'BKSDensity', 'BKSThreshold']:
        means = []
        stds = []
        for af in adv_fracs:
            vals = [r['competitive_ratio'] for r in exp7_data
                   if r['scenario'] == 'adv_fraction' and r['adversarial_fraction'] == af
                   and r['scheduler'] == sched]
            means.append(np.mean(vals) if vals else 1)
            stds.append(np.std(vals) if vals else 0)
        label = sched.replace('BKS', 'BKS-')
        ax1.errorbar(adv_fracs, means, yerr=stds, marker='o', capsize=3,
                    color=COLORS[sched], label=label)

    ax1.set_xlabel('Adversarial Epoch Fraction')
    ax1.set_ylabel('Competitive Ratio\n(optimal/scheduler benefit)')
    ax1.set_title('CR vs Adversarial Fraction (R=512)')
    ax1.legend(fontsize=8)
    ax1.set_yscale('log')

    # Panel 2: CR vs R
    R_values = sorted(set(r['R'] for r in exp7_data if r['scenario'] == 'varying_R'))
    for sched in ['GreedyByRank', 'BKSDensity', 'BKSThreshold']:
        means = []
        stds = []
        for R in R_values:
            vals = [r['competitive_ratio'] for r in exp7_data
                   if r['scenario'] == 'varying_R' and r['R'] == R
                   and r['scheduler'] == sched]
            means.append(np.mean(vals) if vals else 1)
            stds.append(np.std(vals) if vals else 0)
        label = sched.replace('BKS', 'BKS-')
        ax2.errorbar(R_values, means, yerr=stds, marker='o', capsize=3,
                    color=COLORS[sched], label=label)

    # Theoretical bounds
    R_theory = np.array(R_values, dtype=float)
    ax2.plot(R_theory, R_theory, '--', color='gray', alpha=0.5, label='O(R) bound')
    ax2.plot(R_theory, np.log2(R_theory) + 1, ':', color='gray', alpha=0.5, label='O(log R) bound')

    ax2.set_xlabel('Page Size Ratio R')
    ax2.set_ylabel('Competitive Ratio')
    ax2.set_title('CR vs Page Size Ratio')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig8_competitive_ratio.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig8_competitive_ratio.png'))
    plt.close()
    print("  fig8_competitive_ratio saved")


# ============================
# Statistical analysis & results.json
# ============================
def compute_statistics_and_results(exp1, exp2, exp3, exp4, exp5, exp6, exp7):
    """Compute statistics, evaluate success criteria, and generate results.json."""
    from scipy import stats

    archetypes = ['mcf-like', 'lbm-like', 'xalancbmk-like', 'omnetpp-like',
                  'bwaves-like', 'cactuBSSN-like']

    results = {
        'experiment_parameters': {
            'total_accesses': 2_000_000,
            'epoch_length': 100_000,
            'seeds': [42, 123, 456, 789, 1024],
            'fast_latency_ns': 80,
            'slow_latency_ns': 200,
            'fast_fraction': 0.3,
            'default_bandwidth_fraction': 0.2,
            'page_sizes': {'small': 4096, 'large': 2097152},
            'page_size_ratio_R': 512,
            'deviations_from_plan': [
                'Used 2M accesses instead of 10M (CPU feasibility)',
                'Used 5 seeds instead of 3 (improved statistical power)',
                'Epoch length 100K as planned',
                'Page counts match plan (not scaled down)',
            ],
        },
        'experiments': {},
        'success_criteria': {},
    }

    # --- Exp 1: Single-tenant ---
    exp1_summary = {}
    bw_data_02 = [r for r in exp1 if r['bandwidth_pct'] == 0.2 and r['trace'] in archetypes]
    for arch in archetypes:
        bks_by_seed = {r['seed']: r['total_latency'] for r in bw_data_02
                      if r['scheduler'] == 'BKSThreshold' and r['trace'] == arch}
        gr_by_seed = {r['seed']: r['total_latency'] for r in bw_data_02
                     if r['scheduler'] == 'GreedyByRank' and r['trace'] == arch}
        seeds = sorted(set(bks_by_seed.keys()) & set(gr_by_seed.keys()))
        if seeds:
            bks_lats = [bks_by_seed[s] for s in seeds]
            gr_lats = [gr_by_seed[s] for s in seeds]
            improvements = [(g - b) / g * 100 for g, b in zip(gr_lats, bks_lats)]
            t_stat, p_val = stats.ttest_rel(gr_lats, bks_lats) if len(seeds) >= 2 else (0, 1)
            exp1_summary[arch] = {
                'improvement_pct': {'mean': np.mean(improvements), 'std': np.std(improvements)},
                'bks_latency': {'mean': np.mean(bks_lats), 'std': np.std(bks_lats)},
                'greedy_latency': {'mean': np.mean(gr_lats), 'std': np.std(gr_lats)},
                'paired_ttest_pvalue': float(p_val),
                'n_seeds': len(seeds),
            }
    results['experiments']['exp1_single_tenant'] = exp1_summary

    # --- Exp 2: Heterogeneity ---
    exp2_summary = {}
    for variant in ['uniform-4KB', 'uniform-2MB', 'mixed']:
        v_results = {}
        for arch in archetypes:
            bks_by_seed = {r['seed']: r['total_latency'] for r in exp2
                          if r['scheduler'] == 'BKSThreshold' and r['trace'] == arch
                          and r['page_size_variant'] == variant}
            gr_by_seed = {r['seed']: r['total_latency'] for r in exp2
                         if r['scheduler'] == 'GreedyByRank' and r['trace'] == arch
                         and r['page_size_variant'] == variant}
            seeds = sorted(set(bks_by_seed.keys()) & set(gr_by_seed.keys()))
            if seeds:
                improvements = [(gr_by_seed[s] - bks_by_seed[s]) / gr_by_seed[s] * 100 for s in seeds]
                v_results[arch] = {'improvement_pct': {'mean': np.mean(improvements), 'std': np.std(improvements)}}
        exp2_summary[variant] = v_results
    results['experiments']['exp2_heterogeneity'] = exp2_summary

    # --- Exp 3: Multi-tenant ---
    exp3_summary = {}
    for nt in [2, 4, 8]:
        nt_results = {}
        for policy in ['equal-share', 'greedy-global', 'BKS-global', 'BKS-fair']:
            vals = [r for r in exp3 if r['num_tenants'] == nt and r['policy'] == policy]
            if vals:
                nt_results[policy] = {
                    'system_latency': {'mean': np.mean([r['system_latency'] for r in vals]),
                                       'std': np.std([r['system_latency'] for r in vals])},
                    'jain_fairness': {'mean': np.mean([r['jain_fairness'] for r in vals]),
                                      'std': np.std([r['jain_fairness'] for r in vals])},
                }
        exp3_summary[f'{nt}_tenants'] = nt_results
    results['experiments']['exp3_multitenant'] = exp3_summary

    # --- Exp 4: Ablation ---
    exp4_summary = {}
    for sched in ['BKSDecay', 'BKSNoDecay', 'BKSNoThreshold', 'BKSNoDensity', 'BKSDensity']:
        vals = [r['total_latency'] for r in exp4 if r['scheduler'] == sched and r['bandwidth_pct'] == 0.2]
        if vals:
            exp4_summary[sched] = {'latency': {'mean': np.mean(vals), 'std': np.std(vals)}}
    results['experiments']['exp4_ablation'] = exp4_summary

    # --- Exp 5: Sensitivity ---
    exp5_summary = {}
    for sweep in ['epoch_length', 'latency_ratio', 'fast_fraction']:
        sweep_data = [r for r in exp5 if r['sweep'] == sweep]
        param_values = sorted(set(r['param_value'] for r in sweep_data))
        sweep_results = {}
        for pv in param_values:
            bks_vals = [r['total_latency'] for r in sweep_data
                       if r['param_value'] == pv and r['scheduler'] == 'BKSThreshold']
            gr_vals = [r['total_latency'] for r in sweep_data
                      if r['param_value'] == pv and r['scheduler'] == 'GreedyByRank']
            if bks_vals and gr_vals:
                imp = (np.mean(gr_vals) - np.mean(bks_vals)) / np.mean(gr_vals) * 100
                sweep_results[str(pv)] = {'improvement_pct': imp}
        exp5_summary[sweep] = sweep_results
    results['experiments']['exp5_sensitivity'] = exp5_summary

    # --- Exp 6: Composability ---
    exp6_summary = {}
    for ranker in ['TPP', 'Colloid', 'ALTO']:
        bks_by_seed = {}
        gr_by_seed = {}
        for r in exp6:
            if r['ranker'] == ranker:
                if r['scheduler'] == 'BKSThreshold':
                    bks_by_seed.setdefault(r['seed'], []).append(r['total_latency'])
                elif r['scheduler'] == 'GreedyByRank':
                    gr_by_seed.setdefault(r['seed'], []).append(r['total_latency'])
        seeds = sorted(set(bks_by_seed.keys()) & set(gr_by_seed.keys()))
        if seeds:
            improvements = []
            for s in seeds:
                bks_mean = np.mean(bks_by_seed[s])
                gr_mean = np.mean(gr_by_seed[s])
                if gr_mean > 0:
                    improvements.append((gr_mean - bks_mean) / gr_mean * 100)
            bks_lats = [np.mean(bks_by_seed[s]) for s in seeds]
            gr_lats = [np.mean(gr_by_seed[s]) for s in seeds]
            t_stat, p_val = stats.ttest_rel(gr_lats, bks_lats) if len(seeds) >= 2 else (0, 1)
            exp6_summary[ranker] = {
                'improvement_pct': {'mean': np.mean(improvements), 'std': np.std(improvements)},
                'paired_ttest_pvalue': float(p_val),
            }
    results['experiments']['exp6_composability'] = exp6_summary

    # --- Exp 7: Adversarial ---
    exp7_summary = {}
    # CR vs adversarial fraction
    for af in [0.1, 0.5, 1.0]:
        af_results = {}
        for sched in ['GreedyByRank', 'BKSDensity', 'BKSThreshold']:
            vals = [r['competitive_ratio'] for r in exp7
                   if r['scenario'] == 'adv_fraction' and r['adversarial_fraction'] == af
                   and r['scheduler'] == sched]
            if vals:
                af_results[sched] = {'competitive_ratio': {'mean': np.mean(vals), 'std': np.std(vals)}}
        exp7_summary[f'adv_frac_{af}'] = af_results

    # CR vs R
    R_values = sorted(set(r['R'] for r in exp7 if r['scenario'] == 'varying_R'))
    for R in R_values:
        r_results = {}
        for sched in ['GreedyByRank', 'BKSDensity', 'BKSThreshold']:
            vals = [r['competitive_ratio'] for r in exp7
                   if r['scenario'] == 'varying_R' and r['R'] == R
                   and r['scheduler'] == sched]
            if vals:
                r_results[sched] = {'competitive_ratio': {'mean': np.mean(vals), 'std': np.std(vals)}}
        exp7_summary[f'R_{R}'] = r_results
    results['experiments']['exp7_adversarial'] = exp7_summary

    # ============================
    # SUCCESS CRITERIA EVALUATION
    # ============================

    # Criterion 1: Theory validation
    # GreedyByRank CR should approach R on adversarial inputs, BKS should stay O(log R)
    # Use varying_R data at R=512 for the strongest adversarial signal
    gr_crs = {}
    bks_crs = {}
    for R in R_values:
        gr_data = exp7_summary.get(f'R_{R}', {}).get('GreedyByRank', {}).get('competitive_ratio', {})
        bks_data = exp7_summary.get(f'R_{R}', {}).get('BKSThreshold', {}).get('competitive_ratio', {})
        if gr_data:
            gr_crs[R] = gr_data.get('mean', 0)
        if bks_data:
            bks_crs[R] = bks_data.get('mean', 0)

    gr_cr_R512 = gr_crs.get(512, exp7_summary.get('adv_frac_1.0', {}).get('GreedyByRank', {}).get('competitive_ratio', {}).get('mean', 0))
    bks_cr_R512 = bks_crs.get(512, exp7_summary.get('adv_frac_1.0', {}).get('BKSThreshold', {}).get('competitive_ratio', {}).get('mean', 0))

    theory_pass = gr_cr_R512 > 10 and bks_cr_R512 < 2 * np.log2(512)
    results['success_criteria']['criterion_1_theory'] = {
        'description': 'Omega(R) lower bound for greedy, O(log R) upper bound for BKS',
        'greedy_CR_at_R512': gr_cr_R512,
        'BKS_CR_at_R512': bks_cr_R512,
        'greedy_CR_vs_R': gr_crs,
        'BKS_CR_vs_R': bks_crs,
        'target_greedy_CR': '> 10 (empirical lower bound demonstrating growth with R)',
        'target_BKS_CR': f'< 2*log2(R) = {2*np.log2(512):.1f}',
        'pass': theory_pass,
    }

    # Criterion 2: Practical improvement >= 3% on at least 4/6 archetypes
    improved_count = 0
    for arch in archetypes:
        if arch in exp1_summary:
            if exp1_summary[arch]['improvement_pct']['mean'] >= 3.0:
                improved_count += 1
    practice_pass = improved_count >= 4
    results['success_criteria']['criterion_2_practice'] = {
        'description': 'BKS improves latency by >= 3% on at least 4/6 archetypes at BW=20%',
        'archetypes_improved_3pct': improved_count,
        'per_archetype': {a: exp1_summary.get(a, {}).get('improvement_pct', {}) for a in archetypes},
        'pass': practice_pass,
    }

    # Criterion 3: Composability with at least 2/3 ranking policies
    comp_improved = 0
    for ranker in ['TPP', 'Colloid', 'ALTO']:
        if ranker in exp6_summary:
            if exp6_summary[ranker]['improvement_pct']['mean'] > 0:
                comp_improved += 1
    comp_pass = comp_improved >= 2
    results['success_criteria']['criterion_3_composability'] = {
        'description': 'BKS improves at least 2/3 ranking policies',
        'rankers_improved': comp_improved,
        'per_ranker': exp6_summary,
        'pass': comp_pass,
    }

    # Criterion 4: BKS-fair has better Jain fairness than equal-share
    fairness_pass = True
    fairness_details = {}
    for nt in [2, 4, 8]:
        key = f'{nt}_tenants'
        if key in exp3_summary:
            bks_fair_jain = exp3_summary[key].get('BKS-fair', {}).get('jain_fairness', {}).get('mean', 0)
            equal_jain = exp3_summary[key].get('equal-share', {}).get('jain_fairness', {}).get('mean', 0)
            fairness_details[key] = {
                'BKS-fair_jain': bks_fair_jain,
                'equal-share_jain': equal_jain,
                'BKS_better': bks_fair_jain >= equal_jain,
            }
            if bks_fair_jain < equal_jain:
                fairness_pass = False
    results['success_criteria']['criterion_4_fairness'] = {
        'description': 'BKS-fair has better Jain fairness than equal-share',
        'details': fairness_details,
        'pass': fairness_pass,
    }

    # Overall
    results['success_criteria']['overall'] = {
        'criteria_passed': sum([theory_pass, practice_pass, comp_pass, fairness_pass]),
        'total_criteria': 4,
    }

    return results


if __name__ == '__main__':
    print("Loading experiment results...")
    try:
        exp1 = load_results('exp1_single_tenant')
        exp2 = load_results('exp2_heterogeneity')
        exp3 = load_results('exp3_multitenant')
        exp4 = load_results('exp4_ablation')
        exp5 = load_results('exp5_sensitivity')
        exp6 = load_results('exp6_composability')
        exp7 = load_results('exp7_adversarial')
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run run_all_experiments.py first!")
        sys.exit(1)

    print("\nGenerating figures...")
    fig1_motivation(exp7)
    fig2_main_results(exp1)
    fig3_heterogeneity(exp2)
    fig4_multitenant(exp3)
    fig5_ablation(exp4)
    fig6_sensitivity(exp5)
    fig7_composability(exp6)
    fig8_competitive_ratio(exp7)

    print("\nComputing statistics and generating results.json...")
    results = compute_statistics_and_results(exp1, exp2, exp3, exp4, exp5, exp6, exp7)

    results_path = os.path.join(BASE_DIR, 'results.json')

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {results_path}")

    # Print summary
    sc = results['success_criteria']
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA SUMMARY")
    print("=" * 60)
    for key, val in sc.items():
        if key == 'overall':
            print(f"\n  OVERALL: {val['criteria_passed']}/{val['total_criteria']} criteria passed")
        else:
            status = 'PASS' if val.get('pass', False) else 'FAIL'
            print(f"  [{status}] {val.get('description', key)}")

    # Save CSV summary
    csv_path = os.path.join(RESULTS_DIR, 'success_criteria_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('criterion,description,target,achieved,pass\n')
        for key, val in sc.items():
            if key != 'overall':
                desc = val.get('description', '')
                passed = val.get('pass', False)
                f.write(f'{key},"{desc}","","",{passed}\n')
    print(f"\nCSV summary saved to {csv_path}")
