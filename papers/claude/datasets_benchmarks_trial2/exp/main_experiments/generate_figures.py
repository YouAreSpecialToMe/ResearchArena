#!/usr/bin/env python3
"""
Generate all publication-quality figures for ConsistBench paper.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})
colors = sns.color_palette('colorblind', 10)

# Load data
with open(os.path.join(RESULTS_DIR, 'cross_format_results.json')) as f:
    fmt_results = json.load(f)
with open(os.path.join(RESULTS_DIR, 'cross_phrasing_results.json')) as f:
    phr_results = json.load(f)
with open(os.path.join(RESULTS_DIR, 'domain_analysis.json')) as f:
    domain_analysis = json.load(f)
with open(os.path.join(RESULTS_DIR, 'ablation_format_pairs.json')) as f:
    abl_format = json.load(f)
with open(os.path.join(RESULTS_DIR, 'ablation_size_scaling.json')) as f:
    abl_scale = json.load(f)

# Sort models by size
model_order = sorted(fmt_results.keys(), key=lambda m: fmt_results[m]['model_size_b'])
model_colors = {m: colors[i] for i, m in enumerate(model_order)}
model_short = {m: m.replace('Qwen2.5-', 'Q').replace('Llama-3.1-', 'L').replace('Mistral-', 'M').replace('Phi-3.5-', 'P') for m in model_order}

formats = ['mcq', 'open', 'yesno', 'truefalse', 'fitb']
format_labels = {'mcq': 'MCQ', 'open': 'Open', 'yesno': 'Yes/No', 'truefalse': 'T/F', 'fitb': 'FITB'}
ptypes = ['lexical', 'syntactic', 'voice', 'formality', 'negation', 'elaborative']

# ============================================================
# Figure 1: Main Results - Format accuracy + CFA bar chart
# ============================================================
print("Figure 1: Main results...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Accuracy per format
x = np.arange(len(formats))
width = 0.15
for i, model in enumerate(model_order):
    acc = [fmt_results[model]['accuracy_per_format'].get(fmt, 0) for fmt in formats]
    axes[0].bar(x + i * width, acc, width, label=model, color=model_colors[model])
axes[0].set_xlabel('Answer Format')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('(a) Accuracy by Format')
axes[0].set_xticks(x + width * 2)
axes[0].set_xticklabels([format_labels[f] for f in formats])
axes[0].legend(fontsize=8, loc='upper right')
axes[0].set_ylim(0, 1.0)

# Right: CFA with CI
models_cfa = [(model, fmt_results[model]['cfa_mean'],
               fmt_results[model]['cfa_ci_lower'], fmt_results[model]['cfa_ci_upper'])
              for model in model_order]
y_pos = np.arange(len(models_cfa))
cfa_vals = [m[1] for m in models_cfa]
cfa_errs = [[m[1]-m[2] for m in models_cfa], [m[3]-m[1] for m in models_cfa]]
bars = axes[1].barh(y_pos, cfa_vals, xerr=cfa_errs, color=[model_colors[m[0]] for m in models_cfa],
                     capsize=3, edgecolor='black', linewidth=0.5)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels([m[0] for m in models_cfa])
axes[1].set_xlabel('Cross-Format Agreement (CFA)')
axes[1].set_title('(b) Cross-Format Agreement')
axes[1].axvline(x=0.90, color='red', linestyle='--', linewidth=1, label='90% threshold')
axes[1].axvline(x=0.27, color='gray', linestyle=':', linewidth=1, label='Random baseline')
axes[1].legend(fontsize=8)
axes[1].set_xlim(0, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure_main_results.png'))
plt.savefig(os.path.join(FIGURES_DIR, 'figure_main_results.pdf'))
plt.close()

# ============================================================
# Figure 2: Cross-Format Agreement Heatmap
# ============================================================
print("Figure 2: Format heatmap...")
n_models = len(model_order)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, model in enumerate(model_order):
    ax = axes[idx]
    pa = abl_format[model]['pairwise_agreement']

    # Build 5x5 matrix
    matrix = np.ones((5, 5))
    for i, f1 in enumerate(formats):
        for j, f2 in enumerate(formats):
            if i == j:
                continue
            key1 = f"{f1}-{f2}"
            key2 = f"{f2}-{f1}"
            val = pa.get(key1, pa.get(key2, 0.5))
            matrix[i][j] = val
            matrix[j][i] = val

    sns.heatmap(matrix, ax=ax, vmin=0.3, vmax=1.0, annot=True, fmt='.2f',
                xticklabels=[format_labels[f] for f in formats],
                yticklabels=[format_labels[f] for f in formats],
                cmap='RdYlGn', square=True, cbar_kws={'shrink': 0.8})
    ax.set_title(f'{model} ({fmt_results[model]["model_size_b"]}B)')

# Hide extra subplot
if n_models < 6:
    axes[-1].axis('off')

plt.suptitle('Cross-Format Pairwise Agreement', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure_format_heatmap.png'))
plt.savefig(os.path.join(FIGURES_DIR, 'figure_format_heatmap.pdf'))
plt.close()

# ============================================================
# Figure 3: Paraphrase Fragility Profile
# ============================================================
print("Figure 3: Paraphrase fragility...")
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(ptypes))
width = 0.15

for i, model in enumerate(model_order):
    pfi_vals = [phr_results[model]['pfi_per_type'].get(pt, 0) for pt in ptypes]
    ci_vals = phr_results[model].get('cpa_ci_per_type', {})
    errs = []
    for pt in ptypes:
        if pt in ci_vals:
            ci = ci_vals[pt]
            err = (ci['upper'] - ci['lower']) / 2
        else:
            err = 0
        errs.append(err)
    ax.bar(x + i * width, pfi_vals, width, label=model, color=model_colors[model],
           yerr=errs, capsize=2)

ax.set_xlabel('Paraphrase Type')
ax.set_ylabel('Paraphrase Fragility Index (PFI)')
ax.set_title('Paraphrase Fragility by Type and Model')
ax.set_xticks(x + width * 2)
ax.set_xticklabels([pt.capitalize() for pt in ptypes], rotation=30, ha='right')
ax.axhline(y=0.95, color='gray', linestyle=':', linewidth=1, label='Random baseline')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, max(0.6, max(phr_results[model_order[0]]['pfi_per_type'].values()) + 0.1))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure_paraphrase_fragility.png'))
plt.savefig(os.path.join(FIGURES_DIR, 'figure_paraphrase_fragility.pdf'))
plt.close()

# ============================================================
# Figure 4: Domain Consistency
# ============================================================
print("Figure 4: Domain consistency...")
domains = ['science', 'history', 'math', 'commonsense', 'world_knowledge', 'logic']
domain_labels = ['Science', 'History', 'Math', 'Common\nSense', 'World\nKnow.', 'Logic']

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(domains))
width = 0.15

for i, model in enumerate(model_order):
    dcs = domain_analysis.get(model, {}).get('dcs', {})
    vals = [dcs.get(d, 0) for d in domains]
    ax.bar(x + i * width, vals, width, label=model, color=model_colors[model])

ax.set_xlabel('Knowledge Domain')
ax.set_ylabel('Domain Consistency Score (DCS)')
ax.set_title('Consistency by Knowledge Domain')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(domain_labels)
ax.legend(fontsize=8)
ax.set_ylim(0.3, 0.8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure_domain_consistency.png'))
plt.savefig(os.path.join(FIGURES_DIR, 'figure_domain_consistency.pdf'))
plt.close()

# ============================================================
# Figure 5: Consistency vs Accuracy Scatter
# ============================================================
print("Figure 5: Consistency vs accuracy...")
fig, ax = plt.subplots(figsize=(8, 6))

family_markers = {'Phi': 'o', 'Mistral': 's', 'Qwen': '^', 'Llama': 'D'}

for model in model_order:
    r = fmt_results[model]
    size = r['model_size_b']
    family = r['model_family']
    marker = family_markers.get(family, 'o')
    markersize = 50 + size * 8

    ax.scatter(r['overall_accuracy'], r['cfa_mean'], s=markersize,
               marker=marker, color=model_colors[model], edgecolors='black',
               linewidth=1, zorder=5)
    ax.annotate(model, (r['overall_accuracy'], r['cfa_mean']),
                textcoords="offset points", xytext=(5, 5), fontsize=8)

# Diagonal line (CFA = accuracy)
ax.plot([0.3, 0.7], [0.3, 0.7], 'k--', alpha=0.3, label='CFA = Accuracy')
# Random baseline
ax.axhline(y=0.27, color='gray', linestyle=':', alpha=0.5, label='Random CFA')
# 90% threshold
ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='90% CFA threshold')

ax.set_xlabel('Overall Accuracy')
ax.set_ylabel('Cross-Format Agreement (CFA)')
ax.set_title('Consistency vs. Accuracy')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure_consistency_vs_accuracy.png'))
plt.savefig(os.path.join(FIGURES_DIR, 'figure_consistency_vs_accuracy.pdf'))
plt.close()

# ============================================================
# Figure 6: Format-Conditional Accuracy Gap
# ============================================================
print("Figure 6: Format accuracy gap...")
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_order))
width = 0.15
bottom_vals = [0] * len(model_order)

for j, fmt in enumerate(formats):
    vals = [fmt_results[model]['accuracy_per_format'].get(fmt, 0) for model in model_order]
    ax.bar(x, vals, width, bottom=None, label=format_labels[fmt], color=colors[j], alpha=0.8)
    # Offset bars side by side
    pass

# Actually make grouped bars
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(model_order))
width = 0.15

for j, fmt in enumerate(formats):
    vals = [fmt_results[model]['accuracy_per_format'].get(fmt, 0) for model in model_order]
    ax.bar(x + j * width, vals, width, label=format_labels[fmt], color=colors[j])

ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Format and Model (Format-Conditional Accuracy Gap)')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(model_order, rotation=20, ha='right')
ax.legend()
ax.set_ylim(0, 1.0)

# Add FCAG annotations
for i, model in enumerate(model_order):
    fcag = fmt_results[model]['fcag']
    best_acc = max(fmt_results[model]['accuracy_per_format'].values())
    ax.annotate(f'Gap: {fcag:.2f}', (i + width * 2, best_acc + 0.02), fontsize=8, ha='center')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure_format_accuracy_gap.png'))
plt.savefig(os.path.join(FIGURES_DIR, 'figure_format_accuracy_gap.pdf'))
plt.close()

# ============================================================
# Figure 7: Size Scaling
# ============================================================
print("Figure 7: Size scaling...")
fig, ax = plt.subplots(figsize=(8, 6))

sizes = [fmt_results[m]['model_size_b'] for m in model_order]
accs = [fmt_results[m]['overall_accuracy'] for m in model_order]
cfas = [fmt_results[m]['cfa_mean'] for m in model_order]

ax.plot(sizes, accs, 'o-', color=colors[0], label='Accuracy', markersize=8, linewidth=2)
ax.plot(sizes, cfas, 's-', color=colors[1], label='CFA', markersize=8, linewidth=2)

# Annotate each point
for i, model in enumerate(model_order):
    ax.annotate(model, (sizes[i], accs[i]), textcoords="offset points",
                xytext=(0, 8), fontsize=7, ha='center')

ax.set_xlabel('Model Size (B parameters)')
ax.set_ylabel('Score')
ax.set_title('Accuracy and Consistency vs. Model Size')
ax.set_xscale('log')
ax.legend()
ax.set_ylim(0.3, 0.8)

# Highlight intra-family connections
# Qwen family
qwen_sizes = [fmt_results[m]['model_size_b'] for m in model_order if fmt_results[m]['model_family'] == 'Qwen']
qwen_cfas = [fmt_results[m]['cfa_mean'] for m in model_order if fmt_results[m]['model_family'] == 'Qwen']
if len(qwen_sizes) >= 2:
    ax.plot(qwen_sizes, qwen_cfas, '--', color=colors[1], alpha=0.3, linewidth=3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure_size_scaling.png'))
plt.savefig(os.path.join(FIGURES_DIR, 'figure_size_scaling.pdf'))
plt.close()

# ============================================================
# Figure 8: CAR comparison
# ============================================================
print("Figure 8: CAR comparison...")
fig, ax = plt.subplots(figsize=(8, 5))

cars = [(model, fmt_results[model]['car']) for model in model_order]
y_pos = np.arange(len(cars))
bar_colors = [model_colors[m[0]] for m in cars]

ax.barh(y_pos, [c[1] for c in cars], color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([c[0] for c in cars])
ax.set_xlabel('Consistency-Accuracy Ratio (CAR)')
ax.set_title('CAR: How Consistent is Each Model Relative to Its Accuracy?')
ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, label='CAR = 1.0')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure_car_comparison.png'))
plt.savefig(os.path.join(FIGURES_DIR, 'figure_car_comparison.pdf'))
plt.close()

print(f"\nAll figures saved to {FIGURES_DIR}/")
print("Figures generated:")
for f in sorted(os.listdir(FIGURES_DIR)):
    print(f"  {f}")
