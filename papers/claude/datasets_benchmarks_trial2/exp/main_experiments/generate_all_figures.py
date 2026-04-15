#!/usr/bin/env python3
"""Generate all publication-quality figures for ConsistBench."""
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
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})
PALETTE = sns.color_palette('colorblind', 10)

# Load data
with open(os.path.join(BASE_DIR, 'results.json')) as f:
    results = json.load(f)

models = results['models']
model_names = list(models.keys())
n_models = len(model_names)

FORMATS = ['mcq', 'open', 'yesno', 'truefalse', 'fitb']
FORMAT_LABELS = {'mcq': 'MCQ', 'open': 'Open', 'yesno': 'Yes/No', 'truefalse': 'T/F', 'fitb': 'FITB'}

# ============================================================
# Figure 1: Main Results Table (as figure)
# ============================================================
print("Generating Figure 1: Main Results Table...")
fig, ax = plt.subplots(figsize=(12, 3 + n_models * 0.5))
ax.axis('off')

headers = ['Model', 'Size', 'Family', 'Accuracy', 'MCQ', 'Open', 'Y/N', 'T/F', 'FITB', 'CFA', 'CAR', 'FCAG']
rows = []
for mn in model_names:
    m = models[mn]
    acc = m['accuracy']
    row = [
        mn, f"{m['size_b']}B", m['family'],
        f"{acc['mean']:.3f}",
        f"{acc['per_format'].get('mcq', 0):.3f}",
        f"{acc['per_format'].get('open', 0):.3f}",
        f"{acc['per_format'].get('yesno', 0):.3f}",
        f"{acc['per_format'].get('truefalse', 0):.3f}",
        f"{acc['per_format'].get('fitb', 0):.3f}",
        f"{m['cfa']['mean']:.3f}",
        f"{m['car']:.3f}",
        f"{m['fcag']:.3f}",
    ]
    rows.append(row)

table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Bold best values
for col_idx in [3, 9, 10]:  # Accuracy, CFA, CAR columns
    vals = [float(rows[i][col_idx]) for i in range(len(rows))]
    if col_idx == 10:  # CAR: closest to 1.0 is best
        best_idx = min(range(len(vals)), key=lambda i: abs(vals[i] - 1.0))
    else:
        best_idx = np.argmax(vals)
    table[best_idx + 1, col_idx].set_text_props(fontweight='bold')

# Header styling
for j in range(len(headers)):
    table[0, j].set_facecolor('#e6e6e6')
    table[0, j].set_text_props(fontweight='bold')

plt.title('ConsistBench: Main Results', fontsize=14, fontweight='bold', pad=20)
for ext in ['pdf', 'png']:
    plt.savefig(os.path.join(FIGURES_DIR, f'figure_main_results.{ext}'))
plt.close()

# ============================================================
# Figure 2: Cross-Format Agreement Heatmap
# ============================================================
print("Generating Figure 2: Format Heatmaps...")
n_cols = min(3, n_models)
n_rows = (n_models + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
if n_models == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

for idx, mn in enumerate(model_names):
    r, c = idx // n_cols, idx % n_cols
    ax = axes[r, c]

    m = models[mn]
    pair_agree = m.get('format_pair_agreement', {})

    matrix = np.zeros((5, 5))
    np.fill_diagonal(matrix, 1.0)
    for fi in range(5):
        for fj in range(fi + 1, 5):
            key = f"{FORMATS[fi]}-{FORMATS[fj]}"
            val = pair_agree.get(key, 0)
            matrix[fi, fj] = val
            matrix[fj, fi] = val

    sns.heatmap(matrix, ax=ax, vmin=0.3, vmax=1.0, cmap='YlOrRd',
                annot=True, fmt='.2f', square=True,
                xticklabels=[FORMAT_LABELS[f] for f in FORMATS],
                yticklabels=[FORMAT_LABELS[f] for f in FORMATS],
                cbar=idx == 0)
    ax.set_title(f'{mn}', fontsize=11)

# Hide empty subplots
for idx in range(n_models, n_rows * n_cols):
    r, c = idx // n_cols, idx % n_cols
    axes[r, c].set_visible(False)

plt.suptitle('Cross-Format Pairwise Agreement', fontsize=14, fontweight='bold')
plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(os.path.join(FIGURES_DIR, f'figure_format_heatmap.{ext}'))
plt.close()

# ============================================================
# Figure 3: Paraphrase Fragility Profile
# ============================================================
print("Generating Figure 3: Paraphrase Fragility...")
PTYPES = ['lexical', 'syntactic', 'voice', 'formality', 'negation', 'elaborative']
PTYPE_LABELS = {'lexical': 'Lexical', 'syntactic': 'Syntactic', 'voice': 'Voice',
                'formality': 'Formality', 'negation': 'Negation', 'elaborative': 'Elaborative'}

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(PTYPES))
width = 0.8 / n_models

for i, mn in enumerate(model_names):
    m = models[mn]
    pfi_vals = [m.get('pfi_per_type', {}).get(pt, 0) for pt in PTYPES]
    offset = (i - n_models / 2 + 0.5) * width
    ax.bar(x + offset, pfi_vals, width, label=mn, color=PALETTE[i], alpha=0.85)

ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
ax.set_xlabel('Paraphrase Type')
ax.set_ylabel('Paraphrase Fragility Index (PFI)')
ax.set_title('Paraphrase Fragility by Type and Model')
ax.set_xticks(x)
ax.set_xticklabels([PTYPE_LABELS[pt] for pt in PTYPES])
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.set_ylim(0, 1.0)

plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(os.path.join(FIGURES_DIR, f'figure_paraphrase_fragility.{ext}'))
plt.close()

# ============================================================
# Figure 4: Domain Consistency
# ============================================================
print("Generating Figure 4: Domain Consistency...")
with open(os.path.join(RESULTS_DIR, 'domain_analysis.json')) as f:
    domain_data = json.load(f)

domains = sorted(set(d for mn in domain_data for d in domain_data[mn].get('dcs', {})))
domain_labels = {d: d.replace('_', ' ').title() for d in domains}

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(domains))
width = 0.8 / n_models

for i, mn in enumerate(model_names):
    dcs = domain_data.get(mn, {}).get('dcs', {})
    vals = [dcs.get(d, 0) for d in domains]
    offset = (i - n_models / 2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=mn, color=PALETTE[i], alpha=0.85)

ax.set_xlabel('Knowledge Domain')
ax.set_ylabel('Domain Consistency Score (DCS)')
ax.set_title('Consistency Across Knowledge Domains')
ax.set_xticks(x)
ax.set_xticklabels([domain_labels.get(d, d) for d in domains], rotation=15, ha='right')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(os.path.join(FIGURES_DIR, f'figure_domain_consistency.{ext}'))
plt.close()

# ============================================================
# Figure 5: Consistency vs Accuracy Scatter
# ============================================================
print("Generating Figure 5: Consistency vs Accuracy...")
fig, ax = plt.subplots(figsize=(8, 6))

families = list(set(models[mn]['family'] for mn in model_names))
family_colors = {f: PALETTE[i] for i, f in enumerate(families)}

for mn in model_names:
    m = models[mn]
    acc = m['accuracy']['mean']
    cfa = m['cfa']['mean']
    size = m['size_b']
    family = m['family']

    marker_size = max(40, min(200, size * 3))
    ax.scatter(acc, cfa, s=marker_size, c=[family_colors[family]],
               edgecolors='black', linewidth=0.5, zorder=3)
    ax.annotate(mn, (acc, cfa), fontsize=7, ha='center', va='bottom',
                xytext=(0, 8), textcoords='offset points')

# Diagonal line (CFA = accuracy)
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.3, label='CFA = Accuracy')
ax.axhline(y=0.27, color='red', linestyle=':', alpha=0.4, label='Random baseline')

# Legend for families
for fam, col in family_colors.items():
    ax.scatter([], [], c=[col], s=60, edgecolors='black', linewidth=0.5, label=fam)
ax.legend(loc='lower right', fontsize=9)

ax.set_xlabel('Overall Accuracy')
ax.set_ylabel('Cross-Format Agreement (CFA)')
ax.set_title('Consistency vs. Accuracy')
ax.set_aspect('equal')

plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(os.path.join(FIGURES_DIR, f'figure_consistency_vs_accuracy.{ext}'))
plt.close()

# ============================================================
# Figure 6: Format-Conditional Accuracy Gap
# ============================================================
print("Generating Figure 6: Format Accuracy Gap...")
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(FORMATS))
width = 0.8 / n_models

for i, mn in enumerate(model_names):
    m = models[mn]
    acc_vals = [m['accuracy']['per_format'].get(f, 0) for f in FORMATS]
    offset = (i - n_models / 2 + 0.5) * width
    ax.bar(x + offset, acc_vals, width, label=mn, color=PALETTE[i], alpha=0.85)

ax.set_xlabel('Answer Format')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Format and Model')
ax.set_xticks(x)
ax.set_xticklabels([FORMAT_LABELS[f] for f in FORMATS])
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.set_ylim(0, 1.0)

plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(os.path.join(FIGURES_DIR, f'figure_format_accuracy_gap.{ext}'))
plt.close()

# ============================================================
# Figure 7: Size Scaling
# ============================================================
print("Generating Figure 7: Size Scaling...")
fig, ax = plt.subplots(figsize=(8, 5))

sizes = [models[mn]['size_b'] for mn in model_names]
accs = [models[mn]['accuracy']['mean'] for mn in model_names]
cfas = [models[mn]['cfa']['mean'] for mn in model_names]
cars = [models[mn]['car'] for mn in model_names]

ax.scatter(sizes, accs, color=PALETTE[0], s=80, marker='o', label='Accuracy', zorder=3)
ax.scatter(sizes, cfas, color=PALETTE[1], s=80, marker='s', label='CFA', zorder=3)

# Connect intra-family with solid lines
for family in set(models[mn]['family'] for mn in model_names):
    fam_models = [(models[mn]['size_b'], models[mn]['accuracy']['mean'], models[mn]['cfa']['mean'])
                  for mn in model_names if models[mn]['family'] == family]
    fam_models.sort()
    if len(fam_models) >= 2:
        fam_sizes = [f[0] for f in fam_models]
        fam_accs = [f[1] for f in fam_models]
        fam_cfas = [f[2] for f in fam_models]
        ax.plot(fam_sizes, fam_accs, color=PALETTE[0], alpha=0.5, linestyle='-')
        ax.plot(fam_sizes, fam_cfas, color=PALETTE[1], alpha=0.5, linestyle='-')

for mn in model_names:
    ax.annotate(mn, (models[mn]['size_b'], models[mn]['accuracy']['mean']),
                fontsize=7, ha='left', xytext=(5, 3), textcoords='offset points')

ax.set_xscale('log')
ax.set_xlabel('Model Size (B parameters)')
ax.set_ylabel('Score')
ax.set_title('Accuracy and Consistency vs. Model Size')
ax.legend(loc='lower right')

plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(os.path.join(FIGURES_DIR, f'figure_size_scaling.{ext}'))
plt.close()

# ============================================================
# Figure 8: CAR Comparison
# ============================================================
print("Generating Figure 8: CAR Comparison...")
fig, ax = plt.subplots(figsize=(8, 4))

car_vals = [(mn, models[mn]['car']) for mn in model_names]
car_vals.sort(key=lambda x: models[x[0]]['size_b'])

names = [c[0] for c in car_vals]
vals = [c[1] for c in car_vals]
colors = [PALETTE[list(set(models[mn]['family'] for mn in model_names)).index(models[mn]['family'])]
          for mn in names]

bars = ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='CAR = 1.0 (perfect)')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel('Consistency-Accuracy Ratio (CAR)')
ax.set_title('CAR by Model (lower = more knowledge is format-dependent)')
ax.legend()

plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(os.path.join(FIGURES_DIR, f'figure_car_comparison.{ext}'))
plt.close()

# ============================================================
# Figure 9: Judge Validation (if data exists)
# ============================================================
print("Generating Figure 9: Judge Validation...")
judge_path = os.path.join(BASE_DIR, 'exp', 'manual_validation', 'judge_calibration.json')
if os.path.exists(judge_path):
    with open(judge_path) as f:
        cal = json.load(f)

    cm = cal['confusion_matrix']
    matrix = np.array([
        [cm['both_equivalent'], cm['llm_only']],
        [cm['rule_only'], cm['both_not_equivalent']]
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rule: Equiv', 'Rule: Not Equiv'],
                yticklabels=['LLM: Equiv', 'LLM: Not Equiv'], ax=ax)
    ax.set_title(f'Answer Equivalence Judge Validation\nAgreement: {cal["agreement_rate"]:.1%}')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIGURES_DIR, f'figure_judge_validation.{ext}'))
    plt.close()

print("\nAll figures generated successfully!")
print(f"Saved to {FIGURES_DIR}/")
