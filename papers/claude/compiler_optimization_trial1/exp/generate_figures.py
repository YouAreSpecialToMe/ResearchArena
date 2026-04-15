"""Generate all figures and tables for the paper."""
import sys
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))
from exp.shared.utils import *

FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Pass categories for coloring
LOOP_PASSES = {'indvars', 'licm', 'loop-deletion', 'loop-idiom', 'loop-reduce',
               'loop-rotate', 'loop-simplify', 'loop-sink', 'loop-fusion',
               'loop-distribute', 'loop-unroll', 'lcssa'}
SCALAR_PASSES = {'instcombine', 'gvn', 'newgvn', 'sccp', 'reassociate',
                 'aggressive-instcombine', 'early-cse', 'float2int',
                 'constraint-elimination', 'nary-reassociate', 'instsimplify',
                 'div-rem-pairs', 'gvn-hoist', 'gvn-sink', 'correlated-propagation'}
MEMORY_PASSES = {'mem2reg', 'sroa', 'dse', 'memcpyopt'}
CFG_PASSES = {'simplifycfg', 'jump-threading', 'adce', 'bdce', 'dce',
              'sink', 'mergereturn', 'flattencfg'}

def get_category(p):
    if p in LOOP_PASSES: return 'Loop'
    if p in SCALAR_PASSES: return 'Scalar'
    if p in MEMORY_PASSES: return 'Memory'
    if p in CFG_PASSES: return 'CFG'
    return 'Other'

CAT_COLORS = {'Loop': '#2196F3', 'Scalar': '#4CAF50', 'Memory': '#FF9800',
              'CFG': '#F44336', 'Other': '#9E9E9E'}


def fig1_idempotency():
    """Bar chart of per-pass idempotency rate."""
    print("  Figure 1: Idempotency...")
    data = []
    with open(RESULTS_DIR / "idempotency_summary.csv") as f:
        for row in csv.DictReader(f):
            data.append(row)

    data.sort(key=lambda x: -float(x['structural_idempotency_rate']))

    fig, ax = plt.subplots(figsize=(14, 5))
    names = [d['pass_name'] for d in data]
    rates = [float(d['structural_idempotency_rate']) for d in data]
    colors = [CAT_COLORS[get_category(n)] for n in names]

    bars = ax.bar(range(len(names)), rates, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel('Structural Idempotency Rate')
    ax.set_title('Per-Pass Idempotency Rate Across All Benchmarks')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')

    legend_patches = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()]
    ax.legend(handles=legend_patches, loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_idempotency.pdf", dpi=150, bbox_inches='tight')
    plt.close()


def fig2_commutativity_heatmap():
    """Commutativity heatmap."""
    print("  Figure 2: Commutativity heatmap...")
    passes = get_pass_list()

    matrix = np.ones((len(passes), len(passes)))
    with open(RESULTS_DIR / "commutativity_matrix.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            for j in range(1, len(row)):
                matrix[i][j-1] = float(row[j])

    # Sort by category
    order = sorted(range(len(passes)), key=lambda i: (get_category(passes[i]), passes[i]))
    matrix_sorted = matrix[np.ix_(order, order)]
    names_sorted = [passes[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix_sorted, xticklabels=names_sorted, yticklabels=names_sorted,
                cmap='RdYlGn', vmin=0, vmax=1, ax=ax, linewidths=0.1,
                cbar_kws={'label': 'Commutativity Rate'})
    ax.set_title('Pairwise Commutativity Matrix (grouped by pass category)')
    ax.set_xticklabels(names_sorted, rotation=90, fontsize=6)
    ax.set_yticklabels(names_sorted, rotation=0, fontsize=6)

    # Add category separators
    categories = [get_category(n) for n in names_sorted]
    prev_cat = categories[0]
    for i, cat in enumerate(categories):
        if cat != prev_cat:
            ax.axhline(y=i, color='black', linewidth=1.5)
            ax.axvline(x=i, color='black', linewidth=1.5)
            prev_cat = cat

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_commutativity_heatmap.pdf", dpi=150, bbox_inches='tight')
    plt.close()


def fig3_commutativity_graph():
    """Network visualization of commutativity graph."""
    print("  Figure 3: Commutativity graph...")
    import networkx as nx

    passes = get_pass_list()
    matrix = {}
    with open(RESULTS_DIR / "commutativity_matrix.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            pi = row[0]
            for j, pj in enumerate(header[1:]):
                matrix[(pi, pj)] = float(row[j+1])

    G = nx.Graph()
    for p in passes:
        G.add_node(p, category=get_category(p))

    # Add edges for non-commutative pairs (commutativity < 0.5)
    for pi, pj in combinations(passes, 2):
        rate = matrix.get((pi, pj), 1.0)
        if rate < 0.5:
            G.add_edge(pi, pj, weight=1 - rate)

    fig, ax = plt.subplots(figsize=(12, 10))

    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    node_colors = [CAT_COLORS[get_category(n)] for n in G.nodes()]
    node_sizes = [300 + 100 * G.degree(n) for n in G.nodes()]

    # Draw edges with varying width
    edges = G.edges(data=True)
    edge_weights = [d.get('weight', 0.5) * 2 for _, _, d in edges]

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.8, ax=ax, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_labels(G, pos, font_size=5, ax=ax)

    legend_patches = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=9)
    ax.set_title(f'Non-Commutativity Graph ({G.number_of_edges()} non-commutative pairs)')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_commutativity_graph.pdf", dpi=150, bbox_inches='tight')
    plt.close()


def fig4_interference_heatmap():
    """Interference heatmap."""
    print("  Figure 4: Interference heatmap...")
    passes = get_pass_list()

    matrix = np.zeros((len(passes), len(passes)))
    with open(RESULTS_DIR / "interference_matrix.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            for j in range(1, len(row)):
                matrix[i][j-1] = float(row[j])

    order = sorted(range(len(passes)), key=lambda i: (get_category(passes[i]), passes[i]))
    matrix_sorted = matrix[np.ix_(order, order)]
    names_sorted = [passes[i] for i in order]

    # Clip for better visualization
    vmax = np.percentile(np.abs(matrix_sorted), 95)
    vmax = max(vmax, 1)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix_sorted, xticklabels=names_sorted, yticklabels=names_sorted,
                cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax, ax=ax, linewidths=0.1,
                cbar_kws={'label': 'Mean Interference (%)'})
    ax.set_title('Pairwise Interference Matrix (red=destructive, blue=constructive)')
    ax.set_xticklabels(names_sorted, rotation=90, fontsize=6)
    ax.set_yticklabels(names_sorted, rotation=0, fontsize=6)

    categories = [get_category(n) for n in names_sorted]
    prev_cat = categories[0]
    for i, cat in enumerate(categories):
        if cat != prev_cat:
            ax.axhline(y=i, color='black', linewidth=1.5)
            ax.axvline(x=i, color='black', linewidth=1.5)
            prev_cat = cat

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_interference_heatmap.pdf", dpi=150, bbox_inches='tight')
    plt.close()


def fig5_convergence():
    """Convergence curves for representative benchmarks."""
    print("  Figure 5: Convergence curves...")

    # Load convergence data
    conv_data = {}
    with open(RESULTS_DIR / "convergence_raw.csv") as f:
        for row in csv.DictReader(f):
            bm = row['benchmark']
            pipeline = row.get('pipeline', '-O2')
            if pipeline not in conv_data:
                conv_data[pipeline] = {}
            if bm not in conv_data[pipeline]:
                conv_data[pipeline][bm] = []
            conv_data[pipeline][bm].append({
                'iteration': int(row['iteration']),
                'instcount': int(row['instcount']),
                'oscillating': row.get('oscillating', 'False') == 'True'
            })

    # Select representative benchmarks
    # Find one fast-converging, one slow, one oscillating
    o2_data = conv_data.get('-O2', {})
    fast_conv = None
    slow_conv = None
    oscillating = None

    for bm, traj in o2_data.items():
        n_iters = len(traj)
        is_osc = any(t['oscillating'] for t in traj)
        if is_osc and oscillating is None:
            oscillating = bm
        elif n_iters <= 3 and not is_osc and fast_conv is None:
            fast_conv = bm
        elif n_iters >= 5 and not is_osc and slow_conv is None:
            slow_conv = bm

    # Fallbacks
    if fast_conv is None:
        fast_conv = list(o2_data.keys())[0]
    if slow_conv is None:
        slow_conv = list(o2_data.keys())[min(5, len(o2_data)-1)]

    selected = []
    if fast_conv: selected.append(('Fast convergence', fast_conv))
    if slow_conv: selected.append(('Slow convergence', slow_conv))
    if oscillating: selected.append(('Oscillating', oscillating))

    fig, axes = plt.subplots(1, len(selected), figsize=(5*len(selected), 4), sharey=False)
    if len(selected) == 1:
        axes = [axes]

    for idx, (label, bm) in enumerate(selected):
        ax = axes[idx]
        for pipeline, color, marker in [('-O2', '#2196F3', 'o'), ('-O3', '#4CAF50', 's'), ('-Oz', '#FF9800', '^')]:
            if pipeline in conv_data and bm in conv_data[pipeline]:
                traj = conv_data[pipeline][bm]
                iters = [t['iteration'] for t in traj]
                ics = [t['instcount'] for t in traj]
                ax.plot(iters, ics, marker=marker, markersize=4, label=pipeline,
                       color=color, linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Instruction Count')
        ax.set_title(f'{label}\n({bm})', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Convergence Behavior Under Iterative Pipeline Application', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_convergence.pdf", dpi=150, bbox_inches='tight')
    plt.close()


def fig6_cycles():
    """State-transition diagrams for detected cycles."""
    print("  Figure 6: Cycle state-transition diagrams...")

    import networkx as nx

    cycle_details = []
    with open(RESULTS_DIR / "minimal_cycles.csv") as f:
        for row in csv.DictReader(f):
            cycle_details.append(row)

    if not cycle_details:
        print("    No cycles found, skipping fig6")
        return

    # Get unique cycles by (subset, cycle_length), pick diverse examples
    unique_cycles = {}
    for c in cycle_details:
        key = (c['subset'], c['cycle_length'])
        if key not in unique_cycles:
            unique_cycles[key] = c

    # Select 3 most interesting (different lengths/subsets)
    sorted_cycles = sorted(unique_cycles.values(),
                          key=lambda c: (-int(c['ic_amplitude']), c['subset']))[:3]

    # Load trajectory data for these cycles
    traj_data = {}
    with open(RESULTS_DIR / "cycle_detection_raw.csv") as f:
        for row in csv.DictReader(f):
            key = (row['pass_subset'], row['benchmark'])
            if key not in traj_data:
                traj_data[key] = []
            traj_data[key].append(row)

    n_plots = min(3, len(sorted_cycles))
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    for idx, cycle in enumerate(sorted_cycles[:n_plots]):
        ax = axes[idx]
        key = (cycle['subset'], cycle['benchmark'])
        traj = traj_data.get(key, [])

        if not traj:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Plot IC trajectory showing the cycle
        iters = [int(t['iteration']) for t in traj]
        ics = [int(t['instcount']) for t in traj]
        hashes = [t['ir_hash'][:8] for t in traj]

        ax.plot(iters, ics, 'b-o', markersize=3, linewidth=1)

        # Highlight the cycling region
        cycle_start = int(cycle.get('cycle_start_iter', 0))
        cycle_len = int(cycle['cycle_length'])
        if cycle_start > 0 and cycle_start < len(iters):
            cycle_iters = iters[cycle_start-1:]
            cycle_ics = ics[cycle_start-1:]
            ax.plot(cycle_iters, cycle_ics, 'r-o', markersize=4, linewidth=2, label='Cycle region')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Instruction Count')
        ax.set_title(f'{cycle["subset"]}\non {cycle["benchmark"]}\n'
                    f'(length={cycle["cycle_length"]}, amp={cycle["ic_amplitude"]})',
                    fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Detected Optimization Cycles (Oscillating IR States)', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_cycles.pdf", dpi=150, bbox_inches='tight')
    plt.close()


def fig7_ordering_comparison():
    """Grouped bar chart comparing all methods."""
    print("  Figure 7: Ordering comparison...")

    # Load algebra ordering results
    alg_json = Path(__file__).parent / "algebra_ordering" / "results.json"
    with open(alg_json) as f:
        alg_data = json.load(f)

    methods = alg_data.get('methods_comparison', {})

    # Order methods
    method_order = ['O3', 'O2', 'random', 'O1', 'Oz', 'algebra', 'greedy']
    method_names = [m for m in method_order if m in methods]
    values = [methods[m] for m in method_names]
    reductions = [(1 - v) * 100 for v in values]

    # Colors
    colors = []
    for m in method_names:
        if m in ['O1', 'O2', 'O3', 'Oz']:
            colors.append('#90CAF9')  # Light blue for standard
        elif m == 'algebra':
            colors.append('#4CAF50')  # Green for our method
        elif m == 'greedy':
            colors.append('#FFA726')  # Orange for greedy
        else:
            colors.append('#BDBDBD')  # Gray for random

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(method_names)), reductions, color=colors, edgecolor='black',
                  linewidth=0.5)

    # Add value labels
    for bar, red in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{red:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    display_names = {
        'O1': '-O1', 'O2': '-O2', 'O3': '-O3', 'Oz': '-Oz',
        'random': 'Random', 'greedy': 'Greedy', 'algebra': 'Algebra-\nGuided'
    }
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([display_names.get(m, m) for m in method_names], fontsize=10)
    ax.set_ylabel('Instruction Count Reduction (%)')
    ax.set_title('Optimization Method Comparison (Geometric Mean Across 87 Benchmarks)')
    ax.grid(axis='y', alpha=0.3)

    legend_patches = [
        mpatches.Patch(color='#90CAF9', label='Standard LLVM levels'),
        mpatches.Patch(color='#4CAF50', label='Algebra-guided (ours)'),
        mpatches.Patch(color='#FFA726', label='Greedy search'),
        mpatches.Patch(color='#BDBDBD', label='Random ordering'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_ordering_comparison.pdf", dpi=150, bbox_inches='tight')
    plt.close()


def fig8_ablation():
    """Ablation study bar chart."""
    print("  Figure 8: Ablation study...")

    abl_json = Path(__file__).parent / "ablation" / "results.json"
    with open(abl_json) as f:
        abl_data = json.load(f)

    variants = abl_data.get('variants', [])
    if not variants:
        print("    No ablation data, skipping")
        return

    # Full method as reference
    full = next((v for v in variants if v['variant'] == 'full'), None)
    if not full:
        return

    full_red = full['reduction_pct']

    fig, ax = plt.subplots(figsize=(10, 5))

    names = []
    reductions = []
    stds = []
    colors_list = []

    display = {
        'full': 'Full Method',
        'no_synergy': 'No Synergy\nChaining',
        'no_anti_interference': 'No Anti-\nInterference',
        'no_pruning': 'No Idempotency\nPruning',
        'no_phases': 'No Phase\nOrdering',
        'top_k_only': 'Top-K\nPasses Only',
    }

    for v in variants:
        names.append(display.get(v['variant'], v['variant']))
        reductions.append(v['reduction_pct'])
        stds.append(v['std_ratio'] * 100)
        if v['variant'] == 'full':
            colors_list.append('#4CAF50')
        elif v['reduction_pct'] < full_red - 1:
            colors_list.append('#EF5350')
        else:
            colors_list.append('#90CAF9')

    bars = ax.bar(range(len(names)), reductions, yerr=stds, color=colors_list,
                  edgecolor='black', linewidth=0.5, capsize=3)

    for bar, red in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds) + 0.5,
               f'{red:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Instruction Count Reduction (%)')
    ax.set_title('Ablation Study: Contribution of Each Algebraic Component')
    ax.axhline(y=full_red, color='green', linestyle='--', alpha=0.5, label='Full method')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_ablation.pdf", dpi=150, bbox_inches='tight')
    plt.close()


def table1_summary():
    """Summary statistics LaTeX table."""
    print("  Table 1: Summary statistics...")

    stats = {}
    with open(RESULTS_DIR / "statistical_analysis.json") as f:
        stats = json.load(f)

    # Load additional data
    idem = stats.get('idempotency', {})
    comm = stats.get('commutativity', {})
    interf = stats.get('interference', {})
    osc = stats.get('oscillation', {})
    order = stats.get('ordering', {})

    tex = r"""\begin{table}[t]
\centering
\caption{Summary of Algebraic Properties Across 87 Benchmarks and 46 LLVM Passes}
\label{tab:summary}
\begin{tabular}{lrrl}
\toprule
\textbf{Property} & \textbf{Value} & \textbf{Criterion} & \textbf{Met?} \\
\midrule
Idempotency rate & """ + f"{idem.get('rate', 0)*100:.1f}" + r"""\% & $>60\%$ & """ + ('Yes' if idem.get('criterion_met') else 'No') + r""" \\
Non-commutativity rate & """ + f"{comm.get('non_commutative_rate', 0)*100:.1f}" + r"""\% & $>30\%$ & """ + ('Yes' if comm.get('criterion_met') else 'No') + r""" \\
Significant interference rate & """ + f"{interf.get('significant_rate', 0)*100:.1f}" + r"""\% & $>10\%$ & """ + ('Yes' if interf.get('criterion_met') else 'No') + r""" \\
Oscillating benchmarks (-O2) & """ + str(osc.get('oscillating_benchmarks_O2', 0)) + r"""/87 & $>0$ & """ + ('Yes' if osc.get('criterion_met') else 'No') + r""" \\
True pass oscillation cycles & """ + str(osc.get('true_pass_cycles', 0)) + r""" & $>0$ & """ + ('Yes' if osc.get('criterion_met') else 'No') + r""" \\
Algebra-guided IC ratio & """ + f"{order.get('algebra_geo_mean', 0):.4f}" + r""" & $<$ -O2 (""" + f"{order.get('O2_geo_mean', 0):.4f}" + r""") & """ + ('Yes' if order.get('criterion_met') else 'No') + r""" \\
\midrule
\multicolumn{4}{l}{\textit{Criteria met: """ + str(sum(1 for k in ['idempotency', 'commutativity', 'interference', 'oscillation', 'ordering', 'clustering'] if stats.get(k, {}).get('criterion_met'))) + r"""/6}} \\
\bottomrule
\end{tabular}
\end{table}"""

    with open(FIGURES_DIR / "table1_summary.tex", 'w') as f:
        f.write(tex)


def table2_top_pairs():
    """Top synergistic and destructive pass pairs LaTeX table."""
    print("  Table 2: Top pass pairs...")

    interf_data = json.load(open(Path(__file__).parent / "interference" / "results.json"))

    tex = r"""\begin{table}[t]
\centering
\caption{Top Synergistic and Destructive Pass Pairs}
\label{tab:top_pairs}
\begin{tabular}{llr}
\toprule
\textbf{Pass A} & \textbf{Pass B} & \textbf{Interference (\%)} \\
\midrule
\multicolumn{3}{l}{\textit{Top Constructive (Synergistic) Pairs}} \\
"""
    for pair in interf_data.get('top_constructive', [])[:5]:
        tex += f"\\texttt{{{pair['pass_i']}}} & \\texttt{{{pair['pass_j']}}} & +{pair['interference_pct']:.2f} \\\\\n"

    tex += r"""\midrule
\multicolumn{3}{l}{\textit{Top Destructive Pairs}} \\
"""
    for pair in interf_data.get('top_destructive', [])[:5]:
        tex += f"\\texttt{{{pair['pass_i']}}} & \\texttt{{{pair['pass_j']}}} & {pair['interference_pct']:.2f} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}"""

    with open(FIGURES_DIR / "table2_top_pairs.tex", 'w') as f:
        f.write(tex)


def main():
    print("Generating figures and tables...")
    fig1_idempotency()
    fig2_commutativity_heatmap()
    fig3_commutativity_graph()
    fig4_interference_heatmap()
    fig5_convergence()
    fig6_cycles()
    fig7_ordering_comparison()
    fig8_ablation()
    table1_summary()
    table2_top_pairs()
    print("All figures and tables generated.")


if __name__ == '__main__':
    main()
