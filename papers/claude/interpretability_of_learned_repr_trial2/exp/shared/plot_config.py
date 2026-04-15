"""Consistent figure styling for publication-quality plots."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Colorblind-friendly palette
PALETTE = sns.color_palette("colorblind", 10)
CAPABILITY_COLORS = {
    'factual': PALETTE[0],
    'syntax': PALETTE[1],
    'semantic': PALETTE[2],
    'sentiment': PALETTE[3],
    'ner': PALETTE[4],
    'reasoning': PALETTE[5],
}

CAPABILITY_LABELS = {
    'factual': 'Factual Knowledge',
    'syntax': 'Syntactic Processing',
    'semantic': 'Semantic Understanding',
    'sentiment': 'Sentiment Analysis',
    'ner': 'Named Entity Recognition',
    'reasoning': 'Simple Reasoning',
}

def setup_style():
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
    })
    sns.set_style("whitegrid")
