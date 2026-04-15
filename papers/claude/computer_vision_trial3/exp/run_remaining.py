"""Complete remaining experiments: tau ablation + overhead."""
import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
os.environ['TMPDIR'] = '/var/tmp'
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import numpy as np

from data_loader import (ImageNetValDataset, CorruptedImageNetDataset, get_transform,
                         get_dataloader, get_calibration_indices,
                         REPRESENTATIVE_CORRUPTIONS)
from models import load_model, MODEL_CONFIGS
from stg import SpectralTokenGating
from metrics import evaluate_accuracy, save_results, set_seed, measure_latency

ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(ROOT, 'results')
DATA_DIR = os.path.join(ROOT, 'data')
LOG = os.path.join(ROOT, 'logs', 'run_remaining.log')

N_EVAL = 1000
N_CALIB = 1000
EVAL_INDICES = np.arange(N_EVAL)
NW = 4

import builtins
_print = builtins.print
def log_print(*args, **kwargs):
    _print(*args, **kwargs, flush=True)
    with open(LOG, 'a') as f:
        _print(*args, **kwargs, file=f, flush=True)
builtins.print = log_print


def eval_corruption_sev5(model, corruption, bs):
    ds = CorruptedImageNetDataset(DATA_DIR, corruption, 5, indices=EVAL_INDICES)
    loader = get_dataloader(ds, batch_size=bs, num_workers=NW)
    return evaluate_accuracy(model, loader)

def eval_clean(model, bs, n_images=None):
    indices = np.arange(n_images) if n_images else None
    ds = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=indices)
    loader = get_dataloader(ds, batch_size=bs, num_workers=NW)
    return evaluate_accuracy(model, loader)


# ============================================================
# TAU ABLATION (complete remaining: 95th and 99th)
# ============================================================
result_path = os.path.join(RESULTS_DIR, 'ablation_tau.json')
if not os.path.exists(result_path):
    print("=== Completing tau ablation ===")
    results = {'ablation': 'tau', 'model': 'deit_small', 'seed': 42,
               'corruptions_tested': REPRESENTATIVE_CORRUPTIONS, 'variants': []}

    for tau_pct in [75, 85, 90, 95, 99]:
        set_seed(42)
        model, config = load_model('deit_small')
        bs = config['batch_size']
        stg = SpectralTokenGating(model, 'deit_small', K=3, alpha=5.0, tau_percentile=tau_pct)
        cal_indices = get_calibration_indices(42, n_calibration=N_CALIB)
        cal_ds = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=cal_indices)
        cal_loader = get_dataloader(cal_ds, batch_size=bs, num_workers=NW)
        stg.calibrate(cal_loader, n_images=N_CALIB)
        stg.enable()
        clean = eval_clean(model, bs, N_EVAL)
        corr_accs = {}
        for c in REPRESENTATIVE_CORRUPTIONS:
            acc = eval_corruption_sev5(model, c, bs)
            corr_accs[c] = acc['top1']
        mean_corr = np.mean(list(corr_accs.values()))
        results['variants'].append({
            'label': f'tau={tau_pct}th', 'clean_top1': clean['top1'],
            'corruption_accs': corr_accs, 'mean_corruption_acc': float(mean_corr),
        })
        print(f"  [tau] tau={tau_pct}th: clean={clean['top1']*100:.1f}%, corr={mean_corr*100:.1f}%")
        stg.disable(); del model, stg; torch.cuda.empty_cache()

    save_results(results, result_path)
    print("Tau ablation saved.")
else:
    print("[SKIP] ablation_tau")


# ============================================================
# OVERHEAD MEASUREMENT
# ============================================================
result_path = os.path.join(RESULTS_DIR, 'overhead.json')
if not os.path.exists(result_path):
    print("\n=== Measuring computational overhead ===")
    results = {}
    for model_key in ['deit_small', 'deit_base', 'swin_tiny']:
        set_seed(42)
        model, config = load_model(model_key)
        bs = config['batch_size']
        dummy = torch.randn(bs, 3, 224, 224).cuda()

        vanilla_lat = measure_latency(model, dummy, n_warmup=10, n_measure=50)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(): model(dummy)
        vanilla_mem = torch.cuda.max_memory_allocated() / 1e6

        stg = SpectralTokenGating(model, model_key, K=3, alpha=5.0, tau_percentile=95)
        cal_indices = get_calibration_indices(42, n_calibration=100)
        cal_ds = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=cal_indices)
        cal_loader = get_dataloader(cal_ds, batch_size=bs, num_workers=NW)
        stg.calibrate(cal_loader, n_images=100)
        stg.enable()

        torch.cuda.reset_peak_memory_stats()
        stg_lat = measure_latency(model, dummy, n_warmup=10, n_measure=50)
        with torch.no_grad(): model(dummy)
        stg_mem = torch.cuda.max_memory_allocated() / 1e6
        stg.disable()

        overhead_pct = (stg_lat['mean_ms'] - vanilla_lat['mean_ms']) / vanilla_lat['mean_ms'] * 100
        results[model_key] = {
            'vanilla_latency_ms': vanilla_lat,
            'stg_latency_ms': stg_lat,
            'overhead_pct': float(overhead_pct),
            'vanilla_memory_MB': float(vanilla_mem),
            'stg_memory_MB': float(stg_mem),
            'memory_overhead_MB': float(stg_mem - vanilla_mem),
            'batch_size': bs,
        }
        print(f"  {model_key}: overhead={overhead_pct:.1f}%")
        del model, stg; torch.cuda.empty_cache()

    save_results(results, result_path)
    print("Overhead saved.")
else:
    print("[SKIP] overhead")

print("\nAll remaining experiments done!")
