"""Master experiment runner for STG experiments.

Optimized strategy:
- Clean accuracy: 10K images (reliable, fast)
- Corruption evaluation: 1K images × severity 5 only for main results
- Multi-severity: DeiT-S only on 5 representative corruptions
- Ablations: 1K images × severity 5 × 5 corruptions
"""
import sys, os, json, time, copy, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
os.environ['TMPDIR'] = '/var/tmp'
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data_loader import (ImageNetValDataset, CorruptedImageNetDataset, get_transform,
                         get_dataloader, get_calibration_indices, CORRUPTION_TYPES,
                         REPRESENTATIVE_CORRUPTIONS, CATEGORY_MAP, compute_mce, ALEXNET_ERR)
from models import load_model, MODEL_CONFIGS, get_model_blocks
from stg import SpectralTokenGating
from metrics import evaluate_accuracy, save_results, set_seed, measure_latency

ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(ROOT, 'results')
DATA_DIR = os.path.join(ROOT, 'data')
CHECKPOINT_DIR = os.path.join(ROOT, 'checkpoints')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
N_EVAL = 1000
N_CLEAN = 10000  # images for clean accuracy eval
N_CALIB = 1000
EVAL_INDICES = np.arange(N_EVAL)
CLEAN_INDICES = np.arange(N_CLEAN)
NW = 4  # num workers


def eval_corruption_sev5(model, corruption, bs):
    """Evaluate on a single corruption at severity 5."""
    ds = CorruptedImageNetDataset(DATA_DIR, corruption, 5, indices=EVAL_INDICES)
    loader = get_dataloader(ds, batch_size=bs, num_workers=NW)
    return evaluate_accuracy(model, loader)


def eval_clean(model, bs, n_images=None):
    """Evaluate on clean images."""
    indices = np.arange(n_images) if n_images else None
    ds = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=indices)
    loader = get_dataloader(ds, batch_size=bs, num_workers=NW)
    return evaluate_accuracy(model, loader)


# ============================================================
# VANILLA BASELINE
# ============================================================
def run_vanilla_baseline(model_key):
    result_path = os.path.join(RESULTS_DIR, f'vanilla_{model_key}.json')
    if os.path.exists(result_path):
        print(f"[SKIP] vanilla_{model_key}")
        return json.load(open(result_path))

    set_seed(42)
    model, config = load_model(model_key)
    bs = config['batch_size']

    print(f"\n[Vanilla {model_key}] Clean eval...")
    clean = eval_clean(model, bs, N_CLEAN)
    print(f"  Clean top1: {clean['top1']*100:.2f}%")

    print(f"  Corruption eval (sev 5, {N_EVAL} imgs)...")
    t0 = time.time()
    sev5_accs = {}
    for c in CORRUPTION_TYPES:
        acc = eval_corruption_sev5(model, c, bs)
        sev5_accs[c] = acc['top1']
        print(f"    {c}: {acc['top1']*100:.1f}%")

    results = {
        'model': model_key, 'method': 'vanilla',
        'clean': clean,
        'sev5_accs': sev5_accs,
        'mean_sev5_acc': float(np.mean(list(sev5_accs.values()))),
        'mean_sev5_err': float(np.mean([1-v for v in sev5_accs.values()])),
        'time_min': (time.time() - t0) / 60,
    }

    save_results(results, result_path)
    del model; torch.cuda.empty_cache()
    print(f"  Mean sev5 err: {results['mean_sev5_err']*100:.1f}%")
    return results


# ============================================================
# TENT BASELINE
# ============================================================
def run_tent_baseline(model_key):
    result_path = os.path.join(RESULTS_DIR, f'tent_{model_key}.json')
    if os.path.exists(result_path):
        print(f"[SKIP] tent_{model_key}")
        return json.load(open(result_path))

    set_seed(42)
    model, config = load_model(model_key)
    bs = min(config['batch_size'], 64)

    def tent_evaluate(model, dataloader, lr=1e-3):
        original_state = {n: p.data.clone() for n, p in model.named_parameters()}
        model.train()
        for p in model.parameters():
            p.requires_grad = False
        trainable = []
        for m in model.modules():
            if isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                trainable.extend([m.weight, m.bias])
        optimizer = optim.SGD(trainable, lr=lr)

        correct1 = correct5 = total = 0
        for images, labels in dataloader:
            images = images.to('cuda', non_blocking=True)
            labels = labels.to('cuda', non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                _, p1 = outputs.topk(1, dim=1)
                _, p5 = outputs.topk(5, dim=1)
                correct1 += (p1.squeeze() == labels).sum().item()
                correct5 += (p5 == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)

        for n, p in model.named_parameters():
            p.data.copy_(original_state[n])
        model.eval()
        return {'top1': correct1/total, 'top5': correct5/total}

    print(f"\n[Tent {model_key}] Clean eval...")
    ds = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=CLEAN_INDICES)
    loader = get_dataloader(ds, batch_size=bs, num_workers=NW)
    clean = tent_evaluate(model, loader)
    print(f"  Clean top1: {clean['top1']*100:.2f}%")

    print(f"  Corruption eval (sev 5)...")
    t0 = time.time()
    sev5_accs = {}
    for c in CORRUPTION_TYPES:
        ds = CorruptedImageNetDataset(DATA_DIR, c, 5, indices=EVAL_INDICES)
        loader = get_dataloader(ds, batch_size=bs, num_workers=NW)
        acc = tent_evaluate(model, loader)
        sev5_accs[c] = acc['top1']
        print(f"    {c}: {acc['top1']*100:.1f}%")

    results = {
        'model': model_key, 'method': 'tent',
        'clean': clean,
        'sev5_accs': sev5_accs,
        'mean_sev5_acc': float(np.mean(list(sev5_accs.values()))),
        'mean_sev5_err': float(np.mean([1-v for v in sev5_accs.values()])),
        'time_min': (time.time() - t0) / 60,
    }
    save_results(results, result_path)
    del model; torch.cuda.empty_cache()
    print(f"  Mean sev5 err: {results['mean_sev5_err']*100:.1f}%")
    return results


# ============================================================
# STG MAIN
# ============================================================
def run_stg_experiment(model_key, seed):
    result_path = os.path.join(RESULTS_DIR, f'stg_{model_key}_seed{seed}.json')
    if os.path.exists(result_path):
        print(f"[SKIP] stg_{model_key}_seed{seed}")
        return json.load(open(result_path))

    set_seed(seed)
    model, config = load_model(model_key)
    bs = config['batch_size']

    # Calibration
    print(f"\n[STG {model_key} seed={seed}] Calibrating...")
    cal_indices = get_calibration_indices(seed, n_calibration=N_CALIB)
    cal_ds = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=cal_indices)
    cal_loader = get_dataloader(cal_ds, batch_size=bs, num_workers=NW)

    stg = SpectralTokenGating(model, model_key, K=3, alpha=5.0, tau_percentile=95)
    t0 = time.time()
    stg.calibrate(cal_loader, n_images=N_CALIB)
    cal_time = time.time() - t0

    cal_path = os.path.join(CHECKPOINT_DIR, f'calibration_{model_key}_seed{seed}.pt')
    stg.save_calibration(cal_path)
    stg.enable()

    # Clean accuracy
    clean = eval_clean(model, bs, N_CLEAN)
    print(f"  Clean top1: {clean['top1']*100:.2f}%")

    # Severity 5 corruption accuracy
    print(f"  Corruption eval (sev 5)...")
    t0 = time.time()
    sev5_accs = {}
    for c in CORRUPTION_TYPES:
        acc = eval_corruption_sev5(model, c, bs)
        sev5_accs[c] = acc['top1']
        print(f"    {c}: {acc['top1']*100:.1f}%")

    stg.disable()

    results = {
        'model': model_key, 'seed': seed, 'method': 'stg',
        'stg_config': {'K': 3, 'alpha': 5.0, 'tau_percentile': 95,
                       'layers': config['default_stg_layers']},
        'calibration_time_s': cal_time,
        'clean': clean,
        'sev5_accs': sev5_accs,
        'mean_sev5_acc': float(np.mean(list(sev5_accs.values()))),
        'mean_sev5_err': float(np.mean([1-v for v in sev5_accs.values()])),
        'time_min': (time.time() - t0) / 60,
    }
    save_results(results, result_path)
    del model, stg; torch.cuda.empty_cache()
    print(f"  Mean sev5 err: {results['mean_sev5_err']*100:.1f}%")
    return results


# ============================================================
# MULTI-SEVERITY ANALYSIS (DeiT-S only, representative corruptions)
# ============================================================
def run_severity_analysis():
    """Full severity analysis for DeiT-S vanilla vs STG on representative corruptions."""
    result_path = os.path.join(RESULTS_DIR, 'severity_analysis.json')
    if os.path.exists(result_path):
        print(f"[SKIP] severity_analysis")
        return json.load(open(result_path))

    print(f"\n[Severity Analysis] DeiT-S, representative corruptions, all severities...")
    results = {'model': 'deit_small', 'corruptions': REPRESENTATIVE_CORRUPTIONS}

    for method in ['vanilla', 'stg']:
        set_seed(42)
        model, config = load_model('deit_small')
        bs = config['batch_size']

        stg_obj = None
        if method == 'stg':
            stg_obj = SpectralTokenGating(model, 'deit_small', K=3, alpha=5.0, tau_percentile=95)
            cal_indices = get_calibration_indices(42, n_calibration=N_CALIB)
            cal_ds = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=cal_indices)
            cal_loader = get_dataloader(cal_ds, batch_size=bs, num_workers=NW)
            stg_obj.calibrate(cal_loader, n_images=N_CALIB)
            stg_obj.enable()

        method_results = {}
        for c in REPRESENTATIVE_CORRUPTIONS:
            method_results[c] = {}
            for sev in [1, 2, 3, 4, 5]:
                ds = CorruptedImageNetDataset(DATA_DIR, c, sev, indices=EVAL_INDICES)
                loader = get_dataloader(ds, batch_size=bs, num_workers=NW)
                acc = evaluate_accuracy(model, loader)
                method_results[c][str(sev)] = acc['top1']
                print(f"    {method} {c} sev{sev}: {acc['top1']*100:.1f}%")

        results[method] = method_results
        if stg_obj:
            stg_obj.disable()
        del model; torch.cuda.empty_cache()

    save_results(results, result_path)
    return results


# ============================================================
# ABLATION STUDIES
# ============================================================
def run_ablation(name, stg_kwargs_list, labels):
    result_path = os.path.join(RESULTS_DIR, f'ablation_{name}.json')
    if os.path.exists(result_path):
        print(f"[SKIP] ablation_{name}")
        return json.load(open(result_path))

    results = {'ablation': name, 'model': 'deit_small', 'seed': 42,
               'corruptions_tested': REPRESENTATIVE_CORRUPTIONS, 'variants': []}

    for label, kwargs in zip(labels, stg_kwargs_list):
        set_seed(42)
        model, config = load_model('deit_small')
        bs = config['batch_size']

        stg = SpectralTokenGating(model, 'deit_small', **kwargs)
        cal_indices = get_calibration_indices(42, n_calibration=N_CALIB)
        cal_ds = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=cal_indices)
        cal_loader = get_dataloader(cal_ds, batch_size=bs, num_workers=NW)
        stg.calibrate(cal_loader, n_images=N_CALIB)
        stg.enable()

        # Clean accuracy on eval subset
        clean = eval_clean(model, bs, N_EVAL)

        # Representative corruptions at severity 5
        corr_accs = {}
        for c in REPRESENTATIVE_CORRUPTIONS:
            acc = eval_corruption_sev5(model, c, bs)
            corr_accs[c] = acc['top1']

        mean_corr = np.mean(list(corr_accs.values()))
        results['variants'].append({
            'label': label,
            'clean_top1': clean['top1'],
            'corruption_accs': corr_accs,
            'mean_corruption_acc': float(mean_corr),
        })
        print(f"  [{name}] {label}: clean={clean['top1']*100:.1f}%, corr={mean_corr*100:.1f}%")
        stg.disable(); del model, stg; torch.cuda.empty_cache()

    save_results(results, result_path)
    return results


def run_all_ablations():
    print(f"\n{'='*60}")
    print("Running ablation studies...")

    run_ablation('K',
        [{'K': k, 'alpha': 5.0, 'tau_percentile': 95} for k in [2, 3, 4, 5]],
        [f'K={k}' for k in [2, 3, 4, 5]])

    run_ablation('layers',
        [{'K': 3, 'alpha': 5.0, 'tau_percentile': 95, 'target_layers': l}
         for l in [[0,1,2], [4,5,6,7], [9,10,11], [3,6,9,11], list(range(12))]],
        ['early[0-2]', 'middle[4-7]', 'late[9-11]', 'spread[3,6,9,11]', 'all[0-11]'])

    run_ablation('gating',
        [{'K': 3, 'alpha': 5.0, 'tau_percentile': 95, 'gating_fn': g}
         for g in ['sigmoid', 'hard', 'linear']],
        ['sigmoid', 'hard', 'linear'])

    # Calibration size - custom logic
    result_path = os.path.join(RESULTS_DIR, 'ablation_calibration_size.json')
    if not os.path.exists(result_path):
        print("\n--- Ablation: Calibration size ---")
        results = {'ablation': 'calibration_size', 'model': 'deit_small', 'seed': 42,
                   'corruptions_tested': REPRESENTATIVE_CORRUPTIONS, 'variants': []}
        for n_cal in [50, 100, 250, 500, 1000, 2000]:
            set_seed(42)
            model, config = load_model('deit_small')
            bs = config['batch_size']
            stg = SpectralTokenGating(model, 'deit_small', K=3, alpha=5.0, tau_percentile=95)
            cal_indices = get_calibration_indices(42, n_calibration=max(n_cal, 2000))[:n_cal]
            cal_ds = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=cal_indices)
            cal_loader = get_dataloader(cal_ds, batch_size=bs, num_workers=NW)
            stg.calibrate(cal_loader, n_images=n_cal)
            stg.enable()
            clean = eval_clean(model, bs, N_EVAL)
            corr_accs = {}
            for c in REPRESENTATIVE_CORRUPTIONS:
                acc = eval_corruption_sev5(model, c, bs)
                corr_accs[c] = acc['top1']
            mean_acc = np.mean(list(corr_accs.values()))
            results['variants'].append({
                'label': f'n={n_cal}', 'n_calibration': n_cal,
                'clean_top1': clean['top1'],
                'corruption_accs': corr_accs,
                'mean_corruption_acc': float(mean_acc),
            })
            print(f"  [cal_size] n={n_cal}: clean={clean['top1']*100:.1f}%, corr={mean_acc*100:.1f}%")
            stg.disable(); del model, stg; torch.cuda.empty_cache()
        save_results(results, result_path)

    run_ablation('tau',
        [{'K': 3, 'alpha': 5.0, 'tau_percentile': t} for t in [75, 85, 90, 95, 99]],
        [f'tau={t}th' for t in [75, 85, 90, 95, 99]])


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================
def run_spectral_analysis():
    result_path = os.path.join(RESULTS_DIR, 'spectral_analysis.json')
    if os.path.exists(result_path):
        print(f"[SKIP] spectral_analysis")
        return json.load(open(result_path))

    print(f"\n{'='*60}")
    print("Running spectral signature analysis...")

    set_seed(42)
    model, config = load_model('deit_small')
    bs = config['batch_size']
    target_layers = config['default_stg_layers']
    from functools import partial

    def collect_spectral_ratios(model, dataloader, target_layers, K=3, n_images=500):
        stg = SpectralTokenGating(model, 'deit_small', target_layers=target_layers, K=K)
        layer_ratios = {l: [] for l in target_layers}

        def hook_fn(layer_idx, module, input, output):
            tokens = output[:, 1:, :]
            tokens_flat = tokens.reshape(-1, tokens.shape[-1])
            ratios = stg._compute_spectral_ratios(tokens_flat)
            layer_ratios[layer_idx].append(ratios.cpu())

        hooks = []
        blocks = get_model_blocks(model, 'deit_small')
        for l in target_layers:
            h = blocks[l].register_forward_hook(partial(hook_fn, l))
            hooks.append(h)

        n = 0
        with torch.no_grad():
            for images, _ in dataloader:
                if n >= n_images: break
                images = images.to('cuda')
                model(images)
                n += images.shape[0]

        for h in hooks:
            h.remove()
        return {l: torch.cat(layer_ratios[l], dim=0) for l in target_layers}

    results = {'corruptions': {}, 'clean': {}}
    n_analysis = 300
    indices = np.arange(n_analysis)

    print("  Clean reference...")
    ds = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=indices)
    loader = get_dataloader(ds, batch_size=bs, num_workers=NW)
    clean_ratios = collect_spectral_ratios(model, loader, target_layers, K=3, n_images=n_analysis)

    for l in target_layers:
        r = clean_ratios[l]
        results['clean'][str(l)] = {
            'mean': r.mean(dim=0).tolist(),
            'std': r.std(dim=0).tolist(),
        }

    for corruption in CORRUPTION_TYPES:
        print(f"  {corruption}...")
        ds = CorruptedImageNetDataset(DATA_DIR, corruption, 5, indices=indices)
        loader = get_dataloader(ds, batch_size=bs, num_workers=NW)
        corr_ratios = collect_spectral_ratios(model, loader, target_layers, K=3, n_images=n_analysis)

        corr_result = {}
        for l in target_layers:
            r = corr_ratios[l]
            cr = clean_ratios[l]
            shift = (r.mean(dim=0) - cr.mean(dim=0)).tolist()
            mean_clean = cr.mean(dim=0)
            centered = cr - mean_clean
            cov = (centered.T @ centered) / (centered.shape[0] - 1) + 1e-5 * torch.eye(3)
            cov_inv = torch.linalg.inv(cov)
            mahal_clean = torch.sqrt((centered @ cov_inv * centered).sum(dim=-1))
            tau_95 = torch.quantile(mahal_clean, 0.95).item()
            diff_corr = r - mean_clean
            mahal_corr = torch.sqrt((diff_corr @ cov_inv * diff_corr).sum(dim=-1))
            separability = float((mahal_corr > tau_95).float().mean().item())

            corr_result[str(l)] = {
                'mean': r.mean(dim=0).tolist(),
                'shift_from_clean': shift,
                'mean_mahal_distance': float(mahal_corr.mean().item()),
                'separability_at_95pct': separability,
            }
        results['corruptions'][corruption] = corr_result

    # Category summary
    categories = {}
    for corruption in CORRUPTION_TYPES:
        cat = CATEGORY_MAP.get(corruption, 'other')
        if cat not in categories:
            categories[cat] = {'separability': [], 'mahal_dist': []}
        for l in target_layers:
            s = results['corruptions'][corruption][str(l)]
            categories[cat]['separability'].append(s['separability_at_95pct'])
            categories[cat]['mahal_dist'].append(s['mean_mahal_distance'])

    results['category_summary'] = {
        cat: {
            'mean_separability': float(np.mean(v['separability'])),
            'mean_mahal_distance': float(np.mean(v['mahal_dist'])),
        } for cat, v in categories.items()
    }
    save_results(results, result_path)
    del model; torch.cuda.empty_cache()
    return results


# ============================================================
# OVERHEAD MEASUREMENT
# ============================================================
def run_overhead_measurement():
    result_path = os.path.join(RESULTS_DIR, 'overhead.json')
    if os.path.exists(result_path):
        print(f"[SKIP] overhead")
        return json.load(open(result_path))

    print(f"\n{'='*60}")
    print("Measuring computational overhead...")
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
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    start_time = time.time()
    print("="*60)
    print("SPECTRAL TOKEN GATING - EXPERIMENT SUITE")
    print(f"N_EVAL={N_EVAL}, N_CLEAN={N_CLEAN}, severity=5 only for main")
    print("="*60)

    # Step 1: Vanilla baselines
    print("\n>>> STEP 1: Vanilla Baselines <<<")
    for mk in ['deit_small', 'deit_base', 'swin_tiny']:
        run_vanilla_baseline(mk)
    print(f"  Elapsed: {(time.time()-start_time)/60:.1f} min")

    # Step 2: Tent baseline (DeiT-S only for speed)
    print("\n>>> STEP 2: Tent Baseline <<<")
    run_tent_baseline('deit_small')
    print(f"  Elapsed: {(time.time()-start_time)/60:.1f} min")

    # Step 3: STG main experiments
    print("\n>>> STEP 3: STG Main <<<")
    for seed in SEEDS:
        run_stg_experiment('deit_small', seed)
    run_stg_experiment('deit_base', 42)
    run_stg_experiment('swin_tiny', 42)
    print(f"  Elapsed: {(time.time()-start_time)/60:.1f} min")

    # Step 4: Multi-severity analysis
    print("\n>>> STEP 4: Severity Analysis <<<")
    run_severity_analysis()
    print(f"  Elapsed: {(time.time()-start_time)/60:.1f} min")

    # Step 5: Spectral analysis
    print("\n>>> STEP 5: Spectral Analysis <<<")
    run_spectral_analysis()
    print(f"  Elapsed: {(time.time()-start_time)/60:.1f} min")

    # Step 6: Ablation studies
    print("\n>>> STEP 6: Ablation Studies <<<")
    run_all_ablations()
    print(f"  Elapsed: {(time.time()-start_time)/60:.1f} min")

    # Step 7: Overhead
    print("\n>>> STEP 7: Overhead <<<")
    run_overhead_measurement()

    total = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"ALL DONE! Total: {total:.1f} min")
    print(f"{'='*60}")
