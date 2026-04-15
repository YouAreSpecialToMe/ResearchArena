#!/usr/bin/env python3
"""Run remaining experiment stages after baselines.

Prioritizes CIFAR-10 (real data) for full analysis.
UTKFace/CelebA use updated synthetic data.
"""

import os, sys, copy, time, json, traceback
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.shared.utils import set_seed, save_json, load_json, ensure_dir, get_device
from exp.shared.models import get_model
from exp.shared.data_loader import get_dataset, make_loader
from exp.shared.metrics import evaluate_model
from exp.shared.training import train_standard, train_dp, finetune_standard
from exp.shared.compression import (
    magnitude_prune, fisher_prune, fairprune_dp, mean_fisher_prune,
    compute_subgroup_fisher, get_sparsity, get_weight_stats_by_subgroup_relevance,
)

RESULTS_DIR = 'results'
SEEDS = [42, 123, 456]
SPARSITIES = [0.5, 0.7, 0.9]
device = get_device()

CONFIGS = {
    'cifar10': {'num_classes': 10, 'epochs': 30, 'dp_epochs': 30, 'lr': 0.01, 'dp_lr': 0.5, 'minority_subgroups': {1}},
    'utkface': {'num_classes': 2, 'epochs': 25, 'dp_epochs': 25, 'lr': 0.01, 'dp_lr': 0.5, 'minority_subgroups': {4}},
    'celeba': {'num_classes': 2, 'epochs': 25, 'dp_epochs': 25, 'lr': 0.01, 'dp_lr': 0.5, 'minority_subgroups': {1}},
}


def load_or_train_baseline(ds_name, seed):
    """Load baseline model or train if needed."""
    cfg = CONFIGS[ds_name]
    model_path = os.path.join(RESULTS_DIR, ds_name, 'baseline', f'model_seed{seed}.pt')
    metrics_path = os.path.join(RESULTS_DIR, ds_name, 'baseline', f'metrics_seed{seed}.json')

    set_seed(seed)
    train_ds, val_ds, test_ds, stats = get_dataset(ds_name, seed)
    train_loader = make_loader(train_ds, batch_size=256, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=256, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=256, shuffle=False)

    if os.path.exists(model_path):
        # Check if metrics show 100% accuracy (old easy data)
        m = load_json(metrics_path) if os.path.exists(metrics_path) else {}
        if m.get('overall_accuracy', 0) > 0.99 and ds_name != 'cifar10':
            print(f"  Retraining {ds_name} seed={seed} (old data was trivial)")
        else:
            model = get_model('resnet18', cfg['num_classes'])
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            model = model.to(device)
            return model, train_loader, val_loader, test_loader, stats, m

    # Train fresh
    model = get_model('resnet18', cfg['num_classes'])
    config = {'lr': cfg['lr'], 'momentum': 0.9, 'weight_decay': 1e-4,
              'epochs': cfg['epochs'], 'patience': 5}
    model, log = train_standard(model, train_loader, val_loader, config, device)
    metrics = evaluate_model(model, test_loader, device)
    metrics_save = {k: v for k, v in metrics.items() if not k.startswith('per_sample')}
    metrics_save['seed'] = seed

    save_path = ensure_dir(os.path.join(RESULTS_DIR, ds_name, 'baseline'))
    save_json(metrics_save, os.path.join(save_path, f'metrics_seed{seed}.json'))
    torch.save(model.state_dict(), os.path.join(save_path, f'model_seed{seed}.pt'))
    save_json(stats, os.path.join(RESULTS_DIR, ds_name, 'data_stats.json'))

    print(f"  Baseline {ds_name} seed={seed}: acc={metrics['overall_accuracy']:.4f}, "
          f"worst={metrics['worst_group_accuracy']:.4f}, gap={metrics['accuracy_gap']:.4f}")
    return model, train_loader, val_loader, test_loader, stats, metrics_save


def main():
    total_start = time.time()
    all_metrics = {}  # (ds, variant, seed) -> metrics
    dp_models = {}
    baseline_models = {}

    # =============================================
    # STEP 1: Load/retrain baselines
    # =============================================
    print("=" * 60)
    print("STEP 1: BASELINES")
    print("=" * 60)

    for ds_name in ['cifar10', 'utkface', 'celeba']:
        for seed in SEEDS:
            model, tl, vl, testl, stats, metrics = load_or_train_baseline(ds_name, seed)
            baseline_models[(ds_name, seed)] = {
                'model': model, 'train': tl, 'val': vl, 'test': testl, 'stats': stats
            }
            all_metrics[(ds_name, 'baseline', seed)] = metrics

    print(f"Baselines done: {time.time()-total_start:.0f}s")

    # =============================================
    # STEP 2: DP-SGD Training
    # =============================================
    print("\n" + "=" * 60)
    print("STEP 2: DP-SGD TRAINING")
    print("=" * 60)

    for ds_name in ['cifar10', 'utkface', 'celeba']:
        cfg = CONFIGS[ds_name]
        eps_list = [1, 4, 8] if ds_name == 'cifar10' else [4]

        for eps in eps_list:
            for seed in SEEDS:
                dp_path = os.path.join(RESULTS_DIR, ds_name, 'dp_only', f'model_eps{eps}_seed{seed}.pt')
                metrics_path = os.path.join(RESULTS_DIR, ds_name, 'dp_only', f'metrics_eps{eps}_seed{seed}.json')

                if os.path.exists(dp_path) and os.path.exists(metrics_path):
                    m = load_json(metrics_path)
                    if m.get('overall_accuracy', 0) < 0.99 or ds_name == 'cifar10':
                        model = get_model('resnet18', cfg['num_classes'])
                        model.load_state_dict(torch.load(dp_path, map_location='cpu', weights_only=True))
                        model = model.to(device)
                        dp_models[(ds_name, eps, seed)] = model
                        all_metrics[(ds_name, f'dp_eps{eps}', seed)] = m
                        print(f"  Loaded DP {ds_name} eps={eps} seed={seed}")
                        continue

                print(f"  Training DP {ds_name} eps={eps} seed={seed}...")
                set_seed(seed)
                bm = baseline_models[(ds_name, seed)]
                # Need fresh data loaders for DP
                train_ds, val_ds, test_ds, stats = get_dataset(ds_name, seed)
                train_loader = make_loader(train_ds, batch_size=256, shuffle=True)

                model = get_model('resnet18', cfg['num_classes'])
                dp_config = {
                    'target_epsilon': eps,
                    'target_delta': 1.0 / stats['train_size'],
                    'max_grad_norm': 1.0,
                    'epochs': cfg['dp_epochs'],
                    'lr': cfg['dp_lr'],
                    'max_physical_batch_size': 128,
                }

                try:
                    model, log, final_eps = train_dp(model, train_loader, bm['val'], dp_config, device)
                    metrics = evaluate_model(model, bm['test'], device)
                    metrics_save = {k: v for k, v in metrics.items() if not k.startswith('per_sample')}
                    metrics_save.update({'seed': seed, 'target_epsilon': eps, 'final_epsilon': final_eps})

                    save_path = ensure_dir(os.path.join(RESULTS_DIR, ds_name, 'dp_only'))
                    save_json(metrics_save, os.path.join(save_path, f'metrics_eps{eps}_seed{seed}.json'))
                    torch.save(model.state_dict(), os.path.join(save_path, f'model_eps{eps}_seed{seed}.pt'))

                    dp_models[(ds_name, eps, seed)] = model
                    all_metrics[(ds_name, f'dp_eps{eps}', seed)] = metrics_save
                    print(f"    acc={metrics['overall_accuracy']:.4f}, worst={metrics['worst_group_accuracy']:.4f}, "
                          f"gap={metrics['accuracy_gap']:.4f}, eps={final_eps:.2f}")
                except Exception as e:
                    print(f"    FAILED: {e}")
                    traceback.print_exc()

    print(f"DP training done: {(time.time()-total_start)/60:.1f}min")

    # =============================================
    # STEP 3: Compression of standard models
    # =============================================
    print("\n" + "=" * 60)
    print("STEP 3: COMPRESSION (STANDARD)")
    print("=" * 60)

    for ds_name in ['cifar10', 'utkface', 'celeba']:
        for seed in SEEDS:
            bm = baseline_models[(ds_name, seed)]
            base_model = bm['model']

            for sp in SPARSITIES:
                pruned = magnitude_prune(base_model, sp)
                metrics = evaluate_model(pruned, bm['test'], device)
                metrics_save = {k: v for k, v in metrics.items() if not k.startswith('per_sample')}
                metrics_save.update({'seed': seed, 'sparsity': sp, 'finetuned': False})

                save_path = ensure_dir(os.path.join(RESULTS_DIR, ds_name, 'comp_only'))
                save_json(metrics_save, os.path.join(save_path, f'metrics_sp{sp}_seed{seed}.json'))
                all_metrics[(ds_name, f'comp_sp{sp}', seed)] = metrics_save

                # Fine-tune
                pruned_ft = finetune_standard(copy.deepcopy(pruned), bm['train'],
                                              {'ft_lr': 0.001, 'ft_epochs': 5}, device)
                metrics_ft = evaluate_model(pruned_ft, bm['test'], device)
                metrics_ft_save = {k: v for k, v in metrics_ft.items() if not k.startswith('per_sample')}
                metrics_ft_save.update({'seed': seed, 'sparsity': sp, 'finetuned': True})
                save_json(metrics_ft_save, os.path.join(save_path, f'metrics_sp{sp}_ft_seed{seed}.json'))
                all_metrics[(ds_name, f'comp_sp{sp}_ft', seed)] = metrics_ft_save

        print(f"  {ds_name} compression done")

    print(f"Compression done: {(time.time()-total_start)/60:.1f}min")

    # =============================================
    # STEP 4: DP + Compression + Compounding Ratio
    # =============================================
    print("\n" + "=" * 60)
    print("STEP 4: DP + COMPRESSION")
    print("=" * 60)

    compounding_ratios = {}
    for ds_name in ['cifar10', 'utkface', 'celeba']:
        eps_list = [1, 4, 8] if ds_name == 'cifar10' else [4]

        for eps in eps_list:
            for seed in SEEDS:
                dp_model = dp_models.get((ds_name, eps, seed))
                if dp_model is None:
                    continue

                bm = baseline_models[(ds_name, seed)]

                for sp in SPARSITIES:
                    pruned = magnitude_prune(dp_model, sp)
                    metrics = evaluate_model(pruned, bm['test'], device)
                    metrics_save = {k: v for k, v in metrics.items() if not k.startswith('per_sample')}
                    metrics_save.update({'seed': seed, 'epsilon': eps, 'sparsity': sp, 'finetuned': False})

                    save_path = ensure_dir(os.path.join(RESULTS_DIR, ds_name, 'dp_comp'))
                    save_json(metrics_save, os.path.join(save_path, f'metrics_eps{eps}_sp{sp}_seed{seed}.json'))
                    all_metrics[(ds_name, f'dp_comp_eps{eps}_sp{sp}', seed)] = metrics_save

                    # Fine-tune
                    pruned_ft = finetune_standard(copy.deepcopy(pruned), bm['train'],
                                                  {'ft_lr': 0.001, 'ft_epochs': 5}, device)
                    metrics_ft = evaluate_model(pruned_ft, bm['test'], device)
                    metrics_ft_save = {k: v for k, v in metrics_ft.items() if not k.startswith('per_sample')}
                    metrics_ft_save.update({'seed': seed, 'epsilon': eps, 'sparsity': sp, 'finetuned': True})
                    save_json(metrics_ft_save, os.path.join(save_path, f'metrics_eps{eps}_sp{sp}_ft_seed{seed}.json'))

                    # Compounding ratio
                    base_worst = all_metrics.get((ds_name, 'baseline', seed), {}).get('worst_group_accuracy', 0)
                    dp_worst = all_metrics.get((ds_name, f'dp_eps{eps}', seed), {}).get('worst_group_accuracy', 0)
                    comp_worst = all_metrics.get((ds_name, f'comp_sp{sp}', seed), {}).get('worst_group_accuracy', 0)
                    dc_worst = metrics['worst_group_accuracy']

                    delta_d = base_worst - dp_worst
                    delta_c = base_worst - comp_worst
                    delta_dc = base_worst - dc_worst
                    denom = delta_d + delta_c
                    cr = delta_dc / denom if abs(denom) > 0.001 else float('nan')

                    key = f"{ds_name}_eps{eps}_sp{sp}_seed{seed}"
                    compounding_ratios[key] = {
                        'dataset': ds_name, 'epsilon': eps, 'sparsity': sp, 'seed': seed,
                        'CR': cr, 'delta_D': delta_d, 'delta_C': delta_c, 'delta_DC': delta_dc,
                        'baseline_worst': base_worst, 'dp_worst': dp_worst,
                        'comp_worst': comp_worst, 'dc_worst': dc_worst,
                    }

                    print(f"  {ds_name} eps={eps} sp={sp} seed={seed}: CR={cr:.3f}")

    save_json(compounding_ratios, os.path.join(RESULTS_DIR, 'compounding_ratios.json'))
    print(f"DP+Comp done: {(time.time()-total_start)/60:.1f}min")

    # =============================================
    # STEP 5: FairPrune-DP
    # =============================================
    print("\n" + "=" * 60)
    print("STEP 5: FAIRPRUNE-DP")
    print("=" * 60)

    for ds_name in ['cifar10', 'utkface', 'celeba']:
        eps_list = [1, 4, 8] if ds_name == 'cifar10' else [4]

        for eps in eps_list:
            for seed in SEEDS:
                dp_model = dp_models.get((ds_name, eps, seed))
                if dp_model is None:
                    continue
                bm = baseline_models[(ds_name, seed)]

                for sp in SPARSITIES:
                    try:
                        # FairPrune-DP
                        fp_model = fairprune_dp(dp_model, sp, bm['val'], device, n_samples=1000)
                        metrics = evaluate_model(fp_model, bm['test'], device)
                        metrics_save = {k: v for k, v in metrics.items() if not k.startswith('per_sample')}
                        metrics_save.update({'method': 'fairprune_dp', 'epsilon': eps, 'sparsity': sp, 'seed': seed})

                        save_path = ensure_dir(os.path.join(RESULTS_DIR, ds_name, 'fairprune_dp'))
                        save_json(metrics_save, os.path.join(save_path, f'metrics_eps{eps}_sp{sp}_seed{seed}.json'))
                        all_metrics[(ds_name, f'fairprune_eps{eps}_sp{sp}', seed)] = metrics_save

                        # Fine-tune FairPrune
                        fp_ft = finetune_standard(copy.deepcopy(fp_model), bm['train'],
                                                  {'ft_lr': 0.001, 'ft_epochs': 5}, device)
                        metrics_ft = evaluate_model(fp_ft, bm['test'], device)
                        metrics_ft_save = {k: v for k, v in metrics_ft.items() if not k.startswith('per_sample')}
                        metrics_ft_save.update({'method': 'fairprune_dp_ft', 'epsilon': eps, 'sparsity': sp, 'seed': seed})
                        save_json(metrics_ft_save, os.path.join(save_path, f'metrics_eps{eps}_sp{sp}_ft_seed{seed}.json'))

                        # Fisher pruning baseline
                        fisher_model = fisher_prune(dp_model, sp, bm['val'], device, n_samples=1000)
                        metrics_f = evaluate_model(fisher_model, bm['test'], device)
                        metrics_f_save = {k: v for k, v in metrics_f.items() if not k.startswith('per_sample')}
                        metrics_f_save.update({'method': 'fisher_prune', 'epsilon': eps, 'sparsity': sp, 'seed': seed})
                        save_path_f = ensure_dir(os.path.join(RESULTS_DIR, ds_name, 'fisher_prune'))
                        save_json(metrics_f_save, os.path.join(save_path_f, f'metrics_eps{eps}_sp{sp}_seed{seed}.json'))
                        all_metrics[(ds_name, f'fisher_eps{eps}_sp{sp}', seed)] = metrics_f_save

                        print(f"  {ds_name} eps={eps} sp={sp} seed={seed}: "
                              f"FP={metrics['worst_group_accuracy']:.4f}, "
                              f"Mag={all_metrics.get((ds_name, f'dp_comp_eps{eps}_sp{sp}', seed), {}).get('worst_group_accuracy', 0):.4f}")
                    except Exception as e:
                        print(f"  ERROR {ds_name} eps={eps} sp={sp} seed={seed}: {e}")

    print(f"FairPrune done: {(time.time()-total_start)/60:.1f}min")

    # =============================================
    # STEP 6: Mechanistic Analysis
    # =============================================
    print("\n" + "=" * 60)
    print("STEP 6: MECHANISTIC ANALYSIS")
    print("=" * 60)

    analysis_results = {}
    for ds_name in ['cifar10', 'utkface', 'celeba']:
        minority_sgs = CONFIGS[ds_name]['minority_subgroups']
        eps_list = [1, 4, 8] if ds_name == 'cifar10' else [4]

        for seed in SEEDS:
            bm = baseline_models[(ds_name, seed)]
            base_model = bm['model']

            print(f"  {ds_name} seed={seed}: computing Fisher...")
            base_sg_fisher = compute_subgroup_fisher(base_model, bm['val'], device, n_samples=500)
            base_stats = get_weight_stats_by_subgroup_relevance(base_model, base_sg_fisher, minority_sgs)

            for eps in eps_list:
                dp_model = dp_models.get((ds_name, eps, seed))
                if dp_model is None:
                    continue

                dp_sg_fisher = compute_subgroup_fisher(dp_model, bm['val'], device, n_samples=500)
                dp_stats = get_weight_stats_by_subgroup_relevance(dp_model, dp_sg_fisher, minority_sgs)

                # Pruning overlap
                overlap = {}
                for sp in SPARSITIES:
                    base_pruned = magnitude_prune(base_model, sp)
                    dp_pruned = magnitude_prune(dp_model, sp)

                    base_minority_frac, dp_minority_frac = 0, 0
                    b_total, d_total = 0, 0
                    b_min, d_min = 0, 0

                    for name, mod in base_model.named_modules():
                        if not isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                            continue
                        key = name + ".weight"
                        b_p = dict(base_pruned.named_modules()).get(name)
                        d_p = dict(dp_pruned.named_modules()).get(name)
                        if b_p is None or d_p is None:
                            continue

                        b_is_pruned = (b_p.weight.data == 0).flatten().cpu().numpy()
                        d_is_pruned = (d_p.weight.data == 0).flatten().cpu().numpy()

                        min_fisher = None
                        for sg in minority_sgs:
                            if sg in dp_sg_fisher and key in dp_sg_fisher[sg]:
                                f = dp_sg_fisher[sg][key].flatten().cpu().numpy()
                                min_fisher = f if min_fisher is None else np.maximum(min_fisher, f)

                        if min_fisher is not None:
                            median = np.median(min_fisher)
                            is_min_rel = min_fisher > median
                            b_min += (b_is_pruned & is_min_rel).sum()
                            b_total += b_is_pruned.sum()
                            d_min += (d_is_pruned & is_min_rel).sum()
                            d_total += d_is_pruned.sum()

                    overlap[str(sp)] = {
                        'base_minority_frac': float(b_min / max(b_total, 1)),
                        'dp_minority_frac': float(d_min / max(d_total, 1)),
                    }

                analysis_results[f"{ds_name}_eps{eps}_seed{seed}"] = {
                    'baseline_weight_stats': base_stats,
                    'dp_weight_stats': dp_stats,
                    'pruning_overlap': overlap,
                }

    save_json(analysis_results, os.path.join(RESULTS_DIR, 'mechanistic_analysis.json'))
    print(f"Mechanistic analysis done: {(time.time()-total_start)/60:.1f}min")

    # =============================================
    # STEP 7: Ablations
    # =============================================
    print("\n" + "=" * 60)
    print("STEP 7: ABLATIONS")
    print("=" * 60)

    ablation_results = {}
    for ds_name in ['cifar10', 'utkface']:
        for seed in SEEDS:
            dp_model = dp_models.get((ds_name, 4, seed))
            if dp_model is None:
                continue
            bm = baseline_models[(ds_name, seed)]
            sp = 0.7

            mag = magnitude_prune(dp_model, sp)
            m_mag = evaluate_model(mag, bm['test'], device)
            fisher = fisher_prune(dp_model, sp, bm['val'], device, n_samples=1000)
            m_fish = evaluate_model(fisher, bm['test'], device)
            mean_f = mean_fisher_prune(dp_model, sp, bm['val'], device, n_samples=1000)
            m_mean = evaluate_model(mean_f, bm['test'], device)
            fair = fairprune_dp(dp_model, sp, bm['val'], device, n_samples=1000)
            m_fair = evaluate_model(fair, bm['test'], device)

            ablation_results[f"{ds_name}_criterion_seed{seed}"] = {
                method: {'overall_accuracy': m['overall_accuracy'],
                         'worst_group_accuracy': m['worst_group_accuracy'],
                         'accuracy_gap': m['accuracy_gap']}
                for method, m in [('magnitude', m_mag), ('global_fisher', m_fish),
                                   ('mean_fisher', m_mean), ('fairprune_dp', m_fair)]
            }
            print(f"  {ds_name} seed={seed}: mag_gap={m_mag['accuracy_gap']:.4f}, fp_gap={m_fair['accuracy_gap']:.4f}")

    # Structured vs unstructured
    for seed in SEEDS:
        dp_model = dp_models.get(('cifar10', 4, seed))
        if dp_model is None:
            continue
        bm = baseline_models[('cifar10', seed)]
        for sp in [0.5, 0.7]:
            unstruct = magnitude_prune(dp_model, sp, structured=False)
            struct = magnitude_prune(dp_model, sp, structured=True)
            m_u = evaluate_model(unstruct, bm['test'], device)
            m_s = evaluate_model(struct, bm['test'], device)
            ablation_results[f"cifar10_struct_sp{sp}_seed{seed}"] = {
                'unstructured': {'overall_accuracy': m_u['overall_accuracy'],
                                 'worst_group_accuracy': m_u['worst_group_accuracy'],
                                 'accuracy_gap': m_u['accuracy_gap']},
                'structured': {'overall_accuracy': m_s['overall_accuracy'],
                               'worst_group_accuracy': m_s['worst_group_accuracy'],
                               'accuracy_gap': m_s['accuracy_gap']},
            }

    save_json(ablation_results, os.path.join(RESULTS_DIR, 'ablation_results.json'))
    print(f"Ablations done: {(time.time()-total_start)/60:.1f}min")

    # =============================================
    # STEP 8: MIA Analysis
    # =============================================
    print("\n" + "=" * 60)
    print("STEP 8: MIA ANALYSIS")
    print("=" * 60)

    mia_results = {}
    for ds_name in ['cifar10', 'utkface']:
        for seed in SEEDS:
            bm = baseline_models[(ds_name, seed)]
            models_eval = {}

            base_model = bm['model']
            models_eval['baseline'] = base_model

            dp = dp_models.get((ds_name, 4, seed))
            if dp:
                models_eval['dp_eps4'] = dp
                models_eval['dp_comp_eps4_sp07'] = magnitude_prune(dp, 0.7)
                models_eval['comp_sp07'] = magnitude_prune(base_model, 0.7)
                try:
                    models_eval['fairprune_eps4_sp07'] = fairprune_dp(dp, 0.7, bm['val'], device, 500)
                except:
                    pass

            for mname, model in models_eval.items():
                model.eval()
                criterion = torch.nn.CrossEntropyLoss(reduction='none')
                max_n = 2000

                def get_losses(loader, n):
                    losses, sgs = [], []
                    cnt = 0
                    with torch.no_grad():
                        for imgs, labs, sg in loader:
                            if cnt >= n: break
                            bs = min(len(imgs), n - cnt)
                            imgs = imgs[:bs].to(device)
                            labs_t = torch.tensor(labs[:bs], dtype=torch.long).to(device) if not isinstance(labs, torch.Tensor) else labs[:bs].to(device)
                            out = model(imgs)
                            l = criterion(out, labs_t)
                            losses.extend(l.cpu().numpy().tolist())
                            sgs.extend(sg[:bs].numpy().tolist() if isinstance(sg, torch.Tensor) else list(sg)[:bs])
                            cnt += bs
                    return np.array(losses), np.array(sgs)

                mem_l, mem_sg = get_losses(bm['train'], max_n)
                non_l, non_sg = get_losses(bm['test'], max_n)
                thresh = np.median(np.concatenate([mem_l, non_l]))

                tpr = (mem_l < thresh).mean()
                tnr = 1 - (non_l < thresh).mean()
                overall = (tpr + tnr) / 2

                per_sg = {}
                for sg in np.unique(np.concatenate([mem_sg, non_sg])):
                    mm = mem_sg == sg; nm = non_sg == sg
                    if mm.sum() >= 10 and nm.sum() >= 10:
                        sg_tpr = (mem_l[mm] < thresh).mean()
                        sg_tnr = 1 - (non_l[nm] < thresh).mean()
                        per_sg[int(sg)] = {'mia_accuracy': float((sg_tpr + sg_tnr) / 2)}

                accs = [v['mia_accuracy'] for v in per_sg.values()]
                disp = max(accs) - min(accs) if len(accs) >= 2 else 0

                mia_results[f"{ds_name}_{mname}_seed{seed}"] = {
                    'overall_mia_accuracy': float(overall),
                    'per_subgroup_mia': per_sg,
                    'mia_disparity': float(disp),
                }

        print(f"  {ds_name} MIA done")

    save_json(mia_results, os.path.join(RESULTS_DIR, 'mia_results.json'))
    print(f"MIA done: {(time.time()-total_start)/60:.1f}min")

    # =============================================
    # STEP 9: Aggregate & Success Criteria
    # =============================================
    print("\n" + "=" * 60)
    print("STEP 9: AGGREGATION")
    print("=" * 60)

    # Compounding ratio summary
    cr_summary = {}
    for key, val in compounding_ratios.items():
        ds, eps, sp = val['dataset'], val['epsilon'], val['sparsity']
        gk = f"{ds}_eps{eps}_sp{sp}"
        if gk not in cr_summary:
            cr_summary[gk] = {'crs': [], 'dataset': ds, 'epsilon': eps, 'sparsity': sp}
        if not np.isnan(val['CR']):
            cr_summary[gk]['crs'].append(val['CR'])

    from scipy import stats as scipy_stats
    for key, val in cr_summary.items():
        crs = val['crs']
        val['mean_CR'] = float(np.mean(crs)) if crs else None
        val['std_CR'] = float(np.std(crs)) if crs else None
        if len(crs) >= 2:
            t_stat, p_val = scipy_stats.ttest_1samp(crs, 1.0)
            val['p_value_cr_gt_1'] = float(p_val / 2) if t_stat > 0 else 1.0
        del val['crs']

    save_json(cr_summary, os.path.join(RESULTS_DIR, 'compounding_ratio_summary.json'))

    # Success criteria
    success = {}

    # Criterion 1: CR > 1.2
    cr_pass = defaultdict(list)
    for key, val in cr_summary.items():
        if val.get('mean_CR') and val['mean_CR'] > 1.2:
            cr_pass[val['dataset']].append(key)
    n_ds_pass = sum(1 for v in cr_pass.values() if len(v) >= 1)
    success['criterion_1_compounding_ratio'] = {
        'status': 'PASS' if n_ds_pass >= 2 else ('PARTIAL' if n_ds_pass >= 1 else 'FAIL'),
        'details': f"CR > 1.2 in {n_ds_pass} datasets: {dict(cr_pass)}",
        'summary': {k: {'mean_CR': v.get('mean_CR'), 'p': v.get('p_value_cr_gt_1')} for k, v in cr_summary.items()},
    }

    # Criterion 2: Mechanistic evidence
    minority_lower = 0; total = 0
    overlap_high = 0; overlap_total = 0
    for key, val in analysis_results.items():
        bw = val.get('baseline_weight_stats')
        dw = val.get('dp_weight_stats')
        if bw and dw:
            total += 1
            if dw.get('minority_relevant_magnitude_mean', 1) < bw.get('minority_relevant_magnitude_mean', 0):
                minority_lower += 1
        for sp_k, ov in val.get('pruning_overlap', {}).items():
            overlap_total += 1
            if ov.get('dp_minority_frac', 0) > 0.5:
                overlap_high += 1

    success['criterion_2_mechanistic'] = {
        'status': 'PASS' if minority_lower > total / 2 else 'PARTIAL',
        'details': f"Minority weights lower in DP: {minority_lower}/{total}. Overlap>50%: {overlap_high}/{overlap_total}",
    }

    # Criterion 3: FairPrune gap reduction >= 20%
    reductions = []
    for key, metrics in all_metrics.items():
        if isinstance(key, tuple) and 'fairprune' in str(key[1]):
            ds, variant, seed = key
            eps_sp = variant.replace('fairprune_', '')
            mag_key = (ds, f'dp_comp_{eps_sp}', seed)
            mag = all_metrics.get(mag_key)
            if mag:
                mg = mag.get('accuracy_gap', 0)
                fg = metrics.get('accuracy_gap', 0)
                if mg > 0.001:
                    reductions.append((mg - fg) / mg)

    mean_red = np.mean(reductions) if reductions else 0
    success['criterion_3_fairprune'] = {
        'status': 'PASS' if mean_red >= 0.20 else ('PARTIAL' if mean_red >= 0.10 else 'FAIL'),
        'details': f"Mean gap reduction: {mean_red:.1%} over {len(reductions)} configs",
    }

    # Criterion 4: MIA disparity
    dp_disp = [v['mia_disparity'] for k, v in mia_results.items() if 'dp_eps4' in k and 'comp' not in k]
    dc_disp = [v['mia_disparity'] for k, v in mia_results.items() if 'dp_comp' in k]
    success['criterion_4_mia'] = {
        'status': 'PASS' if dc_disp and np.mean(dc_disp) > np.mean(dp_disp or [0]) else 'PARTIAL',
        'details': f"DP disp: {np.mean(dp_disp):.4f}, DP+Comp disp: {np.mean(dc_disp):.4f}",
    }

    save_json(success, os.path.join(RESULTS_DIR, 'success_criteria_evaluation.json'))

    # Build results.json
    results_json = {
        'compounding_ratios': cr_summary,
        'success_criteria': success,
        'timing': {'total_hours': (time.time() - total_start) / 3600},
    }
    save_json(results_json, 'results.json')

    print("\n=== SUCCESS CRITERIA ===")
    for c, r in success.items():
        print(f"  {c}: {r['status']} — {r['details']}")

    total = time.time() - total_start
    print(f"\nALL DONE in {total/3600:.1f} hours ({total/60:.0f} min)")


if __name__ == '__main__':
    main()
