#!/usr/bin/env python3
"""
Aggregate experimental results into final results.json.
"""
import json
import numpy as np
from pathlib import Path

def load_results(path):
    """Load results from a JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def compute_summary(results):
    """Compute summary statistics for results."""
    valid = [r for r in results if 'metrics' in r]
    if not valid:
        return {}
    
    return {
        'n_runs': len(valid),
        'f1': {
            'mean': float(np.mean([r['metrics']['f1'] for r in valid])),
            'std': float(np.std([r['metrics']['f1'] for r in valid]))
        },
        'precision': {
            'mean': float(np.mean([r['metrics']['precision'] for r in valid])),
            'std': float(np.std([r['metrics']['precision'] for r in valid]))
        },
        'recall': {
            'mean': float(np.mean([r['metrics']['recall'] for r in valid])),
            'std': float(np.std([r['metrics']['recall'] for r in valid]))
        },
        'shd': {
            'mean': float(np.mean([r['metrics']['shd'] for r in valid])),
            'std': float(np.std([r['metrics']['shd'] for r in valid]))
        },
        'runtime': {
            'mean': float(np.mean([r['runtime'] for r in valid])),
            'std': float(np.std([r['runtime'] for r in valid]))
        }
    }

def compute_summary_by_nodes(results):
    """Compute summary grouped by number of nodes."""
    by_nodes = {}
    for r in results:
        if 'metrics' not in r:
            continue
        n_nodes = r['config'].get('n_nodes', 0)
        if n_nodes not in by_nodes:
            by_nodes[n_nodes] = []
        by_nodes[n_nodes].append(r)
    
    summary = {}
    for n_nodes, res in sorted(by_nodes.items()):
        summary[f'{n_nodes}_nodes'] = compute_summary(res)
    return summary

def main():
    print("Aggregating experimental results...")
    
    # Load all results
    all_results = {
        'baselines': {},
        'mf_acd': {},
        'ablations': {}
    }
    
    # Baselines
    for baseline in ['pc_fisherz', 'pc_stable', 'fast_pc', 'ges', 'hccd', 'dcilp']:
        path = f'results/baselines/{baseline}/results.json'
        results = load_results(path)
        if results:
            all_results['baselines'][baseline] = {
                'raw': results,
                'summary': compute_summary(results),
                'by_nodes': compute_summary_by_nodes(results)
            }
            print(f"  {baseline}: {len(results)} runs")
    
    # MF-ACD main
    results = load_results('results/mf_acd/main/results.json')
    if results:
        all_results['mf_acd']['main'] = {
            'raw': results,
            'summary': compute_summary(results),
            'by_nodes': compute_summary_by_nodes(results)
        }
        print(f"  mf_acd main: {len(results)} runs")
    
    # MF-ACD real-world
    results = load_results('results/mf_acd/real_world/results.json')
    if results:
        all_results['mf_acd']['real_world'] = {
            'raw': results,
            'summary': compute_summary(results)
        }
        print(f"  mf_acd real_world: {len(results)} runs")
    
    # Ablations
    for ablation in ['fixed_vs_adaptive', 'allocation_sensitivity', 'ugfs_components', 'mtc_comparison']:
        ablation_path = f'results/ablations/{ablation}'
        all_results['ablations'][ablation] = {}
        try:
            for json_file in Path(ablation_path).glob('*.json'):
                results = load_results(json_file)
                if results:
                    variant = json_file.stem
                    all_results['ablations'][ablation][variant] = {
                        'raw': results,
                        'summary': compute_summary(results),
                        'by_nodes': compute_summary_by_nodes(results)
                    }
                    print(f"  {ablation}/{variant}: {len(results)} runs")
        except Exception as e:
            print(f"  {ablation}: no results ({e})")
    
    # Compute key comparisons
    if 'pc_fisherz' in all_results['baselines'] and 'main' in all_results['mf_acd']:
        pc_results = all_results['baselines']['pc_fisherz']['raw']
        mf_results = all_results['mf_acd']['main']['raw']
        
        # Match by config (n_nodes, edge_prob, seed)
        def make_key(r):
            cfg = r.get('config', {})
            return (cfg.get('n_nodes'), cfg.get('edge_prob'), cfg.get('seed'))
        
        pc_by_key = {make_key(r): r for r in pc_results if 'metrics' in r}
        mf_by_key = {make_key(r): r for r in mf_results if 'metrics' in r}
        
        common_keys = set(pc_by_key.keys()) & set(mf_by_key.keys())
        
        if common_keys:
            pc_f1 = [pc_by_key[k]['metrics']['f1'] for k in common_keys]
            mf_f1 = [mf_by_key[k]['metrics']['f1'] for k in common_keys]
            pc_time = [pc_by_key[k]['runtime'] for k in common_keys]
            mf_time = [mf_by_key[k]['runtime'] for k in common_keys]
            
            # Compute savings
            savings = [(pc - mf) / pc * 100 if pc > 0 else 0 for pc, mf in zip(pc_time, mf_time)]
            
            all_results['comparison'] = {
                'n_common': len(common_keys),
                'pc_f1_mean': float(np.mean(pc_f1)),
                'mf_acd_f1_mean': float(np.mean(mf_f1)),
                'f1_difference': float(np.mean(mf_f1) - np.mean(pc_f1)),
                'pc_time_mean': float(np.mean(pc_time)),
                'mf_acd_time_mean': float(np.mean(mf_time)),
                'time_savings_pct': float(np.mean(savings)),
                'cost_savings_pct': float(np.mean([r.get('savings_pct', 0) for r in mf_results if 'savings_pct' in r])) if mf_results else 0
            }
            
            print(f"\nKey findings:")
            print(f"  PC F1: {all_results['comparison']['pc_f1_mean']:.3f}")
            print(f"  MF-ACD F1: {all_results['comparison']['mf_acd_f1_mean']:.3f}")
            print(f"  F1 diff: {all_results['comparison']['f1_difference']:.3f}")
            print(f"  Time savings: {all_results['comparison']['time_savings_pct']:.1f}%")
            print(f"  Cost savings: {all_results['comparison']['cost_savings_pct']:.1f}%")
    
    # Save final results
    # Remove raw data from final results to save space
    final_results = {}
    for key, val in all_results.items():
        if key == 'baselines' or key == 'mf_acd':
            final_results[key] = {}
            for method, data in val.items():
                final_results[key][method] = {
                    'summary': data.get('summary', {}),
                    'by_nodes': data.get('by_nodes', {})
                }
        elif key == 'ablations':
            final_results[key] = {}
            for ablation, variants in val.items():
                final_results[key][ablation] = {}
                for variant, data in variants.items():
                    final_results[key][ablation][variant] = {
                        'summary': data.get('summary', {}),
                        'by_nodes': data.get('by_nodes', {})
                    }
        else:
            final_results[key] = val
    
    with open('results_final.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nResults saved to results_final.json")

if __name__ == '__main__':
    main()
