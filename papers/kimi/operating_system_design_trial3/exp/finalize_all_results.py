"""
Finalize all results and generate comprehensive results.json
"""

import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp')

import os
import json
import numpy as np

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main():
    print("=" * 60)
    print("FINALIZING ALL RESULTS")
    print("=" * 60)
    
    # Load all experiment results
    kaphe = load_json('/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/kaphe/summary_v3.json')
    baseline_default = load_json('/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/baseline_default/summary.json')
    baseline_expert = load_json('/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/baseline_expert/summary.json')
    baseline_mlkaps = load_json('/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/baseline_mlkaps/summary.json')
    ablation_knn = load_json('/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/ablation_knn/summary.json')
    ablation_no_char = load_json('/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/ablation_no_char/summary.json')
    ablation_scaling = load_json('/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/ablation_scaling/summary.json')
    real_validation = load_json('/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/real_validation/summary.json')
    cross_workload = load_json('/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/cross_workload/summary.json')
    
    # Build comprehensive results
    results = {
        "experiment": "kaphe_full_study_v3",
        "methodology_note": "Using Decision Tree instead of RIPPER for rule extraction (adjusted from original plan)",
        "methods": {
            "Baseline_Default": {
                "experiment": "baseline_default",
                "metrics": baseline_default['metrics'],
                "config": baseline_default['config'],
                "num_workloads": baseline_default['num_workloads']
            },
            "Baseline_Expert": {
                "experiment": "baseline_expert",
                "metrics": baseline_expert['metrics'],
                "coverage": baseline_expert['coverage'],
                "num_workloads": baseline_expert['num_workloads'],
                "rule_matches": baseline_expert['rule_matches']
            },
            "Baseline_MLKAPS": {
                "experiment": "baseline_mlkaps",
                "decision_tree": baseline_mlkaps['decision_tree'],
                "gradient_boosting": baseline_mlkaps['gradient_boosting'],
                "num_workloads": baseline_mlkaps['num_workloads']
            },
            "KAPHE": {
                "experiment": "kaphe_v3",
                "methodology": "Decision Tree-based rule extraction (not RIPPER)",
                "aggregated_metrics": kaphe['aggregated_metrics'],
                "interpretability": kaphe['interpretability'],
                "seeds": kaphe['seeds'],
                "all_seed_results": kaphe['all_seed_results'],
                "dataset_info": kaphe['dataset_info']
            },
            "Ablation_kNN": {
                "experiment": "ablation_knn",
                "aggregated_metrics": ablation_knn['aggregated_metrics'],
                "k": ablation_knn['k'],
                "interpretability": ablation_knn['interpretability'],
                "comparison_to_kaphe": ablation_knn['comparison_to_kaphe']
            },
            "Ablation_No_Char": {
                "experiment": "ablation_no_characterization",
                "aggregated_metrics": ablation_no_char['aggregated_metrics'],
                "note": ablation_no_char['note'],
                "comparison_to_kaphe": ablation_no_char['comparison_to_kaphe']
            },
            "Ablation_Scaling": {
                "experiment": "ablation_scaling",
                "results": ablation_scaling['results'],
                "seed": ablation_scaling['seed']
            },
            "Real_Validation": {
                "experiment": "real_kernel_validation",
                "results": real_validation['results'],
                "summary": real_validation['summary'],
                "note": real_validation['note']
            },
            "Cross_Workload": {
                "experiment": "cross_workload_generalization",
                "generalization_matrix": cross_workload['generalization_matrix'],
                "summary": cross_workload['summary']
            }
        },
        "summary": {
            "kaphe_mean_score": kaphe['aggregated_metrics']['mean_normalized_score']['mean'],
            "kaphe_std_score": kaphe['aggregated_metrics']['mean_normalized_score']['std'],
            "kaphe_within_10pct": kaphe['aggregated_metrics']['within_10pct']['mean'],
            "kaphe_num_rules": kaphe['interpretability']['num_rules'],
            "kaphe_avg_rule_length": kaphe['interpretability']['avg_rule_length'],
            "kaphe_avg_confidence": kaphe['interpretability']['avg_confidence'],
            "kaphe_vs_default_improvement": (
                (kaphe['aggregated_metrics']['mean_normalized_score']['mean'] - baseline_default['metrics']['mean_normalized_score'])
                / baseline_default['metrics']['mean_normalized_score'] * 100
            ),
            "kaphe_vs_expert_improvement": (
                (kaphe['aggregated_metrics']['mean_normalized_score']['mean'] - baseline_expert['metrics']['mean_normalized_score'])
                / baseline_expert['metrics']['mean_normalized_score'] * 100
            ),
            "kaphe_vs_mlkaps_difference": (
                (kaphe['aggregated_metrics']['mean_normalized_score']['mean'] - baseline_mlkaps['decision_tree']['metrics']['mean_normalized_score'])
                / baseline_mlkaps['decision_tree']['metrics']['mean_normalized_score'] * 100
            ),
            "primary_success_criterion_met": kaphe['aggregated_metrics']['within_10pct']['mean'] >= 80,
            "interpretability_challenges": "Rules have low sample counts due to small dataset (240 samples / 20 configs = 12 samples per config)",
            "methodology_adjustment": "Switched from RIPPER to Decision Tree due to wurlitzer library issues"
        }
    }
    
    # Save results
    output_path = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults Summary:")
    print(f"  KAPHE mean score: {results['summary']['kaphe_mean_score']:.4f} ± {results['summary']['kaphe_std_score']:.4f}")
    print(f"  Within 10% of optimal: {results['summary']['kaphe_within_10pct']:.1f}%")
    print(f"  Number of rules: {results['summary']['kaphe_num_rules']}")
    print(f"  Avg rule length: {results['summary']['kaphe_avg_rule_length']:.2f}")
    print(f"  vs Default improvement: {results['summary']['kaphe_vs_default_improvement']:.1f}%")
    print(f"  vs Expert improvement: {results['summary']['kaphe_vs_expert_improvement']:.1f}%")
    print(f"  vs MLKAPS difference: {results['summary']['kaphe_vs_mlkaps_difference']:.2f}%")
    print(f"\n  Primary criterion met: {results['summary']['primary_success_criterion_met']}")
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
