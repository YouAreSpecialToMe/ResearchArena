#!/usr/bin/env python3
"""Master script to run all experiments and generate results."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01')

import os
import json
import time
import subprocess
from pathlib import Path


def run_experiment(script_path, description):
    """Run a single experiment script."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*70}")
    
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd='/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01',
            capture_output=False,
            text=True,
            timeout=7200  # 2 hour timeout per experiment
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"✓ {description} completed in {elapsed/60:.1f} minutes")
            return True, elapsed
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            return False, elapsed
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out after 2 hours")
        return False, 7200
    except Exception as e:
        print(f"✗ {description} failed with error: {e}")
        return False, 0


def aggregate_all_results():
    """Aggregate results from all experiments into final results.json."""
    print("\n" + "="*70)
    print("Aggregating all results...")
    print("="*70)
    
    results_dir = Path("exp")
    
    # Load individual results
    all_results = {}
    
    experiments = [
        ("baseline_activation", "Activation Baseline"),
        ("baseline_output_score", "Output Score Baseline"),
        ("fidelity_weighted_steering", "Fidelity-Weighted Steering"),
        ("ablation", "Component Ablation"),
        ("side_effects", "Side Effects"),
    ]
    
    for exp_dir, exp_name in experiments:
        result_file = results_dir / exp_dir / "results.json"
        if result_file.exists():
            try:
                with open(result_file) as f:
                    all_results[exp_dir] = json.load(f)
                print(f"  ✓ Loaded {exp_name}")
            except Exception as e:
                print(f"  ✗ Failed to load {exp_name}: {e}")
        else:
            print(f"  ✗ {exp_name} results not found")
    
    # Create aggregated summary
    summary = {
        "experiment_summary": {
            "title": "Measuring and Mitigating the Causal-Semantic Disconnect in Sparse Autoencoders",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "GPT-2 Small (124M)",
            "sae": "SAELens gpt2-small-res-jb layer 8",
            "experiments_completed": len(all_results)
        }
    }
    
    # Extract IFS statistics if available
    if "fidelity_weighted_steering" in all_results:
        fws = all_results["fidelity_weighted_steering"]
        if "ifs_statistics" in fws:
            summary["ifs_statistics"] = fws["ifs_statistics"]
    
    # Extract steering comparison
    steering_comparison = {
        "fidelity_weighted": {},
        "activation_baseline": {},
        "output_score_baseline": {}
    }
    
    if "fidelity_weighted_steering" in all_results:
        fws = all_results["fidelity_weighted_steering"]
        if "aggregated" in fws:
            steering_comparison["fidelity_weighted"] = fws["aggregated"]
    
    if "baseline_activation" in all_results:
        act = all_results["baseline_activation"]
        if "aggregated" in act:
            steering_comparison["activation_baseline"] = act["aggregated"]
    
    if "baseline_output_score" in all_results:
        out = all_results["baseline_output_score"]
        if "aggregated" in out:
            steering_comparison["output_score_baseline"] = out["aggregated"]
    
    summary["steering_comparison"] = steering_comparison
    
    # Extract component ablation
    if "ablation" in all_results:
        abl = all_results["ablation"]
        if "aggregated" in abl:
            summary["component_ablation"] = abl["aggregated"]
    
    # Extract side effects
    if "side_effects" in all_results:
        se = all_results["side_effects"]
        if "aggregated" in se:
            summary["side_effects"] = se["aggregated"]
        if "side_effects" in se:
            summary["side_effect_scores"] = se["side_effects"]
        if "comparison" in se:
            summary["side_effect_comparison"] = se["comparison"]
    
    # Save aggregated results
    with open("results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to results.json")
    return summary


def main():
    print("="*70)
    print("Running All Experiments - Fixed Version")
    print("="*70)
    
    start_time = time.time()
    
    # Ensure data is prepared first
    print("\nPreparing datasets...")
    from exp.shared.data_loader import prepare_all_datasets
    prepare_all_datasets()
    
    # Define experiments to run
    experiments = [
        ("exp/baseline_activation/run.py", "Activation Baseline"),
        ("exp/baseline_output_score/run.py", "Output Score Baseline"),
        ("exp/fidelity_weighted_steering/run.py", "Fidelity-Weighted Steering"),
        ("exp/ablation/run.py", "Component Ablation"),
        ("exp/side_effects/run.py", "Side Effects Measurement"),
    ]
    
    # Run each experiment
    results = {}
    for script, desc in experiments:
        success, elapsed = run_experiment(script, desc)
        results[desc] = {"success": success, "time": elapsed}
    
    # Aggregate results
    summary = aggregate_all_results()
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"\nExperiments completed:")
    for desc, result in results.items():
        status = "✓" if result["success"] else "✗"
        print(f"  {status} {desc}: {result['time']/60:.1f} min")
    
    # Print key results if available
    if "steering_comparison" in summary:
        print("\n" + "="*70)
        print("KEY RESULTS")
        print("="*70)
        
        sc = summary["steering_comparison"]
        print("\nSteering Effectiveness (k=20):")
        for method in ["fidelity_weighted", "activation_baseline", "output_score_baseline"]:
            if method in sc and "k=20" in sc[method]:
                agg = sc[method]["k=20"]
                mean = agg.get("target_change_mean", 0)
                std = agg.get("target_change_std", 0)
                print(f"  {method:25s}: {mean:.4f} ± {std:.4f}")
    
    print("\n" + "="*70)
    print("All experiments complete!")
    print("="*70)


if __name__ == "__main__":
    main()
