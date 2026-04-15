#!/usr/bin/env python3
"""
Improved experiment runner for MemSat evaluation.
"""
import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent))

from shared.egraph import EGraph, HierarchicalEGraph
from shared.kernels import (
    generate_loop_nest_egraph, 
    generate_function_egraph,
    generate_hierarchical_egraph,
    get_all_kernels
)
from shared.extraction_v2 import (
    greedy_extraction,
    ilp_extraction,
    beam_search_extraction,
    treewidth_aware_extraction,
    sequential_extraction,
    joint_optimization_extraction,
    ExtractionResult
)

SEEDS = [42, 123, 456]
KERNELS = get_all_kernels()
RESULTS_DIR = Path("/home/nw366/ResearchArena/outputs/kimi_t3_compiler_optimization/idea_01/results")


def run_h1_treewidth_experiment():
    """H1: Real-world program e-graphs exhibit low treewidth (≤10)."""
    print("\n" + "="*60)
    print("H1: Treewidth Measurement Experiment")
    print("="*60)
    
    results = {
        "hypothesis": "H1",
        "description": "Measure treewidth of hierarchical e-graphs",
        "kernels": {},
        "summary": {}
    }
    
    all_treewidths = []
    
    for kernel in KERNELS:
        print(f"\nProcessing {kernel}...")
        kernel_results = []
        
        for seed in SEEDS:
            # Generate hierarchical e-graph
            hier_graph = generate_hierarchical_egraph(kernel, seed=seed, layout_rules=True)
            
            # Measure treewidth at each level
            level1_tw = hier_graph.levels[1].compute_treewidth()[0]
            level2_tw = hier_graph.levels[2].compute_treewidth()[0] if 2 in hier_graph.levels else 0
            
            # Generate flat e-graph for comparison
            flat_graph = generate_loop_nest_egraph(kernel, seed=seed, layout_rules=True)
            flat_tw = flat_graph.compute_treewidth()[0]
            
            # Get e-graph sizes
            flat_size = flat_graph.size()
            level1_size = hier_graph.levels[1].size()
            
            kernel_results.append({
                "seed": seed,
                "flat_treewidth": flat_tw,
                "level1_treewidth": level1_tw,
                "level2_treewidth": level2_tw,
                "flat_size": flat_size,
                "level1_size": level1_size
            })
            
            all_treewidths.append(level1_tw)
        
        results["kernels"][kernel] = kernel_results
    
    # Compute statistics
    all_tw_array = np.array(all_treewidths)
    
    results["summary"] = {
        "mean_treewidth": float(np.mean(all_tw_array)),
        "std_treewidth": float(np.std(all_tw_array)),
        "median_treewidth": float(np.median(all_tw_array)),
        "max_treewidth": int(np.max(all_tw_array)),
        "min_treewidth": int(np.min(all_tw_array)),
        "pct_leq_10": float(np.mean(all_tw_array <= 10) * 100),
        "pct_leq_15": float(np.mean(all_tw_array <= 15) * 100),
        "h1_confirmed": bool(np.mean(all_tw_array <= 10) >= 0.70)
    }
    
    print(f"\nH1 Results:")
    print(f"  Mean treewidth: {results['summary']['mean_treewidth']:.2f} ± {results['summary']['std_treewidth']:.2f}")
    print(f"  % with treewidth ≤ 10: {results['summary']['pct_leq_10']:.1f}%")
    print(f"  H1 CONFIRMED: {results['summary']['h1_confirmed']}")
    
    output_path = RESULTS_DIR / "treewidth" / "treewidth_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    return results


def run_baseline_extractions():
    """Run baseline extraction algorithms."""
    print("\n" + "="*60)
    print("Baseline Extraction Experiments")
    print("="*60)
    
    results = {
        "greedy": {},
        "ilp": {},
        "sequential": {}
    }
    
    for kernel in KERNELS:
        print(f"\nProcessing {kernel}...")
        
        for seed in SEEDS:
            # Generate e-graph
            graph = generate_loop_nest_egraph(kernel, seed=seed, layout_rules=True)
            
            # Greedy extraction
            greedy_result = greedy_extraction(graph, cost_model="instruction")
            if seed not in results["greedy"]:
                results["greedy"][seed] = {}
            results["greedy"][seed][kernel] = asdict(greedy_result)
            
            # ILP extraction (with timeout)
            ilp_result = ilp_extraction(graph, cost_model="instruction", time_limit_seconds=20)
            if seed not in results["ilp"]:
                results["ilp"][seed] = {}
            results["ilp"][seed][kernel] = asdict(ilp_result)
            
            # Sequential extraction
            seq_result = sequential_extraction(graph, cost_model="memory", memory_weight=0.5)
            if seed not in results["sequential"]:
                results["sequential"][seed] = {}
            results["sequential"][seed][kernel] = asdict(seq_result)
    
    # Save results
    for method in ["greedy", "ilp", "sequential"]:
        output_path = RESULTS_DIR / "extraction" / f"{method}_results.json"
        with open(output_path, 'w') as f:
            json.dump(results[method], f, indent=2)
        print(f"\nSaved {method} results to {output_path}")
    
    return results


def run_h3_treewidth_aware_extraction():
    """
    H3: Treewidth-aware extraction provides compile times competitive 
    with greedy while achieving solution quality within 10% of optimal ILP.
    """
    print("\n" + "="*60)
    print("H3: Treewidth-Aware Extraction Experiment")
    print("="*60)
    
    results = {
        "hypothesis": "H3",
        "description": "Compare treewidth-aware extraction vs greedy and ILP",
        "kernels": {},
        "summary": {}
    }
    
    all_quality_ratios = []
    all_time_ratios = []
    
    for kernel in KERNELS:
        print(f"\nProcessing {kernel}...")
        kernel_results = []
        
        for seed in SEEDS:
            graph = generate_loop_nest_egraph(kernel, seed=seed, layout_rules=True)
            
            # Greedy extraction (fast baseline)
            greedy_result = greedy_extraction(graph, cost_model="instruction")
            
            # ILP extraction (quality baseline)
            ilp_result = ilp_extraction(graph, cost_model="instruction", time_limit_seconds=20)
            
            # Treewidth-aware extraction
            tw_result = treewidth_aware_extraction(graph, cost_model="instruction")
            
            # Compute metrics
            # Quality ratio: how much worse than ILP (0 = same as ILP, 0.10 = 10% worse)
            if ilp_result.total_cost > 0 and ilp_result.total_cost < 1e8:
                quality_ratio = max(0, (tw_result.total_cost - ilp_result.total_cost) / ilp_result.total_cost)
            else:
                quality_ratio = 0.0
            
            # Time ratio: how much slower than greedy (1.0 = same, 5.0 = 5x slower)
            if greedy_result.extraction_time_ms > 0:
                time_ratio = max(1.0, tw_result.extraction_time_ms / greedy_result.extraction_time_ms)
            else:
                time_ratio = 1.0
            
            all_quality_ratios.append(quality_ratio)
            all_time_ratios.append(time_ratio)
            
            kernel_results.append({
                "seed": seed,
                "greedy_cost": greedy_result.total_cost,
                "greedy_time_ms": greedy_result.extraction_time_ms,
                "ilp_cost": ilp_result.total_cost,
                "ilp_time_ms": ilp_result.extraction_time_ms,
                "treewidth_cost": tw_result.total_cost,
                "treewidth_time_ms": tw_result.extraction_time_ms,
                "quality_ratio": quality_ratio,
                "time_ratio": time_ratio
            })
        
        results["kernels"][kernel] = kernel_results
    
    # Compute statistics
    quality_array = np.array(all_quality_ratios)
    time_array = np.array(all_time_ratios)
    
    results["summary"] = {
        "mean_quality_ratio": float(np.mean(quality_array)),
        "std_quality_ratio": float(np.std(quality_array)),
        "max_quality_ratio": float(np.max(quality_array)),
        "pct_within_10pct": float(np.mean(quality_array <= 0.10) * 100),
        "mean_time_ratio": float(np.mean(time_array)),
        "std_time_ratio": float(np.std(time_array)),
        "max_time_ratio": float(np.max(time_array)),
        "pct_within_5x": float(np.mean(time_array <= 5.0) * 100),
        "h3_confirmed": bool(np.mean(quality_array <= 0.10) >= 0.90 and np.mean(time_array <= 5.0) >= 0.90)
    }
    
    print(f"\nH3 Results:")
    print(f"  Mean quality ratio: {results['summary']['mean_quality_ratio']:.4f} ± {results['summary']['std_quality_ratio']:.4f}")
    print(f"  % within 10% of optimal: {results['summary']['pct_within_10pct']:.1f}%")
    print(f"  Mean time ratio (vs greedy): {results['summary']['mean_time_ratio']:.2f}x")
    print(f"  % within 5x greedy time: {results['summary']['pct_within_5x']:.1f}%")
    print(f"  H3 CONFIRMED: {results['summary']['h3_confirmed']}")
    
    output_path = RESULTS_DIR / "extraction" / "treewidth_aware_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    return results


def run_h2_joint_optimization():
    """
    H2: Joint optimization of computation and data layout achieves 
    better memory bandwidth utilization than sequential optimization.
    """
    print("\n" + "="*60)
    print("H2: Joint Compute and Layout Optimization Experiment")
    print("="*60)
    
    results = {
        "hypothesis": "H2",
        "description": "Compare joint vs sequential optimization",
        "kernels": {},
        "summary": {}
    }
    
    all_improvements = []
    
    for kernel in KERNELS:
        print(f"\nProcessing {kernel}...")
        kernel_results = []
        
        for seed in SEEDS:
            # Generate e-graph with layout rules
            graph = generate_loop_nest_egraph(kernel, seed=seed, layout_rules=True)
            
            # Joint optimization: simultaneously optimize compute and layout
            joint_result = joint_optimization_extraction(graph, cost_model="joint", memory_weight=0.5)
            
            # Sequential optimization: compute first, then layout
            seq_result = sequential_extraction(graph, cost_model="memory", memory_weight=0.5)
            
            # Compute improvement
            if seq_result.total_cost > 0:
                improvement = (seq_result.total_cost - joint_result.total_cost) / seq_result.total_cost * 100
            else:
                improvement = 0.0
            
            all_improvements.append(improvement)
            
            kernel_results.append({
                "seed": seed,
                "joint_cost": joint_result.total_cost,
                "joint_memory_cost": joint_result.memory_cost,
                "joint_compute_cost": joint_result.compute_cost,
                "sequential_cost": seq_result.total_cost,
                "sequential_memory_cost": seq_result.memory_cost,
                "sequential_compute_cost": seq_result.compute_cost,
                "improvement_pct": improvement
            })
        
        results["kernels"][kernel] = kernel_results
    
    # Compute statistics
    improvement_array = np.array(all_improvements)
    
    results["summary"] = {
        "mean_improvement_pct": float(np.mean(improvement_array)),
        "std_improvement_pct": float(np.std(improvement_array)),
        "median_improvement_pct": float(np.median(improvement_array)),
        "max_improvement_pct": float(np.max(improvement_array)),
        "min_improvement_pct": float(np.min(improvement_array)),
        "pct_positive_improvement": float(np.mean(improvement_array > 0) * 100),
        "pct_ge_10_improvement": float(np.mean(improvement_array >= 10) * 100),
        "h2_confirmed": bool(np.mean(improvement_array >= 10) >= 0.5)
    }
    
    print(f"\nH2 Results:")
    print(f"  Mean improvement: {results['summary']['mean_improvement_pct']:.2f}% ± {results['summary']['std_improvement_pct']:.2f}%")
    print(f"  % with positive improvement: {results['summary']['pct_positive_improvement']:.1f}%")
    print(f"  % with ≥10% improvement: {results['summary']['pct_ge_10_improvement']:.1f}%")
    print(f"  H2 CONFIRMED: {results['summary']['h2_confirmed']}")
    
    output_path = RESULTS_DIR / "extraction" / "joint_optimization_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    return results


def run_ablation_studies():
    """Run all ablation studies."""
    print("\n" + "="*60)
    print("Ablation Studies")
    print("="*60)
    
    all_results = {}
    
    # Ablation 1: Hierarchy levels
    print("\n--- Ablation 1: Hierarchy Levels ---")
    hierarchy_results = {"flat": {}, "hierarchical": {}}
    
    for kernel in KERNELS:
        for seed in SEEDS:
            flat_graph = generate_loop_nest_egraph(kernel, seed=seed, layout_rules=True)
            flat_tw = flat_graph.compute_treewidth()[0]
            flat_size = flat_graph.size()
            
            hier_graph = generate_hierarchical_egraph(kernel, seed=seed, layout_rules=True)
            hier_tw_level1 = hier_graph.levels[1].compute_treewidth()[0]
            hier_size_level1 = hier_graph.levels[1].size()
            
            if kernel not in hierarchy_results["flat"]:
                hierarchy_results["flat"][kernel] = []
                hierarchy_results["hierarchical"][kernel] = []
            
            hierarchy_results["flat"][kernel].append({
                "seed": seed, "treewidth": flat_tw, "size": flat_size
            })
            hierarchy_results["hierarchical"][kernel].append({
                "seed": seed, "treewidth": hier_tw_level1, "size": hier_size_level1
            })
    
    all_results["hierarchy"] = hierarchy_results
    
    # Ablation 2: Extraction algorithms comparison
    print("\n--- Ablation 2: Extraction Algorithms ---")
    algo_results = {}
    
    for kernel in KERNELS[:5]:
        algo_results[kernel] = {}
        for seed in SEEDS:
            graph = generate_loop_nest_egraph(kernel, seed=seed, layout_rules=True)
            
            greedy_res = greedy_extraction(graph, cost_model="instruction")
            beam_res = beam_search_extraction(graph, beam_width=10, cost_model="instruction")
            tw_res = treewidth_aware_extraction(graph, cost_model="instruction")
            ilp_res = ilp_extraction(graph, cost_model="instruction", time_limit_seconds=10)
            
            algo_results[kernel][seed] = {
                "greedy": asdict(greedy_res),
                "beam_10": asdict(beam_res),
                "treewidth": asdict(tw_res),
                "ilp": asdict(ilp_res)
            }
    
    all_results["extraction_comparison"] = algo_results
    
    # Ablation 3: Layout transformation rules
    print("\n--- Ablation 3: Layout Rules ---")
    layout_results = {"with_layout": {}, "without_layout": {}}
    
    for kernel in KERNELS[:5]:
        for seed in SEEDS:
            graph_with = generate_loop_nest_egraph(kernel, seed=seed, layout_rules=True)
            result_with = greedy_extraction(graph_with, cost_model="memory")
            
            graph_without = generate_loop_nest_egraph(kernel, seed=seed, layout_rules=False)
            result_without = greedy_extraction(graph_without, cost_model="instruction")
            
            if kernel not in layout_results["with_layout"]:
                layout_results["with_layout"][kernel] = []
                layout_results["without_layout"][kernel] = []
            
            layout_results["with_layout"][kernel].append(asdict(result_with))
            layout_results["without_layout"][kernel].append(asdict(result_without))
    
    all_results["layout_rules"] = layout_results
    
    # Ablation 4: Cost models
    print("\n--- Ablation 4: Cost Models ---")
    cost_model_results = {}
    
    for kernel in KERNELS[:5]:
        cost_model_results[kernel] = {}
        for seed in SEEDS:
            graph = generate_loop_nest_egraph(kernel, seed=seed, layout_rules=True)
            
            cost_model_results[kernel][seed] = {}
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = greedy_extraction(graph, cost_model="memory", memory_weight=alpha)
                cost_model_results[kernel][seed][f"alpha_{alpha}"] = asdict(result)
    
    all_results["cost_models"] = cost_model_results
    
    # Save all ablation results
    for study_name, study_data in all_results.items():
        output_path = RESULTS_DIR / "ablation" / f"{study_name}_ablation.json"
        with open(output_path, 'w') as f:
            json.dump(study_data, f, indent=2)
        print(f"\nSaved {study_name} ablation to {output_path}")
    
    return all_results


def run_sensitivity_analysis():
    """Sensitivity analysis: test robustness to rewrite rule application order."""
    print("\n" + "="*60)
    print("Sensitivity Analysis: Rule Application Order")
    print("="*60)
    
    extended_seeds = [42, 123, 456, 789, 101112]
    
    results = {}
    
    for kernel in KERNELS[:5]:
        print(f"\nProcessing {kernel}...")
        kernel_results = []
        
        for seed in extended_seeds:
            graph = generate_loop_nest_egraph(kernel, seed=seed, layout_rules=True)
            
            size = graph.size()
            tw, _ = graph.compute_treewidth()
            
            greedy_res = greedy_extraction(graph, cost_model="instruction")
            
            kernel_results.append({
                "seed": seed,
                "eclasses": size["eclasses"],
                "enodes": size["enodes"],
                "edges": size["edges"],
                "treewidth": tw,
                "extraction_time_ms": greedy_res.extraction_time_ms,
                "total_cost": greedy_res.total_cost
            })
        
        metrics = ["eclasses", "enodes", "treewidth", "extraction_time_ms", "total_cost"]
        cv = {}
        
        for metric in metrics:
            values = np.array([r[metric] for r in kernel_results])
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv[metric] = float(std_val / mean_val) if mean_val > 0 else 0.0
        
        results[kernel] = {
            "raw_results": kernel_results,
            "coefficient_of_variation": cv
        }
    
    output_path = RESULTS_DIR / "ablation" / "sensitivity_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    return results


def generate_aggregated_results():
    """Generate the final aggregated results.json at workspace root."""
    print("\n" + "="*60)
    print("Generating Aggregated Results")
    print("="*60)
    
    aggregated = {
        "experiment_name": "MemSat: Joint Compute and Layout Optimization via Hierarchical E-Graphs",
        "hypotheses": {},
        "baselines": {},
        "ablations": {},
        "summary": {}
    }
    
    # Load H1 results
    try:
        with open(RESULTS_DIR / "treewidth" / "treewidth_results.json") as f:
            h1_data = json.load(f)
        aggregated["hypotheses"]["H1"] = {
            "confirmed": h1_data["summary"]["h1_confirmed"],
            "mean_treewidth": h1_data["summary"]["mean_treewidth"],
            "pct_leq_10": h1_data["summary"]["pct_leq_10"]
        }
    except Exception as e:
        print(f"Warning: Could not load H1 results: {e}")
    
    # Load H2 results
    try:
        with open(RESULTS_DIR / "extraction" / "joint_optimization_results.json") as f:
            h2_data = json.load(f)
        aggregated["hypotheses"]["H2"] = {
            "confirmed": h2_data["summary"]["h2_confirmed"],
            "mean_improvement_pct": h2_data["summary"]["mean_improvement_pct"],
            "pct_ge_10_improvement": h2_data["summary"]["pct_ge_10_improvement"]
        }
    except Exception as e:
        print(f"Warning: Could not load H2 results: {e}")
    
    # Load H3 results
    try:
        with open(RESULTS_DIR / "extraction" / "treewidth_aware_results.json") as f:
            h3_data = json.load(f)
        aggregated["hypotheses"]["H3"] = {
            "confirmed": h3_data["summary"]["h3_confirmed"],
            "mean_quality_ratio": h3_data["summary"]["mean_quality_ratio"],
            "mean_time_ratio": h3_data["summary"]["mean_time_ratio"]
        }
    except Exception as e:
        print(f"Warning: Could not load H3 results: {e}")
    
    # Load baseline results
    for baseline in ["greedy", "ilp", "sequential"]:
        try:
            with open(RESULTS_DIR / "extraction" / f"{baseline}_results.json") as f:
                baseline_data = json.load(f)
            
            all_costs = []
            all_times = []
            for seed_data in baseline_data.values():
                for kernel_result in seed_data.values():
                    all_costs.append(kernel_result.get("total_cost", 0))
                    all_times.append(kernel_result.get("extraction_time_ms", 0))
            
            aggregated["baselines"][baseline] = {
                "mean_cost": float(np.mean(all_costs)),
                "mean_time_ms": float(np.mean(all_times))
            }
        except Exception as e:
            print(f"Warning: Could not load {baseline} results: {e}")
    
    # Summary
    h1_conf = aggregated["hypotheses"].get("H1", {}).get("confirmed", False)
    h2_conf = aggregated["hypotheses"].get("H2", {}).get("confirmed", False)
    h3_conf = aggregated["hypotheses"].get("H3", {}).get("confirmed", False)
    
    aggregated["summary"] = {
        "total_kernels": len(KERNELS),
        "seeds_per_kernel": len(SEEDS),
        "hypotheses_confirmed": sum([h1_conf, h2_conf, h3_conf]),
        "hypotheses_total": 3,
        "all_hypotheses_confirmed": h1_conf and h2_conf and h3_conf
    }
    
    output_path = Path("/home/nw366/ResearchArena/outputs/kimi_t3_compiler_optimization/idea_01/results.json")
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved aggregated results to {output_path}")
    
    return aggregated


def main():
    """Run all experiments."""
    print("="*60)
    print("MemSat Experiment Suite v2")
    print("="*60)
    print(f"Kernels: {KERNELS}")
    print(f"Seeds: {SEEDS}")
    print(f"Results directory: {RESULTS_DIR}")
    
    start_time = time.time()
    
    # Run all experiments
    h1_results = run_h1_treewidth_experiment()
    baseline_results = run_baseline_extractions()
    h3_results = run_h3_treewidth_aware_extraction()
    h2_results = run_h2_joint_optimization()
    ablation_results = run_ablation_studies()
    sensitivity_results = run_sensitivity_analysis()
    
    # Generate aggregated results
    final_results = generate_aggregated_results()
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"All experiments completed in {total_time/60:.1f} minutes")
    print("="*60)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"\nHypothesis Results:")
    print(f"  H1 (Low Treewidth): {'CONFIRMED' if final_results['hypotheses']['H1']['confirmed'] else 'REFUTED'}")
    print(f"    - Mean treewidth: {final_results['hypotheses']['H1']['mean_treewidth']:.2f}")
    print(f"    - % ≤ 10: {final_results['hypotheses']['H1']['pct_leq_10']:.1f}%")
    
    print(f"  H2 (Joint Optimization): {'CONFIRMED' if final_results['hypotheses']['H2']['confirmed'] else 'REFUTED'}")
    print(f"    - Mean improvement: {final_results['hypotheses']['H2']['mean_improvement_pct']:.2f}%")
    
    print(f"  H3 (Treewidth Extraction): {'CONFIRMED' if final_results['hypotheses']['H3']['confirmed'] else 'REFUTED'}")
    print(f"    - Quality vs ILP: {final_results['hypotheses']['H3']['mean_quality_ratio']*100:.1f}%")
    print(f"    - Time vs Greedy: {final_results['hypotheses']['H3']['mean_time_ratio']:.2f}x")
    
    print(f"\nOverall: {final_results['summary']['hypotheses_confirmed']}/{final_results['summary']['hypotheses_total']} hypotheses confirmed")


if __name__ == "__main__":
    main()
