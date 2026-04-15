"""
Speed Validation Experiment (RQ1)
Validate <100ms generation time target across difficulty levels.
"""

import sys
sys.path.insert(0, '../shared')

import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

from utils import set_seed, save_json, compute_statistics
from scene_generator import SceneGenerator
from query_generator import QueryGenerator
from answer_computer import AnswerComputer
from scene_generator import scene_to_dict
from difficulty import DifficultyController


def measure_generation_time(
    difficulty_level: int,
    query_type: str,
    num_samples: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Measure generation time for a configuration.
    
    Returns:
        Dictionary with timing statistics
    """
    set_seed(seed)
    
    scene_times = []
    query_times = []
    answer_times = []
    total_times = []
    
    for i in range(num_samples):
        instance_seed = seed + i
        rng = random.Random(instance_seed)
        
        # Scene generation
        start = time.time()
        scene_gen = SceneGenerator(instance_seed)
        object_count = DifficultyController.get_object_count(difficulty_level, instance_seed)
        shapes, relations, svg = scene_gen.generate_scene(object_count)
        scene_time = (time.time() - start) * 1000
        scene_times.append(scene_time)
        
        # Query generation
        start = time.time()
        query_gen = QueryGenerator(instance_seed)
        depth = DifficultyController.get_depth_for_type(difficulty_level, query_type)
        query = query_gen.generate_query(query_type, depth)
        query_time = (time.time() - start) * 1000
        query_times.append(query_time)
        
        # Answer computation
        start = time.time()
        scene_dict = scene_to_dict(shapes, relations)
        computer = AnswerComputer(scene_dict)
        answer = computer.compute_answer(query.program)
        answer_time = (time.time() - start) * 1000
        answer_times.append(answer_time)
        
        total_time = scene_time + query_time + answer_time
        total_times.append(total_time)
    
    results = {
        "difficulty_level": difficulty_level,
        "query_type": query_type,
        "num_samples": num_samples,
        "scene_generation_ms": compute_statistics(scene_times),
        "query_generation_ms": compute_statistics(query_times),
        "answer_computation_ms": compute_statistics(answer_times),
        "total_ms": compute_statistics(total_times),
        "hypothesis_check": {
            "target_met": statistics.mean(total_times) < 100,
            "std_check": statistics.stdev(total_times) < 20 if len(total_times) > 1 else True
        }
    }
    
    return results


def run_speed_validation(output_dir: str = "../../results"):
    """Run full speed validation experiment."""
    
    print("=" * 60)
    print("SPEED VALIDATION EXPERIMENT (RQ1)")
    print("=" * 60)
    
    all_results = []
    
    # Test each difficulty level
    for level in range(1, 5):
        print(f"\nTesting Difficulty Level {level}...")
        result = measure_generation_time(level, "existential", num_samples=250, seed=42 + level)
        all_results.append(result)
        
        print(f"  Mean total time: {result['total_ms']['mean']:.2f}ms")
        print(f"  Std total time: {result['total_ms']['std']:.2f}ms")
        print(f"  Component breakdown:")
        print(f"    Scene: {result['scene_generation_ms']['mean']:.2f}ms")
        print(f"    Query: {result['query_generation_ms']['mean']:.2f}ms")
        print(f"    Answer: {result['answer_computation_ms']['mean']:.2f}ms")
        print(f"  Target met (<100ms): {result['hypothesis_check']['target_met']}")
    
    # Test different query types at level 3
    print(f"\nTesting different query types at Level 3...")
    for qtype in ["existential", "universal", "comparative", "transitive", "nested_quant"]:
        result = measure_generation_time(3, qtype, num_samples=100, seed=100)
        all_results.append(result)
        print(f"  {qtype}: {result['total_ms']['mean']:.2f}ms")
    
    # Save results
    output = {
        "experiment": "speed_validation",
        "hypothesis": "Mean generation time <100ms with std<20ms",
        "results": all_results,
        "overall": {
            "all_targets_met": all(r['hypothesis_check']['target_met'] for r in all_results[:4]),
            "max_mean_ms": max(r['total_ms']['mean'] for r in all_results[:4]),
            "max_std_ms": max(r['total_ms']['std'] for r in all_results[:4])
        }
    }
    
    save_json(output, f"{output_dir}/speed_validation.json")
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASS' if output['overall']['all_targets_met'] else 'FAIL'}")
    print(f"Max mean time: {output['overall']['max_mean_ms']:.2f}ms")
    print(f"Max std: {output['overall']['max_std_ms']:.2f}ms")
    print(f"{'='*60}")
    
    # Create visualization
    create_speed_plot(all_results[:4], "../../figures/speed_breakdown.png")
    
    return output


def create_speed_plot(results: List[Dict], output_path: str):
    """Create speed breakdown visualization."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    levels = [r['difficulty_level'] for r in results]
    scene_times = [r['scene_generation_ms']['mean'] for r in results]
    query_times = [r['query_generation_ms']['mean'] for r in results]
    answer_times = [r['answer_computation_ms']['mean'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(levels))
    width = 0.25
    
    ax.bar(x - width, scene_times, width, label='Scene Generation')
    ax.bar(x, query_times, width, label='Query Generation')
    ax.bar(x + width, answer_times, width, label='Answer Computation')
    
    ax.axhline(y=100, color='r', linestyle='--', label='100ms Target')
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Generation Time Breakdown by Difficulty Level')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Level {l}' for l in levels])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Speed plot saved to {output_path}")


if __name__ == "__main__":
    import random
    run_speed_validation()
