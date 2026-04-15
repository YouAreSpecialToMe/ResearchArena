"""
Curriculum Learning Feasibility (Reduced Scope)
Demonstrate 20K instance generation in <15 minutes.
"""

import sys
sys.path.insert(0, '../shared')

import time
from utils import save_json
from generate_dataset import generate_dataset


def run_curriculum_feasibility(output_dir="../../results"):
    """Run curriculum learning feasibility demonstration."""
    
    print("=" * 60)
    print("CURRICULUM LEARNING FEASIBILITY")
    print("=" * 60)
    print("\nGenerating 20K training instances...")
    
    start_time = time.time()
    
    # Generate Level 1 training data: 10,000 instances
    print("\n1. Generating Level 1 training data (10,000 instances)...")
    dataset_l1 = generate_dataset(
        count=10000,
        difficulty_level=1,
        query_type="existential",
        seed=1000,
        output_dir="../../data/curriculum"
    )
    
    # Generate Level 4 training data: 10,000 instances
    print("\n2. Generating Level 4 training data (10,000 instances)...")
    dataset_l4 = generate_dataset(
        count=10000,
        difficulty_level=4,
        query_type="existential",
        seed=2000,
        output_dir="../../data/curriculum"
    )
    
    elapsed = time.time() - start_time
    
    results = {
        "experiment": "curriculum_feasibility",
        "target": "20K instances in <15 minutes",
        "total_instances": 20000,
        "generation_time_seconds": elapsed,
        "generation_time_minutes": elapsed / 60,
        "throughput_instances_per_second": 20000 / elapsed,
        "throughput_instances_per_hour": 20000 / (elapsed / 3600),
        "target_met": elapsed < 15 * 60,
        "breakdown": {
            "level1_time_ms": dataset_l1["metadata"]["timings"]["total_ms"],
            "level4_time_ms": dataset_l4["metadata"]["timings"]["total_ms"]
        },
        "note": "This demonstrates the ENABLING capability for curriculum learning, not full training. Full LoRA fine-tuning deferred to future work."
    }
    
    save_json(results, f"{output_dir}/curriculum_feasibility.json")
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Total time: {elapsed/60:.2f} minutes")
    print(f"  Throughput: {results['throughput_instances_per_hour']:.0f} instances/hour")
    print(f"  Target met (<15 min): {results['target_met']}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    run_curriculum_feasibility()
