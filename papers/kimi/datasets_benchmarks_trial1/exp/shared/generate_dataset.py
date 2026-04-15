"""
Main dataset generation script for CompViz.
"""

import argparse
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from scene_generator import SceneGenerator, scene_to_dict
from query_generator import QueryGenerator
from answer_computer import AnswerComputer
from difficulty import DifficultyController
from utils import set_seed, save_json, Timer


def generate_instance(
    difficulty_level: int,
    query_type: str,
    seed: Optional[int] = None,
    verify_unique: bool = True
) -> Dict[str, Any]:
    """
    Generate a single CompViz instance.
    
    Args:
        difficulty_level: 1-4
        query_type: Type of query to generate
        seed: Random seed
        verify_unique: Whether to verify answer is unique and not vacuous
    
    Returns:
        Dictionary with scene, query, answer, and metadata
    """
    rng = random.Random(seed)
    
    # Get difficulty parameters
    object_count = DifficultyController.get_object_count(difficulty_level, seed)
    depth = DifficultyController.get_depth_for_type(difficulty_level, query_type)
    
    # Generate scene
    scene_gen = SceneGenerator(seed)
    shapes, relations, svg_string = scene_gen.generate_scene(
        num_objects=object_count,
        min_separation=20.0
    )
    scene_dict = scene_to_dict(shapes, relations)
    
    # Generate query
    query_gen = QueryGenerator(seed)
    query = query_gen.generate_query(query_type, depth, scene_dict)
    
    # Compute answer
    computer = AnswerComputer(scene_dict)
    answer = computer.compute_answer(query.program)
    answer_str = computer.answer_to_string(answer)
    
    # Verify non-vacuous
    if verify_unique and answer_str in ["error", "unknown"]:
        # Regenerate if invalid
        return generate_instance(difficulty_level, query_type, seed + 1000 if seed else 1000, verify_unique)
    
    return {
        "scene": scene_dict,
        "query": {
            "text": query.text,
            "program": query.program,
            "type": query.query_type,
            "depth": query.difficulty_depth
        },
        "answer": answer_str,
        "difficulty_level": difficulty_level,
        "object_count": object_count,
        "seed": seed
    }


def generate_dataset(
    count: int,
    difficulty_level: int,
    query_type: str,
    seed: int = 42,
    output_dir: str = "data/scenes"
) -> Dict[str, Any]:
    """
    Generate a dataset of CompViz instances.
    
    Args:
        count: Number of instances to generate
        difficulty_level: 1-4
        query_type: Type of query
        seed: Base random seed
        output_dir: Directory to save outputs
    
    Returns:
        Dataset metadata
    """
    set_seed(seed)
    
    instances = []
    timings = []
    
    print(f"Generating {count} instances (difficulty={difficulty_level}, type={query_type}, seed={seed})")
    
    for i in range(count):
        instance_seed = seed + i
        
        start = time.time()
        instance = generate_instance(difficulty_level, query_type, instance_seed)
        elapsed_ms = (time.time() - start) * 1000
        timings.append(elapsed_ms)
        
        instances.append(instance)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{count} instances (mean time: {sum(timings[-100:])/100:.2f}ms)")
    
    # Compute timing statistics
    timing_stats = {
        "mean_ms": sum(timings) / len(timings),
        "std_ms": (sum((t - sum(timings)/len(timings))**2 for t in timings) / len(timings))**0.5,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "total_ms": sum(timings)
    }
    
    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_file = output_path / f"level{difficulty_level}_{query_type}_n{count}_s{seed}.json"
    
    dataset = {
        "metadata": {
            "count": count,
            "difficulty_level": difficulty_level,
            "query_type": query_type,
            "seed": seed,
            "timings": timing_stats
        },
        "instances": instances
    }
    
    save_json(dataset, str(dataset_file))
    
    print(f"Dataset saved to {dataset_file}")
    print(f"Timing: mean={timing_stats['mean_ms']:.2f}ms, std={timing_stats['std_ms']:.2f}ms")
    
    return dataset


def generate_mixed_dataset(
    counts_per_config: List[Tuple[int, int, str]],  # (count, level, type)
    seed: int = 42,
    output_dir: str = "data/scenes"
) -> Dict[str, Any]:
    """
    Generate a mixed dataset with multiple configurations.
    
    Args:
        counts_per_config: List of (count, difficulty_level, query_type) tuples
        seed: Base random seed
        output_dir: Output directory
    
    Returns:
        Dataset metadata
    """
    set_seed(seed)
    
    all_instances = []
    total_timings = []
    
    for count, level, qtype in counts_per_config:
        dataset = generate_dataset(count, level, qtype, seed, output_dir)
        all_instances.extend(dataset["instances"])
        total_timings.append(dataset["metadata"]["timings"]["total_ms"])
        seed += count
    
    # Save combined dataset
    combined = {
        "metadata": {
            "total_count": len(all_instances),
            "configs": counts_per_config,
            "seed": seed,
            "total_generation_time_ms": sum(total_timings)
        },
        "instances": all_instances
    }
    
    output_path = Path(output_dir)
    combined_file = output_path / f"mixed_n{len(all_instances)}_s{seed}.json"
    save_json(combined, str(combined_file))
    
    return combined


def main():
    parser = argparse.ArgumentParser(description="Generate CompViz datasets")
    parser.add_argument("--count", type=int, required=True, help="Number of instances")
    parser.add_argument("--difficulty", type=int, default=1, help="Difficulty level (1-4)")
    parser.add_argument("--type", type=str, default="existential", 
                       choices=["existential", "universal", "comparative", "transitive", "nested_quant"],
                       help="Query type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/scenes", help="Output directory")
    parser.add_argument("--mixed", action="store_true", help="Generate mixed dataset")
    
    args = parser.parse_args()
    
    if args.mixed:
        # Generate mixed dataset with all types
        configs = [
            (args.count // 5, args.difficulty, "existential"),
            (args.count // 5, args.difficulty, "universal"),
            (args.count // 5, args.difficulty, "comparative"),
            (args.count // 5, args.difficulty, "transitive"),
            (args.count - 4*(args.count//5), args.difficulty, "nested_quant"),
        ]
        generate_mixed_dataset(configs, args.seed, args.output)
    else:
        generate_dataset(args.count, args.difficulty, args.type, args.seed, args.output)


if __name__ == "__main__":
    main()
