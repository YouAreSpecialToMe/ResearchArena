#!/usr/bin/env python3
"""
SkillStack Benchmark Generator

Generates benchmark instances at three composition levels:
- Level 1: 8 single skills × 50 instances = 400
- Level 2: 28 pairwise combinations × 50 instances = 1400
- Level 3: 56 triple combinations × 50 instances = 2800
Total: 4600 instances per seed
"""
import json
import os
import sys
import random
from itertools import combinations
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from exp.generators.skills import SKILL_GENERATORS, SKILL_CODES, SKILL_NAMES
from exp.generators.composer import compose_pairwise, compose_triple


def generate_benchmark(seed: int, instances_per_category: int = 50,
                       output_dir: str = "data", difficulty: str = "medium"):
    """Generate a full benchmark set."""
    rng = random.Random(seed)
    all_instances = []
    instance_id = 0
    stats = {"seed": seed, "difficulty": difficulty, "categories": {}}

    # Level 1: Single skills
    print(f"Generating Level 1 (single skills)...")
    for skill_code in SKILL_CODES:
        cat_name = skill_code
        stats["categories"][cat_name] = {"level": 1, "count": 0}
        for i in range(instances_per_category):
            try:
                q, a, atype, meta = SKILL_GENERATORS[skill_code](rng, difficulty)
                instance = {
                    "id": f"s{seed}_{instance_id:05d}",
                    "skill_combo": cat_name,
                    "skills": [skill_code],
                    "level": 1,
                    "question": q,
                    "answer": a,
                    "answer_type": atype,
                    "template_id": meta.get("template", "unknown"),
                    "dependency": "none",
                    "seed": seed,
                }
                all_instances.append(instance)
                stats["categories"][cat_name]["count"] = stats["categories"][cat_name].get("count", 0) + 1
                instance_id += 1
            except Exception as e:
                print(f"  Warning: failed to generate {skill_code} instance {i}: {e}")

    # Level 2: Pairwise compositions
    print(f"Generating Level 2 (pairwise compositions)...")
    pairs = list(combinations(SKILL_CODES, 2))
    for sa, sb in pairs:
        cat_name = f"{sa}+{sb}"
        stats["categories"][cat_name] = {"level": 2, "count": 0}
        for i in range(instances_per_category):
            try:
                q, a, atype, meta = compose_pairwise(sa, sb, rng, difficulty)
                instance = {
                    "id": f"s{seed}_{instance_id:05d}",
                    "skill_combo": cat_name,
                    "skills": [sa, sb],
                    "level": 2,
                    "question": q,
                    "answer": a,
                    "answer_type": atype,
                    "template_id": meta.get("template", "unknown"),
                    "dependency": meta.get("dependency", "unknown"),
                    "seed": seed,
                }
                if "sub_answers" in meta:
                    instance["sub_answers"] = meta["sub_answers"]
                    instance["sub_types"] = meta["sub_types"]
                all_instances.append(instance)
                stats["categories"][cat_name]["count"] += 1
                instance_id += 1
            except Exception as e:
                print(f"  Warning: failed to generate {cat_name} instance {i}: {e}")

    # Level 3: Triple compositions
    print(f"Generating Level 3 (triple compositions)...")
    triples = list(combinations(SKILL_CODES, 3))
    for sa, sb, sc in triples:
        cat_name = f"{sa}+{sb}+{sc}"
        stats["categories"][cat_name] = {"level": 3, "count": 0}
        for i in range(instances_per_category):
            try:
                q, a, atype, meta = compose_triple(sa, sb, sc, rng, difficulty)
                instance = {
                    "id": f"s{seed}_{instance_id:05d}",
                    "skill_combo": cat_name,
                    "skills": [sa, sb, sc],
                    "level": 3,
                    "question": q,
                    "answer": a,
                    "answer_type": atype,
                    "template_id": meta.get("template", "unknown"),
                    "dependency": meta.get("dependency", "unknown"),
                    "seed": seed,
                }
                if "sub_answers" in meta:
                    instance["sub_answers"] = meta["sub_answers"]
                    instance["sub_types"] = meta["sub_types"]
                all_instances.append(instance)
                stats["categories"][cat_name]["count"] += 1
                instance_id += 1
            except Exception as e:
                print(f"  Warning: failed to generate {cat_name} instance {i}: {e}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"benchmark_seed{seed}.jsonl")
    with open(output_file, "w") as f:
        for inst in all_instances:
            f.write(json.dumps(inst) + "\n")

    stats["total_instances"] = len(all_instances)
    stats["level_counts"] = {
        1: sum(1 for inst in all_instances if inst["level"] == 1),
        2: sum(1 for inst in all_instances if inst["level"] == 2),
        3: sum(1 for inst in all_instances if inst["level"] == 3),
    }

    stats_file = os.path.join(output_dir, f"benchmark_stats_seed{seed}.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Generated {len(all_instances)} instances -> {output_file}")
    print(f"  Level 1: {stats['level_counts'][1]}, Level 2: {stats['level_counts'][2]}, Level 3: {stats['level_counts'][3]}")
    return all_instances, stats


def validate_benchmark(filepath: str):
    """Validate generated benchmark."""
    print(f"\nValidating {filepath}...")
    instances = []
    with open(filepath) as f:
        for line in f:
            instances.append(json.loads(line))

    issues = []
    seen_questions = set()
    for inst in instances:
        # Check required fields
        for field in ["id", "skill_combo", "level", "question", "answer", "answer_type"]:
            if field not in inst:
                issues.append(f"Missing field '{field}' in {inst.get('id', 'unknown')}")

        # Check for duplicates
        q = inst["question"]
        if q in seen_questions:
            issues.append(f"Duplicate question: {inst['id']}")
        seen_questions.add(q)

        # Check answer is non-empty
        if not inst.get("answer"):
            issues.append(f"Empty answer: {inst['id']}")

    print(f"  Total instances: {len(instances)}")
    print(f"  Issues found: {len(issues)}")
    if issues:
        for iss in issues[:10]:
            print(f"    - {iss}")
    return len(issues) == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--instances", type=int, default=50)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--difficulty", default="medium")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / args.output_dir

    for seed in args.seeds:
        instances, stats = generate_benchmark(
            seed=seed,
            instances_per_category=args.instances,
            output_dir=str(output_dir),
            difficulty=args.difficulty,
        )
        validate_benchmark(str(output_dir / f"benchmark_seed{seed}.jsonl"))
