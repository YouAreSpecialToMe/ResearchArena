#!/usr/bin/env python3
"""
Experiment 01: Data Preparation
- Create rewrite rules
- Prepare benchmark programs (PolyBench/C simulation)
- Save configuration
"""

import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.simulation import create_rewrite_rules, create_polybench_programs

def main():
    print("=" * 60)
    print("Experiment 01: Data Preparation")
    print("=" * 60)
    
    # Create directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create rewrite rules
    print("\n[1/3] Creating rewrite rules...")
    rules = create_rewrite_rules(seed=42)
    print(f"  Created {len(rules)} rewrite rules")
    print(f"    - Arithmetic: {sum(1 for r in rules if r.rule_type.value == 'arithmetic')}")
    print(f"    - Control Flow: {sum(1 for r in rules if r.rule_type.value == 'control_flow')}")
    print(f"    - Memory: {sum(1 for r in rules if r.rule_type.value == 'memory')}")
    
    # Save rules
    rules_dict = [r.to_dict() for r in rules]
    with open(data_dir / "rules.json", "w") as f:
        json.dump(rules_dict, f, indent=2)
    print(f"  Saved to data/rules.json")
    
    # Create benchmark programs
    print("\n[2/3] Creating benchmark programs...")
    training_programs, test_programs = create_polybench_programs()
    print(f"  Training programs: {len(training_programs)}")
    print(f"    Programs: {', '.join(p.name for p in training_programs)}")
    print(f"  Test programs: {len(test_programs)}")
    print(f"    Programs: {', '.join(p.name for p in test_programs)}")
    
    # Save programs
    with open(data_dir / "training_programs.pkl", "wb") as f:
        pickle.dump(training_programs, f)
    with open(data_dir / "test_programs.pkl", "wb") as f:
        pickle.dump(test_programs, f)
    print(f"  Saved to data/training_programs.pkl and data/test_programs.pkl")
    
    # Create metadata
    print("\n[3/3] Creating metadata...")
    metadata = {
        "num_rules": len(rules),
        "num_training_programs": len(training_programs),
        "num_test_programs": len(test_programs),
        "training_programs": [p.name for p in training_programs],
        "test_programs": [p.name for p in test_programs],
        "rule_breakdown": {
            "arithmetic": sum(1 for r in rules if r.rule_type.value == 'arithmetic'),
            "control_flow": sum(1 for r in rules if r.rule_type.value == 'control_flow'),
            "memory": sum(1 for r in rules if r.rule_type.value == 'memory'),
        }
    }
    
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to data/metadata.json")
    
    # Save results for this experiment
    results = {
        "experiment": "01_data_prep",
        "status": "completed",
        "output_files": [
            "data/rules.json",
            "data/training_programs.pkl",
            "data/test_programs.pkl",
            "data/metadata.json"
        ]
    }
    
    with open(Path("exp/01_data_prep/results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Data preparation completed successfully!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())
