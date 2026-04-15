#!/usr/bin/env python3
"""Check experiment progress and generate interim reports."""

import json
from pathlib import Path
from datetime import datetime

def check_progress():
    results_dir = Path("exp/results")
    
    # Find all result files
    result_files = list(results_dir.glob("*_gsm8k_seed*.json"))
    
    print("="*70)
    print("Experiment Progress Check")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total result files: {len(result_files)}")
    print()
    
    # Group by method
    methods = {}
    for path in result_files:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            method = data.get("method", "unknown")
            seed = data.get("seed", "unknown")
            accuracy = data.get("accuracy", 0)
            total = data.get("total_problems", 0)
            
            if method not in methods:
                methods[method] = []
            methods[method].append({
                "seed": seed,
                "accuracy": accuracy,
                "total": total,
                "file": path.name
            })
        except Exception as e:
            print(f"Error reading {path}: {e}")
    
    print("Results by method:")
    print("-"*70)
    for method, results in sorted(methods.items()):
        print(f"\n{method}:")
        for r in results:
            print(f"  Seed {r['seed']}: Acc={r['accuracy']:.3f} (n={r['total']})")
    
    print("\n" + "="*70)
    
    # Check for master script log
    master_log = Path("logs/master_run.log")
    if master_log.exists():
        lines = master_log.read_text().split('\n')
        print(f"\nMaster script status: {len(lines)} lines in log")
        
        # Check last few lines
        for line in lines[-10:]:
            if line.strip():
                print(f"  {line}")
    
    # Check running processes
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        python_procs = [l for l in result.stdout.split('\n') if 'python' in l and 'run_batch' in l]
        if python_procs:
            print(f"\nActive experiment processes: {len(python_procs)}")
    except:
        pass

if __name__ == "__main__":
    check_progress()
