"""
Quick test to verify ESR triggers revisions with current thresholds.
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from shared.models import load_model
from shared.data_loader import load_gsm8k, create_cot_prompt
from run_complete_experiments import ESRGenerator

def test_esr():
    print("Loading model...")
    model, tokenizer = load_model("Qwen/Qwen3-1.7B")
    
    print("\nLoading test data...")
    data = load_gsm8k("test")
    
    # Test with different thresholds
    thresholds = [
        (2.5, 1.5, "Original (high)"),
        (1.5, 0.8, "New (lower)"),
        (1.0, 0.5, "Very low"),
    ]
    
    for tau_h, tau_v, desc in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing ESR with tau_h={tau_h}, tau_v={tau_v} ({desc})")
        print('='*60)
        
        generator = ESRGenerator(model, tokenizer, tau_h=tau_h, tau_v=tau_v, r_max=3)
        
        # Test on 5 problems
        total_revisions = 0
        for i, item in enumerate(data[:5]):
            prompt = create_cot_prompt(item["question"])
            result = generator.generate(prompt)
            
            revisions = result["revision_count"]
            total_revisions += revisions
            triggers = len(result.get("uncertainty_triggers", []))
            
            print(f"  Problem {i+1}: {revisions} revisions, {triggers} triggers")
        
        print(f"  Total revisions: {total_revisions}/5 problems")
        if total_revisions > 0:
            print(f"  ✓ Revisions are triggering!")
        else:
            print(f"  ✗ No revisions triggered - thresholds may be too high")

if __name__ == "__main__":
    test_esr()
