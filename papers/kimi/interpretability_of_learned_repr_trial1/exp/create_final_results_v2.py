"""Create final results.json with all available data."""
import json
import os

# Load synthetic results
with open('exp/synthetic_fixed/results.json', 'r') as f:
    synthetic_results = json.load(f)

# Create comprehensive results structure
final_results = {
    "experiment_summary": {
        "title": "CAGER: Causal Geometric Explanation Recovery - Fixed Results V2",
        "description": "Evaluation of interpretability methods using improved C-GAS metric without dimensionality penalty",
        "tasks_evaluated": ["synthetic"],
        "methods": ["SAE", "Oracle"],
        "seeds": 3,
        "fixes_applied": [
            "Removed overly aggressive dimensionality penalty from C-GAS",
            "Fixed JSON serialization with robust numpy type conversion",
            "Trained missing PCA 4x/16x baselines for synthetic task",
            "Implemented validation consistency checks"
        ],
        "known_issues": [
            "Oracle C-GAS unexpectedly low (0.07 vs expected ~1.0)",
            "SAE 16x shows negative C-GAS (dead neuron issue)",
            "IOI experiment failed during validation (indexing bug)",
            "RAVEL experiment failed during baseline training (PCA component mismatch)"
        ]
    },
    "results_by_task": {
        "synthetic": {
            "summary": synthetic_results.get('summary', {}),
            "correlations": synthetic_results.get('correlations', {}),
            "all_results": synthetic_results.get('all_results', [])
        },
        "ioi": {
            "status": "failed",
            "error": "IndexError in validation consistency check",
            "n_validated_dims": 0,
            "completed_steps": ["dataset_creation", "activation_extraction", "causal_identification"]
        },
        "ravel": {
            "status": "failed", 
            "error": "PCA component mismatch (768 components > 120 samples)",
            "n_validated_dims": 0,
            "completed_steps": ["dataset_loading", "activation_extraction"]
        }
    },
    "key_findings": [
        {
            "title": "SAE 1x achieves C-GAS = 0.57 on synthetic task",
            "status": "CONFIRMED",
            "details": "SAE 1x C-GAS: 0.572 ± 0.084 (3 seeds)"
        },
        {
            "title": "Positive correlation between C-GAS and ground-truth recovery",
            "status": "PARTIAL",
            "details": f"Pearson r = {synthetic_results.get('correlations', {}).get('pearson_r', 0):.4f} (p = {synthetic_results.get('correlations', {}).get('pearson_p', 1):.4f})"
        },
        {
            "title": "Oracle C-GAS unexpectedly low",
            "status": "ISSUE",
            "details": "Oracle C-GAS: 0.073 ± 0.027 (expected ~1.0). Indicates metric formulation issue."
        },
        {
            "title": "SAE 16x shows degraded performance",
            "status": "ISSUE",
            "details": "SAE 16x C-GAS: -0.060 ± 0.113. Suggests dead neuron problem in high-dim SAEs."
        },
        {
            "title": "IOI and RAVEL experiments failed",
            "status": "FAILED",
            "details": "Implementation bugs prevented completion. See known_issues."
        }
    ],
    "statistical_tests": {
        "synthetic": {
            "sae_vs_random": "Not computed (random results incomplete)",
            "sae_vs_pca": "Not computed (PCA results incomplete)"
        }
    },
    "success_criteria_assessment": {
        "criterion_1_sae_cgas_above_075": {
            "target": "C-GAS > 0.75",
            "achieved": "0.57 (SAE 1x)",
            "passed": False,
            "notes": "SAE 1x fell short of 0.75 threshold"
        },
        "criterion_2_sae_beats_baselines": {
            "target": "p < 0.01",
            "achieved": "Not computed",
            "passed": False,
            "notes": "Baseline results incomplete due to timeout"
        },
        "criterion_3_cgas_correlates_recovery": {
            "target": "r > 0.8",
            "achieved": f"r = {synthetic_results.get('correlations', {}).get('pearson_r', 0):.2f}",
            "passed": False,
            "notes": "Positive but weaker correlation than expected"
        },
        "criterion_4_validation_improves_predictions": {
            "target": "Yes",
            "achieved": "Not tested",
            "passed": False,
            "notes": "Ablation study not completed"
        }
    }
}

# Save results
with open('results_final_v2.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("Final results saved to results_final_v2.json")
print("\nSummary:")
print(f"  Synthetic task: COMPLETE ({len(synthetic_results.get('all_results', []))} results)")
print(f"  IOI task: FAILED (indexing bug)")
print(f"  RAVEL task: FAILED (PCA mismatch)")
print(f"\nCorrelation C-GAS vs Recovery: r = {synthetic_results.get('correlations', {}).get('pearson_r', 0):.4f}")
