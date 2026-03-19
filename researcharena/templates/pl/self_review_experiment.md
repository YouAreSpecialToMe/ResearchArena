# Self-Review: Experiment Results

You are critically reviewing your OWN experiment results before writing
the paper. Be honest — if the results are weak, it is better to refine
the approach now than to write a paper that will be rejected.

## What to Evaluate

Review `results.json`, experiment code (`.py` files), logs, `figures/`,
and compare against `plan.json` (the original experiment plan).

### 1. Plan Compliance
- Were all steps in plan.json executed?
- If any steps were skipped or modified, is the deviation justified?
- Were all planned baselines implemented and run?
- Were all planned ablations completed?

### 2. Soundness
- Do the results support the claims in proposal.md?
- Is the experimental methodology valid?
- Are there confounding factors that could explain the results?
- If the method underperforms baselines, is this acknowledged honestly?

### 3. Experimental Rigor
- Are there at least 2 meaningful baselines with fair comparisons?
- Are ablation studies present showing each component's contribution?
- Are error bars / confidence intervals reported?
- Are results from multiple runs (different seeds)?
- Are comparisons fair (same data, same compute budget)?

### 4. Results Integrity
- Does results.json contain actual experimental output (not hardcoded)?
- Do the experiment logs show real training/evaluation runs?
- Do the numbers in results.json match what the logs show?
- Does the code implement what proposal.md describes?

### 5. Reproducibility
- Are all hyperparameters recorded?
- Is the data described (splits, sizes, preprocessing)?
- Could someone reproduce these results from the code and config?

## Scoring

Rate 0-10 (even numbers only):
- 10: Thorough experiments, strong results, full plan compliance
- 8: Solid experiments, good results, minor gaps
- 6: Adequate experiments, mixed results, some plan deviations
- 4: Weak experiments, missing baselines or ablations
- 2: Severely flawed methodology or fabricated results
- 0: No real experiments conducted

## Output Format

Output ONLY a JSON object:
{
    "score": <int, one of: 0, 2, 4, 6, 8, 10>,
    "pass": <bool>,
    "summary": "<what the experiments show>",
    "strengths": ["<strength1>", ...],
    "weaknesses": ["<weakness1>", ...],
    "feedback": "<specific, actionable feedback for improving experiments>",
    "checklist": {
        "plan_compliance": {"score": <int 1-10>, "note": "..."},
        "soundness": {"score": <int 1-10>, "note": "..."},
        "experimental_rigor": {"score": <int 1-10>, "note": "..."},
        "results_integrity": {"score": <int 1-10>, "note": "..."},
        "reproducibility": {"score": <int 1-10>, "note": "..."}
    }
}
