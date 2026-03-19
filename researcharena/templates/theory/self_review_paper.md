# Self-Review: Paper (Pre-Submission Check)

You are critically reviewing your OWN paper before sending it to peer
reviewers. This is your last chance to catch issues. Be rigorous —
treat this as if you are reviewing someone else's work.

## What to Evaluate

Review `paper.tex`, `paper.pdf`, `results.json`, experiment code, and logs.

### 1. Novelty
- Does the paper present genuinely new ideas or insights?
- Search online to verify no substantially similar work exists
- Are novelty claims accurate and well-supported?

### 2. Soundness
- Are all claims supported by theory or experiments?
- Is the methodology appropriate for the problem?
- Do the results actually support the claims made?

### 3. Significance
- Does this work matter? Would the community benefit?
- Are the contributions clearly stated and justified?

### 4. Clarity
- Is the paper well-written and organized?
- Are contributions explicitly stated in the introduction?
- Are figures/tables self-contained with descriptive captions?
- Is notation consistent and well-defined?
- Could an expert reproduce the work from the paper alone?

### 5. Reproducibility
- Are all hyperparameters, architectures, and training details in the paper?
- Is the data described (splits, sizes, preprocessing)?
- Is compute specified (hardware, runtime)?

### 6. Experimental Rigor
- At least 2 meaningful baselines?
- Ablation study present?
- Error bars / confidence intervals reported?
- Multiple seeds?
- Fair comparisons?

### 7. References
- Are key related works cited and properly discussed?
- Is the paper well-positioned relative to prior work?

### 8. Reference Integrity
- Are all references real, verifiable publications?
- Search Semantic Scholar or Google Scholar to verify
- Any hallucinated or fabricated citations?

### 9. Results Integrity
- Do numbers in the paper match results.json EXACTLY?
- Does experiment code exist and implement what the paper describes?
- Do logs show actual runs matching reported numbers?

## Automatic Rejection Grounds
Any of these should score <= 4:
- Fabricated references
- Paper numbers don't match results.json
- Missing experiment code
- Logs contradict reported results

## Scoring

Rate 0-10 (even numbers only):
- 10: Publication-ready, no issues found
- 8: Strong paper, minor polish needed
- 6: Acceptable but needs specific improvements
- 4: Major issues that need addressing
- 2: Fundamental problems
- 0: Not a valid submission

## Output Format

Output ONLY a JSON object:
{
    "score": <int, one of: 0, 2, 4, 6, 8, 10>,
    "pass": <bool>,
    "summary": "<what the paper is about>",
    "strengths": ["<strength1>", ...],
    "weaknesses": ["<weakness1>", ...],
    "feedback": "<specific, actionable feedback for improving the paper>",
    "checklist": {
        "novelty": {"score": <int 1-10>, "note": "..."},
        "soundness": {"score": <int 1-10>, "note": "..."},
        "significance": {"score": <int 1-10>, "note": "..."},
        "clarity": {"score": <int 1-10>, "note": "..."},
        "reproducibility": {"score": <int 1-10>, "note": "..."},
        "experimental_rigor": {"score": <int 1-10>, "note": "..."},
        "references": {"score": <int 1-10>, "note": "..."},
        "reference_integrity": {"score": <int 1-10>, "note": "..."},
        "results_integrity": {"score": <int 1-10>, "note": "..."}
    }
}
