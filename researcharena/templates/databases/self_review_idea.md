# Self-Review: Idea & Experiment Plan

You are critically reviewing your OWN research proposal and experiment plan
before committing resources to run experiments. Be honest and rigorous —
it is better to catch problems now than after spending hours of compute.

## What to Evaluate

Review `proposal.md`, `plan.json`, and `references/` in the workspace.

### 1. Novelty
- Is this idea genuinely new, or does similar work already exist?
- **Search online** (arXiv, Semantic Scholar, Google Scholar) to check
  whether the core idea has already been published
- If you find substantially similar prior work, score novelty <= 4
- Novel combinations of existing techniques count IF the combination
  itself provides new insight

### 2. Soundness
- Is the proposed methodology appropriate for the problem?
- Are the assumptions stated and reasonable?
- Could the approach plausibly work, or is it fundamentally flawed?
- Are there obvious failure modes not addressed?

### 3. Significance
- Does this problem matter to the research community?
- Would the results (positive or negative) be useful to others?
- Is the scope appropriate — not too narrow, not too ambitious?

### 4. Experiment Plan Quality
- Is plan.json complete — does it cover baselines, main experiments,
  ablations, and evaluation?
- Are the planned experiments feasible within the resource budget?
- Are success criteria clearly defined?
- Are there at least 2 meaningful baselines planned?
- Is each step detailed enough to follow without ambiguity?

### 5. References
- Are key related works cited in the proposal?
- Are the references real, verifiable publications?
- Is the proposal well-positioned relative to prior work?

## Scoring

Rate 0-10 (even numbers only):
- 10: Exceptional idea, thorough plan, clearly novel
- 8: Strong idea, solid plan, good novelty
- 6: Reasonable idea, adequate plan, some concerns
- 4: Weak idea or major plan gaps
- 2: Fundamentally flawed or clearly not novel
- 0: Trivial or nonsensical

## Output Format

Output ONLY a JSON object:
{
    "score": <int, one of: 0, 2, 4, 6, 8, 10>,
    "pass": <bool>,
    "summary": "<what the proposed research is about>",
    "strengths": ["<strength1>", ...],
    "weaknesses": ["<weakness1>", ...],
    "feedback": "<specific, actionable feedback for improving the idea and plan>",
    "checklist": {
        "novelty": {"score": <int 1-10>, "note": "..."},
        "soundness": {"score": <int 1-10>, "note": "..."},
        "significance": {"score": <int 1-10>, "note": "..."},
        "plan_quality": {"score": <int 1-10>, "note": "..."},
        "references": {"score": <int 1-10>, "note": "..."}
    }
}
