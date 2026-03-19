"""Self-review gates for idea, experiment, and paper stages.

Each gate runs the researcher agent as its own reviewer, using a
stage-specific subset of the reviewer guidelines. The agent evaluates
its own work and produces a score + feedback. If the score is below
the threshold, the pipeline routes back for refinement.
"""

from __future__ import annotations

import json
from pathlib import Path

from researcharena.utils.agent_runner import invoke_agent

# Template directory
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

# Stage -> template filename mapping
_TEMPLATE_FILES = {
    "idea": "self_review_idea.md",
    "experiment": "self_review_experiment.md",
    "paper": "self_review_paper.md",
}


def _load_guidelines(stage: str, domain: str = "ml") -> str:
    """Load self-review guidelines for the given stage and domain."""
    domain_path = _TEMPLATES_DIR / domain / _TEMPLATE_FILES[stage]
    if domain_path.exists():
        return domain_path.read_text()
    # Fallback to ml
    return (_TEMPLATES_DIR / "ml" / _TEMPLATE_FILES[stage]).read_text()


def run_self_review(
    agent_type: str,
    workspace: Path,
    stage: str,
    agent_config: dict | None = None,
    timeout: int = 900,
    domain: str = "ml",
) -> tuple[float, str, object]:
    """Run a self-review for the given stage.

    Args:
        agent_type: The CLI agent type (claude, codex, etc.)
        workspace: The workspace directory containing artifacts to review
        stage: One of "idea", "experiment", "paper"
        agent_config: Agent configuration dict
        timeout: Max seconds for the self-review
        domain: Domain for guideline templates

    Returns:
        (score, feedback, agent_result)
    """
    guidelines = _load_guidelines(stage, domain)
    task = _build_task(stage, guidelines)

    agent_result = invoke_agent(
        agent_type=agent_type,
        task=task,
        workspace=workspace,
        timeout=timeout,
        agent_config=agent_config,
        readonly=True,
    )

    score, feedback = _parse_output(agent_result)
    return score, feedback, agent_result


def _build_task(stage: str, guidelines: str) -> str:
    """Build the self-review task prompt."""
    stage_descriptions = {
        "idea": (
            "You are reviewing your research proposal and experiment plan.\n"
            "Evaluate proposal.md, plan.json, and references/ in the workspace.\n"
            "Be critical — it is better to catch problems before running experiments."
        ),
        "experiment": (
            "You are reviewing your experiment results before writing the paper.\n"
            "Evaluate results.json, experiment code, logs, and figures/ against plan.json.\n"
            "Be honest — weak results should be flagged now, not hidden in the paper."
        ),
        "paper": (
            "You are doing a final pre-submission review of your paper.\n"
            "Evaluate paper.tex/paper.pdf against results.json and experiment code.\n"
            "This is your last chance to catch issues before peer review."
        ),
    }

    return (
        f"=== SELF-REVIEW: {stage.upper()} ===\n\n"
        f"{stage_descriptions[stage]}\n\n"
        "IMPORTANT: Be rigorous and honest. Do NOT inflate your score.\n"
        "Search the web to verify novelty claims and reference integrity.\n"
        "A score of 6 means 'acceptable with concerns' — only give 8+ if\n"
        "the work is genuinely strong.\n\n"
        f"{guidelines}"
    )


def _parse_output(agent_result) -> tuple[float, str]:
    """Parse score and feedback from agent output.

    Searches for a JSON object in stdout containing 'score' and 'feedback'.
    Returns (score, feedback). Defaults to (0, "") on parse failure.
    """
    if not agent_result or not agent_result.stdout:
        return 0.0, "Self-review produced no output."

    stdout = agent_result.stdout

    # Try to find JSON in the output
    # Look for the outermost { ... } containing "score"
    best_score = 0.0
    best_feedback = ""

    # Try parsing from the last JSON block (most likely the final output)
    brace_depth = 0
    json_start = -1
    candidates = []

    for i, ch in enumerate(stdout):
        if ch == '{':
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and json_start >= 0:
                candidates.append(stdout[json_start:i + 1])
                json_start = -1

    # Try candidates from last to first (final output most likely)
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
            if "score" in data:
                best_score = float(data["score"])
                best_feedback = data.get("feedback", "")
                # Also collect weaknesses as additional feedback
                weaknesses = data.get("weaknesses", [])
                if weaknesses and not best_feedback:
                    best_feedback = "; ".join(weaknesses)
                elif weaknesses:
                    best_feedback += "\nWeaknesses: " + "; ".join(weaknesses)
                break
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    return best_score, best_feedback
