"""Stage 1: Let the CLI agent come up with a research idea.

The agent gets a workspace and a seed field (e.g., "cv", "nlp"). It decides
on its own how to explore — searching papers, brainstorming, etc.

At this stage, we only ask for initial thoughts: what's the idea, why it
matters, and a rough approach. Concrete details like datasets, metrics,
and baselines come later in the experiment stage.
"""

from __future__ import annotations

import json
from pathlib import Path

from researcharena.utils.agent_runner import invoke_agent


REQUIRED_FIELDS = ["description", "motivation", "proposed_approach", "related_work"]


def run(
    agent_type: str,
    seed_topic: str,
    workspace: Path,
    history: list[dict] | None = None,
    timeout: int = 1800,
    agent_config: dict | None = None,
    resources: dict | None = None,
    attempt: int = 1,
    max_attempts: int = 5,
) -> tuple[dict | None, object]:
    """Let the agent generate a research idea.

    Returns:
        Tuple of (parsed idea dict or None, AgentResult).
    """
    task = _build_task(seed_topic, history, resources, attempt, max_attempts)

    agent_result = invoke_agent(
        agent_type=agent_type,
        task=task,
        workspace=workspace,
        timeout=timeout,
        agent_config=agent_config,
    )

    return _parse_output(workspace), agent_result


def _build_task(seed_topic: str, history: list[dict] | None, resources: dict | None = None, attempt: int = 1, max_attempts: int = 5) -> str:
    res = resources or {}
    gpus = res.get("gpus", 1)
    gpu_type = res.get("gpu_type", "GPU")
    gpu_mem = res.get("gpu_memory_gb", 80)
    cpus = res.get("cpus", 8)
    mem = res.get("memory_gb", 32)
    hours = res.get("time_hours", 8)

    remaining = max_attempts - attempt
    task = (
        f"=== STAGE 1: IDEATION ===\n\n"
        f"Your seed field is: {seed_topic}\n\n"
        f"ATTEMPT: {attempt}/{max_attempts} "
        f"({'LAST CHANCE — make it count!' if remaining == 0 else f'{remaining} retries remaining if this idea fails'})\n\n"
        "Read idea_guidelines.md for detailed instructions.\n\n"
        "STEP 1 — LITERATURE SEARCH (do this FIRST):\n"
        "   Search for the latest papers in this field using arXiv, Semantic\n"
        "   Scholar, and Google Scholar. Focus on:\n"
        "   - Recent papers (last 1-2 years) to understand the current frontier\n"
        "   - Key open problems and limitations of existing methods\n"
        "   - Trends and promising directions\n"
        "   This grounds your idea in what's actually new and needed.\n\n"
        "STEP 2 — GENERATE IDEAS:\n"
        "   Based on your literature search, come up with a novel, concrete,\n"
        "   and feasible research idea that could result in a publishable ML\n"
        "   conference paper.\n\n"
        "STEP 3 — VERIFY NOVELTY:\n"
        "   Search online again to make sure your specific idea hasn't already\n"
        "   been published. If it has, go back to Step 2.\n\n"
        "COMPUTATIONAL RESOURCES (scope your idea to fit):\n"
        f"   - GPU: {gpus}x {gpu_type} ({gpu_mem}GB VRAM each)\n"
        f"   - RAM: {mem}GB\n"
        f"   - CPU: {cpus} cores\n"
        f"   - Time limit: ~{hours} hours total for ALL experiments\n"
        "   Your idea MUST be feasible within these constraints. Avoid ideas\n"
        "   that require training large models from scratch, massive datasets,\n"
        "   or multi-day compute. Prefer ideas that can show results with\n"
        "   small-to-medium scale experiments.\n\n"
        "Focus on the core idea, not implementation details. You'll design "
        "experiments and write the paper in later stages.\n\n"
        "When done, save your idea to idea.json in the current directory with at least "
        "these fields:\n"
        "  - description: a short description of the idea (1-3 sentences)\n"
        "  - motivation: why this problem matters and what gap you're addressing\n"
        "  - proposed_approach: your high-level approach and why it should work\n"
        "  - related_work: key existing papers and how your idea differs from them\n\n"
        "You may include any other fields you find useful (e.g., hypothesis, "
        "novelty_claim)."
    )

    if history:
        task += "\n\n--- PREVIOUS FAILED ATTEMPTS (avoid these) ---\n"
        for i, h in enumerate(history):
            task += (
                f"\nAttempt {i+1}: \"{h['idea'].get('description', h['idea'].get('title', 'N/A'))}\"\n"
                f"  Failed at: {h['failure_stage']}\n"
                f"  Reason: {h['failure_reason'][:300]}\n"
            )
        task += "\n--- Generate a DIFFERENT idea that avoids these problems. ---\n"

    return task


def run_refinement(
    agent_type: str,
    workspace: Path,
    original_idea: dict,
    reviewer_feedback: str,
    results: dict | None = None,
    timeout: int = 1800,
    agent_config: dict | None = None,
    resources: dict | None = None,
    revision_attempt: int = 1,
    max_revisions: int = 2,
) -> tuple[dict | None, object]:
    """Let the agent refine an existing idea based on reviewer feedback.

    Returns:
        Tuple of (refined idea dict or None, AgentResult).
    """
    task = _build_refinement_task(
        original_idea, reviewer_feedback, results,
        resources, revision_attempt, max_revisions,
    )

    agent_result = invoke_agent(
        agent_type=agent_type,
        task=task,
        workspace=workspace,
        timeout=timeout,
        agent_config=agent_config,
    )

    return _parse_output(workspace), agent_result


def _build_refinement_task(
    original_idea: dict,
    reviewer_feedback: str,
    results: dict | None,
    resources: dict | None = None,
    revision_attempt: int = 1,
    max_revisions: int = 2,
) -> str:
    res = resources or {}
    hours = res.get("time_hours", 8)
    revisions_left = max_revisions - revision_attempt

    idea_desc = original_idea.get("description", original_idea.get("title", "N/A"))
    idea_approach = original_idea.get("proposed_approach", "")

    task = (
        "=== IDEA REFINEMENT (post-review) ===\n\n"
        f"BUDGET: Revision {revision_attempt}/{max_revisions}"
        f" ({revisions_left} revisions left after this).\n\n"
        "Your paper was reviewed and REJECTED. You now have a chance to refine\n"
        "your idea, run additional experiments, and rewrite the paper.\n\n"
        "Read idea_guidelines.md for guidance on structuring your refined idea.\n\n"
        f"ORIGINAL IDEA:\n{idea_desc}\n\n"
        f"APPROACH:\n{idea_approach}\n\n"
        "--- REVIEWER FEEDBACK ---\n"
        f"{reviewer_feedback}\n"
        "--- END FEEDBACK ---\n\n"
    )

    if results:
        # Summarize key results
        method_results = results.get("method", {})
        baseline_results = results.get("baselines", {})
        task += "PREVIOUS EXPERIMENT RESULTS (summary):\n"
        task += f"  Method: {json.dumps(method_results)[:300]}\n"
        for name, scores in baseline_results.items():
            task += f"  Baseline '{name}': {json.dumps(scores)[:200]}\n"
        task += "\n"

    task += (
        "YOUR TASK:\n"
        "1. Read the reviewer feedback carefully\n"
        "2. Decide how to strengthen your idea — you can:\n"
        "   - Refine the approach (fix methodological issues)\n"
        "   - Add new components or modify existing ones\n"
        "   - Plan additional experiments (ablations, baselines, datasets)\n"
        "   - Pivot the framing or motivation\n"
        "3. Update idea.json with your refined idea\n"
        "   Keep the same fields but update the content. Add a 'revision_notes'\n"
        "   field explaining what changed and why.\n\n"
        f"After this, you will re-run experiments (~{hours}h budget) and rewrite the paper.\n"
        "Focus on addressing the specific reviewer concerns."
    )

    return task


def _parse_output(workspace: Path) -> dict | None:
    idea_path = workspace / "idea.json"
    if not idea_path.exists():
        return None

    try:
        idea = json.loads(idea_path.read_text())
    except json.JSONDecodeError:
        return None

    # Check required fields
    missing = [f for f in REQUIRED_FIELDS if f not in idea]
    if missing:
        return None

    return idea
