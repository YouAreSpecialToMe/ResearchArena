"""Stage 2: Let the CLI agent implement and run experiments.

The agent gets the idea.json and workspace. It handles the full experimental
workflow: designing experiments, implementing code, running training/evaluation,
debugging, and producing structured results.

The agent self-iterates within a single invocation — it writes code, runs it,
sees errors, fixes them, reruns, etc. We just check the final output.
"""

from __future__ import annotations

import json
from pathlib import Path

from researcharena.utils.agent_runner import invoke_agent


def run(
    agent_type: str,
    workspace: Path,
    timeout: int = 14400,
    agent_config: dict | None = None,
    prior_errors: list[str] | None = None,
    resources: dict | None = None,
    attempt: int = 1,
    max_attempts: int = 3,
    idea_attempt: int = 1,
    max_ideas: int = 5,
    self_review_feedback: str = "",
) -> tuple[dict | None, object]:
    """Let the agent implement and run experiments.

    Expects idea.json to already exist in workspace.

    Returns:
        Tuple of (parsed results dict or None, AgentResult).
    """
    task = _build_task(workspace, prior_errors, resources, attempt, max_attempts, idea_attempt, max_ideas, self_review_feedback)

    agent_result = invoke_agent(
        agent_type=agent_type,
        task=task,
        workspace=workspace,
        timeout=timeout,
        agent_config=agent_config,
    )

    return _parse_output(workspace), agent_result


def _build_task(
    workspace: Path,
    prior_errors: list[str] | None,
    resources: dict | None = None,
    attempt: int = 1,
    max_attempts: int = 3,
    idea_attempt: int = 1,
    max_ideas: int = 5,
    self_review_feedback: str = "",
) -> str:
    res = resources or {}
    platform = res.get("platform", "gpu")
    gpus = res.get("gpus", 0 if platform == "cpu" else 1)
    gpu_type = res.get("gpu_type", "GPU")
    gpu_mem = res.get("gpu_memory_gb", 80)
    cpus = res.get("cpus", 8)
    mem = res.get("memory_gb", 32)
    hours = res.get("time_hours", 8)

    # Build resource description based on platform
    if platform == "cpu" or gpus == 0:
        resource_block = (
            "AVAILABLE RESOURCES:\n"
            f"   - CPU: {cpus} cores\n"
            f"   - RAM: {mem}GB\n"
            f"   - Time limit: ~{hours} hours total for ALL experiments\n"
            "   - NO GPU available — all computation must run on CPU.\n"
        )
    else:
        resource_block = (
            "AVAILABLE RESOURCES:\n"
            f"   - GPU: {gpus}x {gpu_type} ({gpu_mem}GB VRAM each)\n"
            f"   - RAM: {mem}GB\n"
            f"   - CPU: {cpus} cores\n"
            f"   - Time limit: ~{hours} hours total for ALL experiments\n"
        )

    exp_retries_left = max_attempts - attempt
    ideas_left = max_ideas - idea_attempt
    task = (
        "=== STAGE 2: EXPERIMENTS ===\n\n"
        f"BUDGET: Experiment attempt {attempt}/{max_attempts} for this idea"
        f" ({exp_retries_left} retries left)."
        f" Idea {idea_attempt}/{max_ideas}"
        f" ({ideas_left} new ideas left).\n\n"
        "Read these files from Stage 1:\n"
        "  - idea.json: your research idea summary\n"
        "  - proposal.md: full research proposal (if available)\n"
        "  - plan.json: your detailed experiment plan (if available)\n"
        "  - experiment_guidelines.md: general experiment guidelines\n\n"
        f"{resource_block}\n"
        "If plan.json exists, FOLLOW IT step by step — it contains your\n"
        "pre-designed experiment plan with detailed instructions for each step.\n"
        "Execute every step in order. If a step is infeasible, document why\n"
        "and adjust accordingly.\n\n"
        "If plan.json does not exist, design experiments from scratch based\n"
        "on idea.json and experiment_guidelines.md.\n\n"
        "Read experiment_guidelines.md carefully — it covers experiment design,\n"
        "implementation, workspace structure, and the reproducibility checklist.\n\n"
        "Key requirements:\n"
        "- At least 2 meaningful baselines\n"
        "- At least 3 random seeds with mean +/- std\n"
        "- Ablation studies for each novel component\n"
        "- Organize code under exp/ with one subfolder per experiment\n"
        "- Save aggregated results.json at workspace root\n"
        "- Save figures to figures/\n\n"
        "CRITICAL: Every number in results.json MUST come from actually running "
        "the experiment code. DO NOT fabricate, hardcode, or manually write results. "
        "If your method doesn't beat baselines, report that honestly — negative "
        "results with honest analysis are valuable science."
    )

    if self_review_feedback:
        task += (
            "\n\n--- SELF-REVIEW FEEDBACK (address these issues) ---\n"
            f"{self_review_feedback}\n"
            "--- END FEEDBACK ---\n"
        )

    if prior_errors:
        task += "\n\n--- PREVIOUS ATTEMPTS FAILED ---\n"
        for i, err in enumerate(prior_errors):
            task += f"\nAttempt {i+1} error:\n{err}\n"
        task += "\nFix the issues and try a different approach if needed.\n"

    return task


def _parse_output(workspace: Path) -> dict | None:
    results_path = workspace / "results.json"
    if not results_path.exists():
        return None

    try:
        return json.loads(results_path.read_text())
    except json.JSONDecodeError:
        return None
