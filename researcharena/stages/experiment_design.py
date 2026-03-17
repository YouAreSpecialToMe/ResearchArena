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
) -> tuple[dict | None, object]:
    """Let the agent implement and run experiments.

    Expects idea.json to already exist in workspace.

    Returns:
        Tuple of (parsed results dict or None, AgentResult).
    """
    task = _build_task(workspace, prior_errors, resources, attempt, max_attempts, idea_attempt, max_ideas)

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
            "   Plan your experiments to fit within these resources.\n"
        )
        impl_block = (
            "2. IMPLEMENTATION\n"
            "   - Write clean, runnable experiment code\n"
            "   - Install any packages you need (pip install)\n"
            "   - All computation runs on CPU — optimize accordingly\n\n"
        )
    else:
        resource_block = (
            "AVAILABLE RESOURCES:\n"
            f"   - GPU: {gpus}x {gpu_type} ({gpu_mem}GB VRAM each)\n"
            f"   - RAM: {mem}GB\n"
            f"   - CPU: {cpus} cores\n"
            f"   - Time limit: ~{hours} hours total for ALL experiments\n"
            "   Plan your experiments to fit within these resources.\n"
        )
        impl_block = (
            "2. IMPLEMENTATION\n"
            "   - Write clean, runnable experiment code\n"
            "   - Install any packages you need (pip install)\n"
            "   - Use GPUs for training (CUDA is available)\n\n"
        )

    exp_retries_left = max_attempts - attempt
    ideas_left = max_ideas - idea_attempt
    task = (
        "=== STAGE 2: EXPERIMENTS ===\n\n"
        f"BUDGET: Experiment attempt {attempt}/{max_attempts} for this idea"
        f" ({exp_retries_left} retries left)."
        f" Idea {idea_attempt}/{max_ideas}"
        f" ({ideas_left} new ideas left if this one is abandoned).\n\n"
        "Read idea.json (your research idea from Stage 1) and "
        "experiment_guidelines.md for detailed instructions on designing "
        "and running rigorous experiments.\n\n"
        f"{resource_block}\n"
        "Your job is to conduct the full experimental workflow for the research "
        "idea described in idea.json:\n\n"
        "1. EXPERIMENT DESIGN\n"
        "   - Choose appropriate datasets (use standard benchmarks when possible)\n"
        "   - Define evaluation metrics relevant to your claims\n"
        "   - Select at least 2 meaningful baselines for comparison\n"
        "   - Plan ablation studies to show each component's contribution\n\n"
        f"{impl_block}"
        "3. EXECUTION\n"
        "   - Run your method AND all baselines on the same data splits\n"
        "   - Run with at least 3 different random seeds for error bars\n"
        "   - Run ablation experiments (remove components one at a time)\n"
        "   - If something crashes, debug and fix it — iterate until it works\n\n"
        "4. RESULTS\n"
        "   - Save all results to results.json with this structure:\n"
        "     {\n"
        '       "method": {"metric1": mean±std, ...},\n'
        '       "baselines": {"baseline1": {"metric1": mean±std, ...}, ...},\n'
        '       "ablations": {"no_component_X": {"metric1": mean±std, ...}, ...},\n'
        '       "config": {"seeds": [...], "dataset": "...", "epochs": N, ...}\n'
        "     }\n"
        "   - Save any figures (training curves, comparison plots) to figures/\n\n"
        "You have full autonomy — install packages, write multiple files, "
        "debug and fix errors, run multiple experiment variants, etc.\n\n"
        "CRITICAL: Every number in results.json MUST come from actually running "
        "the experiment code. DO NOT fabricate, hardcode, or manually write results. "
        "If your method doesn't beat baselines, report that honestly — negative "
        "results with honest analysis are valuable science.\n\n"
        "ABANDON OPTION: At any point during experiments, if you determine this "
        "idea is not viable, you may abandon it by writing a file called "
        'abandon.json with: {"abandon": true, "reason": "..."}. '
        "The pipeline will go back to ideation with a new idea. Valid reasons "
        "to abandon include (but are not limited to):\n"
        "  - Method clearly underperforms all baselines (not just a tuning issue)\n"
        "  - Experiment setup is infeasible (dependencies won't install, dataset "
        "too large for available resources, etc.)\n"
        "  - Code won't run after multiple debugging attempts\n"
        "  - Approach is fundamentally intractable with available compute\n"
        "  - Any other reason you judge the idea is not worth pursuing\n"
        "Use your judgment — marginal or mixed results are still worth writing up."
    )

    if prior_errors:
        task += "\n\n--- PREVIOUS ATTEMPTS FAILED ---\n"
        for i, err in enumerate(prior_errors):
            task += f"\nAttempt {i+1} error:\n{err}\n"
        task += "\nFix the issues and try a different approach if needed.\n"

    return task


# Sentinel returned when the agent explicitly abandons the idea
ABANDON_SIGNAL = "__ABANDON__"


def _parse_output(workspace: Path) -> dict | str | None:
    # Check for abandon signal first
    abandon_path = workspace / "abandon.json"
    if abandon_path.exists():
        try:
            data = json.loads(abandon_path.read_text())
            if data.get("abandon"):
                return ABANDON_SIGNAL
        except json.JSONDecodeError:
            pass

    results_path = workspace / "results.json"
    if not results_path.exists():
        return None

    try:
        return json.loads(results_path.read_text())
    except json.JSONDecodeError:
        return None
