"""Stage 1: Let the CLI agent come up with a research idea.

Two invocations:
  1. Idea: agent searches literature, verifies novelty, produces proposal.md,
     idea.json, and references/
  2. Plan: agent reads the proposal and designs a detailed experiment plan
     (plan.json)

After both, the self-review evaluates the full package.

This module handles both fresh ideation and refinement via the same run()
function — the prompt adapts based on context provided.
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
    feedback: str = "",
    previous_results: dict | None = None,
    original_idea: dict | None = None,
    revision_attempt: int = 0,
    max_revisions: int = 2,
) -> tuple[dict | None, object]:
    """Generate or refine a research idea (two invocations).

    Invocation 1: Generate proposal.md + idea.json + references/
    Invocation 2: Generate plan.json based on the proposal

    Returns:
        Tuple of (parsed idea dict or None, last AgentResult).
    """
    # ── Invocation 1: Idea ──
    idea_task = _build_idea_task(
        seed_topic=seed_topic,
        history=history,
        resources=resources,
        attempt=attempt,
        max_attempts=max_attempts,
        feedback=feedback,
        previous_results=previous_results,
        original_idea=original_idea,
        revision_attempt=revision_attempt,
        max_revisions=max_revisions,
    )

    idea_result = invoke_agent(
        agent_type=agent_type,
        task=idea_task,
        workspace=workspace,
        timeout=timeout,
        agent_config=agent_config,
    )

    # Check idea outputs before proceeding to plan
    idea = _parse_idea_output(workspace)
    if idea is None:
        return None, idea_result

    # ── Invocation 2: Plan ──
    plan_task = _build_plan_task(
        seed_topic=seed_topic,
        resources=resources,
        feedback=feedback,
    )

    plan_result = invoke_agent(
        agent_type=agent_type,
        task=plan_task,
        workspace=workspace,
        timeout=timeout,
        agent_config=agent_config,
    )

    # Check plan output
    _parse_plan_output(workspace, idea)

    return idea, plan_result


# ── Idea task prompt ──────────────────────────────────────────────


def _build_idea_task(
    seed_topic: str,
    history: list[dict] | None = None,
    resources: dict | None = None,
    attempt: int = 1,
    max_attempts: int = 5,
    feedback: str = "",
    previous_results: dict | None = None,
    original_idea: dict | None = None,
    revision_attempt: int = 0,
    max_revisions: int = 2,
) -> str:
    res = resources or {}
    resource_block = _build_resource_block(res)
    is_refinement = bool(feedback) and bool(original_idea)

    # ── Header ──
    if is_refinement:
        revisions_left = max_revisions - revision_attempt
        task = (
            "=== STAGE 1a: IDEA REFINEMENT ===\n\n"
            f"Your seed field is: {seed_topic}\n\n"
            f"BUDGET: Revision {revision_attempt}/{max_revisions}"
            f" ({revisions_left} revisions left)."
            f" Idea {attempt}/{max_attempts}.\n\n"
            "Your previous attempt needs improvement. Refine your idea\n"
            "based on the feedback below, then update the outputs.\n\n"
        )
    else:
        remaining = max_attempts - attempt
        task = (
            "=== STAGE 1a: IDEATION ===\n\n"
            f"Your seed field is: {seed_topic}\n\n"
            f"ATTEMPT: {attempt}/{max_attempts} "
            f"({'LAST CHANCE — make it count!' if remaining == 0 else f'{remaining} retries remaining if this idea fails'})\n\n"
            "Come up with a novel, concrete, and feasible research idea that could\n"
            "result in a publishable conference paper. Search the web extensively\n"
            "for related work before committing to an idea.\n\n"
        )

    # ── Body ──
    task += (
        "Read idea_guidelines.md carefully — it covers the full process from\n"
        "field exploration to structured output formats.\n\n"
        f"{resource_block}\n"
        "In this step, produce THREE outputs (experiment plan comes next):\n"
        "  1. proposal.md — full research proposal\n"
        "  2. idea.json — structured idea summary\n"
        "  3. references/ — parsed reference papers\n\n"
        "Do NOT write plan.json yet — that comes in the next step after\n"
        "your proposal is finalized.\n\n"
        "Aim for a proposal strong enough to be accepted at a top venue."
    )

    # ── Feedback ──
    if feedback:
        task += (
            "\n\n--- FEEDBACK (address these issues) ---\n"
            f"{feedback}\n"
            "--- END FEEDBACK ---\n"
        )

    # ── Original idea context ──
    if original_idea:
        idea_desc = original_idea.get("description", original_idea.get("title", "N/A"))
        idea_approach = original_idea.get("proposed_approach", "")
        task += (
            f"\n\nORIGINAL IDEA:\n{idea_desc}\n"
            f"APPROACH:\n{idea_approach}\n"
        )

    # ── Previous results ──
    if previous_results:
        task += "\n\nPREVIOUS EXPERIMENT RESULTS:\n"
        task += f"{json.dumps(previous_results, indent=2)}\n"

    # ── History ──
    if history:
        task += "\n\n--- PREVIOUS ATTEMPTS (learn from these) ---\n"
        for i, h in enumerate(history):
            task += (
                f"\nAttempt {i+1}: \"{h['idea'].get('description', h['idea'].get('title', 'N/A'))}\"\n"
                f"  Stage: {h['failure_stage']}\n"
                f"  Outcome: {h['failure_reason']}\n"
            )
            if h.get("best_score") is not None:
                task += f"  Best score: {h['best_score']:.1f}/10\n"
        task += (
            "\n--- Generate a DIFFERENT idea. Learn from what worked and what didn't "
            "in previous attempts. ---\n"
        )

    return task


# ── Plan task prompt ──────────────────────────────────────────────


def _build_plan_task(
    seed_topic: str,
    resources: dict | None = None,
    feedback: str = "",
) -> str:
    res = resources or {}
    resource_block = _build_resource_block(res)

    task = (
        "=== STAGE 1b: EXPERIMENT PLANNING ===\n\n"
        f"Your seed field is: {seed_topic}\n\n"
        "Read the proposal.md and idea.json you just created, along with\n"
        "the experiment design principles in idea_guidelines.md.\n\n"
        f"{resource_block}\n"
        "Design a detailed, step-by-step experiment plan and save it as plan.json.\n\n"
        "See idea_guidelines.md Step 4.2 for the exact format. Your plan should:\n"
        "- Cover environment setup, data preparation, baselines, main experiments,\n"
        "  ablations, evaluation, and visualization\n"
        "- Be detailed enough to follow without ambiguity\n"
        "- Include specific datasets, model architectures, hyperparameters,\n"
        "  evaluation metrics, and expected output formats\n"
        "- Be feasible within the resource constraints above\n"
        "- Account for parallel execution across available GPUs\n\n"
        "Apply the experiment design principles from idea_guidelines.md:\n"
        "  - At least 2 meaningful baselines (one simple, one strong)\n"
        "  - Ablation studies for each novel component\n"
        "  - At least 3 random seeds\n"
        "  - Clear success criteria\n"
    )

    if feedback:
        # If there's feedback about the plan specifically, include it
        plan_feedback = ""
        for line in feedback.split("\n"):
            lower = line.lower()
            if any(kw in lower for kw in ["plan", "experiment", "baseline", "ablation",
                                           "feasib", "scope", "budget", "runtime"]):
                plan_feedback += line + "\n"
        if plan_feedback:
            task += (
                "\n--- PLAN-RELATED FEEDBACK (address these) ---\n"
                f"{plan_feedback}"
                "--- END FEEDBACK ---\n"
            )

    return task


# ── Shared helpers ────────────────────────────────────────────────


def _build_resource_block(res: dict) -> str:
    platform = res.get("platform", "gpu")
    gpus = res.get("gpus", 0 if platform == "cpu" else 1)
    gpu_type = res.get("gpu_type", "GPU")
    gpu_mem = res.get("gpu_memory_gb", 80)
    cpus = res.get("cpus", 8)
    mem = res.get("memory_gb", 32)
    hours = res.get("time_hours", 8)

    if platform == "cpu" or gpus == 0:
        return (
            "COMPUTATIONAL RESOURCES (scope your idea to fit):\n"
            f"   - CPU: {cpus} cores\n"
            f"   - RAM: {mem}GB\n"
            f"   - Time limit: ~{hours} hours total for ALL experiments\n"
            "   - NO GPU available — your experiments must run on CPU only.\n"
            "   Your idea MUST be feasible within these constraints. Design\n"
            "   experiments that are analytical, algorithmic, or systems-level.\n"
            "   Avoid ideas that require training neural networks or GPU compute.\n"
            "   Prefer ideas that involve algorithm design, formal analysis,\n"
            "   systems benchmarking, program analysis, or similar CPU workloads.\n"
        )
    else:
        return (
            "COMPUTATIONAL RESOURCES (scope your idea to fit):\n"
            f"   - GPU: {gpus}x {gpu_type} ({gpu_mem}GB VRAM each)\n"
            f"   - RAM: {mem}GB\n"
            f"   - CPU: {cpus} cores\n"
            f"   - Time limit: ~{hours} hours total for ALL experiments\n"
            "   Your idea MUST be feasible within these constraints. Avoid ideas\n"
            "   that require training large models from scratch, massive datasets,\n"
            "   or multi-day compute. Prefer ideas that can show results with\n"
            "   small-to-medium scale experiments.\n"
        )


# ── Output parsing ────────────────────────────────────────────────


def _parse_idea_output(workspace: Path) -> dict | None:
    """Parse idea outputs after invocation 1 (proposal + idea + references)."""
    from rich.console import Console
    console = Console()

    idea_path = workspace / "idea.json"
    proposal_path = workspace / "proposal.md"

    if not idea_path.exists():
        console.print("  [red]idea.json not found.[/]")
        return None

    try:
        idea = json.loads(idea_path.read_text())
    except json.JSONDecodeError:
        console.print("  [red]idea.json is not valid JSON.[/]")
        return None

    missing = [f for f in REQUIRED_FIELDS if f not in idea]
    if missing:
        console.print(f"  [yellow]idea.json missing fields: {missing}[/]")
        return None

    if not proposal_path.exists():
        console.print("  [yellow]proposal.md not found — self-review may score lower.[/]")
    else:
        idea["_has_proposal"] = True

    return idea


def _parse_plan_output(workspace: Path, idea: dict) -> None:
    """Check plan.json after invocation 2 and annotate idea dict."""
    from rich.console import Console
    console = Console()

    plan_path = workspace / "plan.json"
    if not plan_path.exists():
        console.print("  [yellow]plan.json not found — experiments will be unstructured.[/]")
    else:
        try:
            plan = json.loads(plan_path.read_text())
            idea["_has_plan"] = True
            idea["_plan_steps"] = len(plan) if isinstance(plan, list) else 0
            console.print(f"  Plan: {idea['_plan_steps']} steps")
        except json.JSONDecodeError:
            console.print("  [yellow]plan.json is not valid JSON.[/]")
