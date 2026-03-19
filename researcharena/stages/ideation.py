"""Stage 1: Let the CLI agent come up with a research idea.

The agent gets a workspace and a seed field (e.g., "cv", "nlp"). It decides
on its own how to explore — searching papers, brainstorming, etc.

The agent produces three structured outputs:
  1. proposal.md — a research proposal with motivation, approach, related work
  2. plan.json — a detailed experiment plan with step-by-step instructions
  3. idea.json — structured idea summary (backward-compatible)
  4. references/ — directory of parsed reference papers (optional)
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
    platform = res.get("platform", "gpu")
    gpus = res.get("gpus", 0 if platform == "cpu" else 1)
    gpu_type = res.get("gpu_type", "GPU")
    gpu_mem = res.get("gpu_memory_gb", 80)
    cpus = res.get("cpus", 8)
    mem = res.get("memory_gb", 32)
    hours = res.get("time_hours", 8)

    remaining = max_attempts - attempt

    # Build resource description based on platform
    if platform == "cpu" or gpus == 0:
        resource_block = (
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
        resource_block = (
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

    task = (
        f"=== STAGE 1: IDEATION ===\n\n"
        f"Your seed field is: {seed_topic}\n\n"
        f"ATTEMPT: {attempt}/{max_attempts} "
        f"({'LAST CHANCE — make it count!' if remaining == 0 else f'{remaining} retries remaining if this idea fails'})\n\n"
        "Read idea_guidelines.md for the step-by-step process.\n\n"
        "Come up with a novel, concrete, and feasible research idea that could "
        "result in a publishable conference paper.\n\n"
        f"{resource_block}\n"
        "Your job is to produce a thorough research proposal AND a detailed\n"
        "experiment plan. Search the web extensively for related work.\n\n"
        "You MUST produce these THREE outputs:\n\n"
        "1. proposal.md — A research proposal with these sections:\n"
        "   - Introduction (context, problem, key insight, hypothesis)\n"
        "   - Proposed Approach (overview, method details, key innovations)\n"
        "   - Related Work (key papers, how your idea differs)\n"
        "   - Experiments (planned setup, benchmarks, metrics, expected results)\n"
        "   - Success Criteria (what would confirm/refute your hypothesis)\n"
        "   - References\n\n"
        "2. plan.json — A JSON array of experiment steps:\n"
        "   [\n"
        '     {"category": "Environment Configuration|Baseline Experiment|Main Experiment|Analysis Experiment",\n'
        '      "title": "short descriptive title",\n'
        '      "description": "what this step does and why",\n'
        '      "steps": {"step1": "detailed instruction", "step2": "...", ...}},\n'
        "     ...\n"
        "   ]\n"
        "   The plan should cover: environment setup, baselines, main experiments,\n"
        "   ablations, and evaluation. Each step must be detailed enough to follow\n"
        "   without ambiguity.\n\n"
        "3. idea.json — Structured summary with at least these fields:\n"
        "   - description: short description (1-3 sentences)\n"
        "   - title: paper title\n"
        "   - motivation: why this problem matters\n"
        "   - proposed_approach: high-level approach\n"
        "   - related_work: key papers and differentiation\n"
        "   - hypothesis: testable hypothesis\n"
        "   - success_criteria: what would confirm/refute the hypothesis\n\n"
        "Optionally, create a references/ directory with parsed reference papers.\n\n"
        "Your proposal and plan will be self-reviewed before experiments begin.\n"
        "A weak proposal or incomplete plan will be sent back for revision."
    )

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
        task += "PREVIOUS EXPERIMENT RESULTS:\n"
        task += f"{json.dumps(results, indent=2)}\n\n"

    task += (
        "YOUR TASK:\n"
        "1. Read the reviewer feedback carefully\n"
        "2. Decide how to strengthen your idea — you can:\n"
        "   - Refine the approach (fix methodological issues)\n"
        "   - Add new components or modify existing ones\n"
        "   - Plan additional experiments (ablations, baselines, datasets)\n"
        "   - Pivot the framing or motivation\n"
        "3. Update ALL structured outputs:\n"
        "   - idea.json: update fields, add 'revision_notes' explaining changes\n"
        "   - proposal.md: revise the relevant sections\n"
        "   - plan.json: update the experiment plan if the approach changed\n\n"
        f"After this, you will re-run experiments (~{hours}h budget) and rewrite the paper.\n"
        "Focus on addressing the specific reviewer concerns."
    )

    return task


def run_experiment_refinement(
    agent_type: str,
    workspace: Path,
    original_idea: dict,
    refine_reason: str,
    results: dict | None = None,
    timeout: int = 1800,
    agent_config: dict | None = None,
    resources: dict | None = None,
    refine_attempt: int = 1,
    max_refine: int = 3,
) -> tuple[dict | None, object]:
    """Let the agent refine an idea mid-experiment based on experimental findings.

    Unlike post-review refinement, this is triggered by the agent itself during
    experiments when it discovers the idea needs a different direction.

    Returns:
        Tuple of (refined idea dict or None, AgentResult).
    """
    task = _build_experiment_refinement_task(
        original_idea, refine_reason, results,
        resources, refine_attempt, max_refine,
    )

    agent_result = invoke_agent(
        agent_type=agent_type,
        task=task,
        workspace=workspace,
        timeout=timeout,
        agent_config=agent_config,
    )

    return _parse_output(workspace), agent_result


def _build_experiment_refinement_task(
    original_idea: dict,
    refine_reason: str,
    results: dict | None,
    resources: dict | None = None,
    refine_attempt: int = 1,
    max_refine: int = 3,
) -> str:
    res = resources or {}
    hours = res.get("time_hours", 8)
    refines_left = max_refine - refine_attempt

    idea_desc = original_idea.get("description", original_idea.get("title", "N/A"))
    idea_approach = original_idea.get("proposed_approach", "")

    task = (
        "=== IDEA REFINEMENT (mid-experiment) ===\n\n"
        f"BUDGET: Refinement {refine_attempt}/{max_refine}"
        f" ({refines_left} refinements left after this).\n\n"
        "During experiments, you determined the idea needs refinement.\n"
        "You now have a chance to rethink and revise the idea with the benefit\n"
        "of what you learned from running experiments.\n\n"
        "Read idea_guidelines.md for guidance on structuring your idea.\n\n"
        f"ORIGINAL IDEA:\n{idea_desc}\n\n"
        f"APPROACH:\n{idea_approach}\n\n"
        "--- REASON FOR REFINEMENT ---\n"
        f"{refine_reason}\n"
        "--- END REASON ---\n\n"
    )

    if results:
        task += "EXPERIMENT FINDINGS SO FAR:\n"
        task += f"{json.dumps(results, indent=2)}\n\n"

    # Check for any partial results or logs in workspace
    task += (
        "You have access to the full workspace — review any code, logs, and\n"
        "partial results from previous experiment attempts.\n\n"
        "YOUR TASK:\n"
        "1. Search for related work — your refined idea should be grounded\n"
        "   in what actually exists in the literature\n"
        "2. Analyze what went wrong or what could be improved\n"
        "3. Revise the idea — you can:\n"
        "   - Change the approach based on experimental findings\n"
        "   - Pivot to a more promising direction discovered during experiments\n"
        "   - Add or remove components that proved useful/useless\n"
        "   - Adjust the hypothesis or claims\n"
        "4. Update ALL structured outputs:\n"
        "   - idea.json: update fields, add 'refinement_notes' explaining changes\n"
        "   - proposal.md: revise the relevant sections\n"
        "   - plan.json: update the experiment plan if the approach changed\n\n"
        f"After this, you will re-run experiments (~{hours}h budget).\n"
        "Make sure your refined idea is feasible and addresses the issues found."
    )

    return task


def _parse_output(workspace: Path) -> dict | None:
    from rich.console import Console
    console = Console()

    idea_path = workspace / "idea.json"
    proposal_path = workspace / "proposal.md"
    plan_path = workspace / "plan.json"

    # Check for idea.json (required)
    if not idea_path.exists():
        console.print("  [red]idea.json not found.[/]")
        return None

    try:
        idea = json.loads(idea_path.read_text())
    except json.JSONDecodeError:
        console.print("  [red]idea.json is not valid JSON.[/]")
        return None

    # Check required fields
    missing = [f for f in REQUIRED_FIELDS if f not in idea]
    if missing:
        console.print(f"  [yellow]idea.json missing fields: {missing}[/]")
        return None

    # Check for structured outputs (warn if missing but don't fail)
    if not proposal_path.exists():
        console.print("  [yellow]proposal.md not found — self-review may score lower.[/]")
    else:
        idea["_has_proposal"] = True

    if not plan_path.exists():
        console.print("  [yellow]plan.json not found — experiments will be unstructured.[/]")
    else:
        try:
            plan = json.loads(plan_path.read_text())
            idea["_has_plan"] = True
            idea["_plan_steps"] = len(plan) if isinstance(plan, list) else 0
        except json.JSONDecodeError:
            console.print("  [yellow]plan.json is not valid JSON.[/]")

    return idea
