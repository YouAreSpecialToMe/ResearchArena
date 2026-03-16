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
) -> tuple[dict | None, object]:
    """Let the agent generate a research idea.

    Returns:
        Tuple of (parsed idea dict or None, AgentResult).
    """
    task = _build_task(seed_topic, history)

    agent_result = invoke_agent(
        agent_type=agent_type,
        task=task,
        workspace=workspace,
        timeout=timeout,
        agent_config=agent_config,
    )

    return _parse_output(workspace), agent_result


def _build_task(seed_topic: str, history: list[dict] | None) -> str:
    task = (
        f"=== STAGE 1: IDEATION ===\n\n"
        f"Your seed field is: {seed_topic}\n\n"
        "Read idea_guidelines.md for the step-by-step process: explore the "
        "field, search existing work online (arXiv, Semantic Scholar, Google "
        "Scholar), generate candidate ideas, and verify novelty.\n\n"
        "Come up with a novel, concrete, and feasible research idea that could "
        "result in a publishable ML conference paper.\n\n"
        "You have full autonomy — search the web, read papers, brainstorm, "
        "whatever you think is necessary.\n\n"
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
