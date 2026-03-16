"""Stage 2+3: Let the CLI agent implement and run experiments.

The agent gets the idea.json and workspace. It must:
  - Write experiment code
  - Run it
  - Save results to results.json
  - Optionally save figures to figures/

We merge design + run into one stage because the agent should be free to
iterate on its own code — write, run, fix, rerun — without us micromanaging.
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
) -> tuple[dict | None, object]:
    """Let the agent implement and run experiments.

    Expects idea.json to already exist in workspace.

    Returns:
        Tuple of (parsed results dict or None, AgentResult).
    """
    task = _build_task(workspace, prior_errors)

    agent_result = invoke_agent(
        agent_type=agent_type,
        task=task,
        workspace=workspace,
        timeout=timeout,
        agent_config=agent_config,
    )

    return _parse_output(workspace), agent_result


def _build_task(workspace: Path, prior_errors: list[str] | None) -> str:
    task = (
        "You are a researcher. Read the idea.json and research_guidelines.md "
        "in the current directory.\n\n"
        "Your job:\n"
        "1. Implement the experiments described in the idea\n"
        "2. Run them (you have access to GPUs)\n"
        "3. Save the results to results.json in the current directory\n"
        "4. Save any figures to a figures/ directory\n\n"
        "You have full autonomy — install packages, write multiple files, "
        "debug and fix errors, run multiple experiment variants, etc.\n\n"
        "The results.json should contain all quantitative results needed to "
        "write a paper (metrics, comparisons, ablations, etc.).\n\n"
        "CRITICAL: Every number in results.json MUST come from actually running "
        "the experiment code. DO NOT fabricate, hardcode, or manually write results. "
        "The review system verifies that results trace back to real code execution. "
        "If your method doesn't beat baselines, report that honestly."
    )

    if prior_errors:
        task += "\n\n--- PREVIOUS ATTEMPTS FAILED ---\n"
        for i, err in enumerate(prior_errors):
            task += f"\nAttempt {i+1} error:\n{err[:500]}\n"
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
