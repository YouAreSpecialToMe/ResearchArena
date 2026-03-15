"""Stage 4: Let the CLI agent write a paper from its own research.

The agent gets the workspace containing idea.json, results.json, and figures/.
It must produce paper.tex (and optionally compile to paper.pdf).

We do NOT separate analysis from writing — the agent should decide how to
structure its findings and present them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from autoresearch.utils.agent_runner import invoke_agent


def run(
    agent_type: str,
    workspace: Path,
    venue: str = "neurips",
    timeout: int = 3600,
    agent_config: dict | None = None,
    revision_feedback: str | None = None,
) -> tuple[bool, object]:
    """Let the agent write a paper.

    Expects idea.json and results.json to exist in workspace.

    Returns:
        Tuple of (True if paper.tex was produced, AgentResult).
    """
    task = _build_task(venue, revision_feedback)

    agent_result = invoke_agent(
        agent_type=agent_type,
        task=task,
        workspace=workspace,
        timeout=timeout,
        agent_config=agent_config,
    )

    # Try to compile if agent didn't
    paper_path = workspace / "paper.tex"
    if paper_path.exists():
        compile_paper(paper_path)
        return True, agent_result

    return False, agent_result


def _build_task(venue: str, revision_feedback: str | None) -> str:
    task = (
        "You are a researcher. The current directory contains idea.json with "
        "your research idea, results.json with your experiment results, and "
        "possibly a figures/ directory.\n\n"
        "FIRST: Read research_guidelines.md — especially the References section. "
        "EVERY reference you cite MUST be a real, verifiable publication. "
        "Search Semantic Scholar (semanticscholar.org) to find real papers. "
        "Fake references cause AUTOMATIC REJECTION.\n\n"
        f"Write a complete {venue.upper()}-format LaTeX paper and save it as paper.tex.\n\n"
        "You have full autonomy — structure the paper as you see fit. "
        "Include proper sections (abstract, introduction, method, experiments, "
        "conclusion, references). Include ablation studies and error bars. "
        "If you can, compile it to PDF."
    )

    if revision_feedback:
        task += (
            "\n\n--- REVIEWER FEEDBACK (address these in your revision) ---\n"
            f"{revision_feedback}\n"
            "--- Revise the paper to address ALL reviewer concerns. ---\n"
        )

    return task


def compile_paper(paper_path: Path) -> Path | None:
    """Try to compile LaTeX to PDF."""
    try:
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", paper_path.name],
                cwd=paper_path.parent,
                capture_output=True,
                timeout=120,
            )
        pdf_path = paper_path.with_suffix(".pdf")
        if pdf_path.exists():
            return pdf_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None
