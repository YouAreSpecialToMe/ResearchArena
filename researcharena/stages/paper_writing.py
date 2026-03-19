"""Stage 3: Let the CLI agent write a paper from its own research.

The agent gets the workspace containing idea.json, results.json, and figures/.
It must produce paper.tex (and optionally compile to paper.pdf).

We do NOT separate analysis from writing — the agent should decide how to
structure its findings and present them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from researcharena.utils.agent_runner import invoke_agent


def run(
    agent_type: str,
    workspace: Path,
    venue: str = "neurips",
    timeout: int = 3600,
    agent_config: dict | None = None,
    revision_feedback: str | None = None,
    revision_attempt: int = 0,
    max_revisions: int = 2,
    idea_attempt: int = 1,
    max_ideas: int = 5,
    self_review_feedback: str = "",
) -> tuple[bool, object]:
    """Let the agent write a paper.

    Expects idea.json and results.json to exist in workspace.

    Returns:
        Tuple of (True if paper.tex was produced, AgentResult).
    """
    task = _build_task(venue, revision_feedback, revision_attempt, max_revisions, idea_attempt, max_ideas, self_review_feedback)

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


def _build_task(
    venue: str,
    revision_feedback: str | None,
    revision_attempt: int = 0,
    max_revisions: int = 2,
    idea_attempt: int = 1,
    max_ideas: int = 5,
    self_review_feedback: str = "",
) -> str:
    revisions_left = max_revisions - revision_attempt
    ideas_left = max_ideas - idea_attempt
    if revision_attempt > 0:
        budget_line = (
            f"BUDGET: Revision {revision_attempt}/{max_revisions}"
            f" ({revisions_left} revisions left)."
            f" Idea {idea_attempt}/{max_ideas}"
            f" ({ideas_left} new ideas left if this one is abandoned).\n\n"
        )
    else:
        budget_line = (
            f"BUDGET: Initial draft. Up to {max_revisions} revisions allowed"
            f" if reviewers request changes."
            f" Idea {idea_attempt}/{max_ideas}.\n\n"
        )

    task = (
        "=== STAGE 3: PAPER WRITING ===\n\n"
        f"{budget_line}"
        "Read paper_writing_guidelines.md for detailed instructions on "
        "paper structure, formatting, tables, figures, and references.\n\n"
        "The workspace contains:\n"
        "  - proposal.md — your research proposal (from Stage 1)\n"
        "  - plan.json — your experiment plan (from Stage 1)\n"
        "  - idea.json — your research idea summary (from Stage 1)\n"
        "  - results.json — your experiment results (from Stage 2)\n"
        "  - figures/ — generated figures (if any)\n"
        "  - references/ — parsed reference papers (if any)\n\n"
        "CRITICAL: EVERY reference you cite MUST be a real, verifiable publication. "
        "Search Semantic Scholar (semanticscholar.org) to find real papers. "
        "Fake references undermine scientific integrity.\n\n"
        f"Write a complete {venue.upper()}-format LaTeX paper and save it as paper.tex.\n\n"
        "Use the results.json data for all numbers in the paper — every number "
        "must match exactly. If you can, compile to PDF."
    )

    if revision_feedback:
        task += (
            "\n\n--- REVIEWER FEEDBACK (address these in your revision) ---\n"
            f"{revision_feedback}\n"
            "--- Revise the paper to address ALL reviewer concerns. ---\n"
        )

    if self_review_feedback:
        task += (
            "\n\n--- SELF-REVIEW FEEDBACK (address these issues) ---\n"
            f"{self_review_feedback}\n"
            "--- END FEEDBACK ---\n"
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
