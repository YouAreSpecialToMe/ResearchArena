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
        "Read paper_writing_guidelines.md for structure, formatting, and LaTeX tips.\n\n"
        "The workspace contains your prior work — reuse it:\n"
        "  - proposal.md — your research proposal with introduction, approach,\n"
        "    related work, and references. Use this as the foundation for your\n"
        "    paper — adapt and expand the content, don't start from scratch.\n"
        "  - references/ — parsed reference papers with BibTeX entries.\n"
        "    Use these for your bibliography.\n"
        "  - exp/ — experiment code and per-experiment results.\n"
        "    Use these for your experiments section.\n"
        "  - figures/ — generated figures. Include them in the paper.\n"
        "  - idea.json — idea summary for quick reference.\n"
        "  - plan.json — experiment plan for reference.\n\n"
        f"Write a complete {venue.upper()}-format LaTeX paper and save it as paper.tex.\n\n"
        "Key requirements:\n"
        "- Build on proposal.md — it already has your intro, approach, and related work\n"
        "- Every number must come from experiment results in exp/\n"
        "- Every reference must be real and verifiable\n"
        "- Compile to PDF if possible"
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
