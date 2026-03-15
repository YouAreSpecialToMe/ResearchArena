"""Stage 6: Multi-source automated paper review.

Before sending to reviewers, we verify all references against
Semantic Scholar and CrossRef. Any fake reference = auto reject.

Agent reviewers receive the full workspace (paper + experiment code + logs +
results) so they can judge whether results are genuine, not just whether
the paper reads well.

Review sources (fully autonomous):
  1. Reference check  — verify citations are real (Semantic Scholar + CrossRef)
  2. paperreview.ai   — external online review (Stanford Agentic Reviewer)
  3. Agent reviewers  — independent LLMs with access to all experiment artifacts
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from autoresearch.utils.llm import LLMClient

console = Console()

_REVIEWER_GUIDELINES_PATH = Path(__file__).parent.parent / "templates" / "reviewer_guidelines.md"

# Output format appended to every reviewer prompt
_REVIEW_OUTPUT_FORMAT = """\

Return a JSON object with your review:
{{
    "scores": {{
        "novelty": <int 1-10>,
        "soundness": <int 1-10>,
        "significance": <int 1-10>,
        "clarity": <int 1-10>,
        "reproducibility": <int 1-10>,
        "experimental_rigor": <int 1-10>,
        "results_integrity": <int 1-10>
    }},
    "overall_score": <float>,
    "decision": "accept" | "weak_accept" | "borderline" | "weak_reject" | "reject",
    "summary": "<2-3 sentence summary of the paper>",
    "strengths": ["<strength1>", ...],
    "weaknesses": ["<weakness1>", ...],
    "detailed_feedback": "<paragraph of actionable feedback for revision>",
    "questions_for_authors": ["<question1>", ...],
    "integrity_assessment": "<your verdict on whether results are genuine, based on code and logs>"
}}
"""


def _load_reviewer_system_prompt(venue: str) -> str:
    """Build the reviewer system prompt from the guidelines template."""
    guidelines = ""
    if _REVIEWER_GUIDELINES_PATH.exists():
        guidelines = _REVIEWER_GUIDELINES_PATH.read_text()

    return (
        f"You are an expert reviewer for a top-tier ML conference ({venue}).\n\n"
        f"{guidelines}\n\n"
        f"{_REVIEW_OUTPUT_FORMAT}"
    )


@dataclass
class ReviewResult:
    reviews: list[dict]
    avg_score: float
    decision: str
    aggregated_feedback: str


# ── Main entry point ─────────────────────────────────────────────────────


def review_paper(
    paper_latex: str,
    paper_pdf_path: str | Path | None,
    agent_configs: list[dict],
    paperreview_config: dict,
    venue: str = "NeurIPS",
    accept_threshold: float = 6.0,
    workspace: Path | None = None,
) -> ReviewResult:
    """Run all review sources and aggregate scores automatically.

    Args:
        paper_latex: LaTeX source (for agent reviewers)
        paper_pdf_path: Compiled PDF path (for paperreview.ai)
        agent_configs: List of agent reviewer configs
        paperreview_config: Config for paperreview.ai
        venue: Target venue name
        accept_threshold: Score threshold for acceptance
        workspace: Full agent workspace (code, logs, results) for reviewer access

    Returns:
        ReviewResult with aggregated scores and automated decision
    """
    all_reviews = []

    # ── Pre-check: Reference verification ──
    console.print("\n[bold]Pre-check: Reference verification[/]")
    from autoresearch.utils.reference_checker import check_references, save_reference_check

    ref_result = check_references(paper_latex)
    ref_save_dir = Path(paper_pdf_path).parent if paper_pdf_path else None
    if ref_save_dir:
        save_reference_check(ref_result, ref_save_dir)

    has_fake_refs = ref_result.unverified > 0
    ref_feedback = ""
    if has_fake_refs:
        fake_refs = [r for r in ref_result.references if r["status"] == "unverified"]
        ref_lines = [
            f"REFERENCE CHECK FAILED: {ref_result.unverified}/{ref_result.total} references "
            f"could NOT be verified. Paper is automatically rejected.",
            "",
            "Unverified references (must be replaced with real, citable papers):",
        ]
        for r in fake_refs:
            title = r.get("title", r.get("raw", "?"))[:80]
            ref_lines.append(f"  - {title}")
        ref_lines.append("")
        ref_lines.append(
            "Every reference in the paper must be a real, verifiable publication. "
            "Search Semantic Scholar or Google Scholar to find real papers, and "
            "cite them with correct titles, authors, and venues."
        )
        ref_feedback = "\n".join(ref_lines)
        console.print(f"  [red]REJECTED: {ref_result.unverified} fake reference(s) found.[/]")

    # ── Source 1: paperreview.ai ──
    console.print("\n[bold]Review Source 1: paperreview.ai[/]")
    pr_review = _run_paperreview(paper_pdf_path, paperreview_config, venue)
    if pr_review:
        all_reviews.append(pr_review)
        console.print(f"  Score: {pr_review.get('overall_score', 'N/A')}")
    else:
        console.print("  [yellow]paperreview.ai review unavailable.[/]")

    # ── Source 2: Agent reviewers (with full workspace access) ──
    workspace_context = _collect_workspace_context(workspace) if workspace else ""
    console.print(f"\n[bold]Review Source 2: Agent reviewers ({len(agent_configs)} agents)[/]")
    if workspace_context:
        console.print(f"  Workspace context: {len(workspace_context)} chars provided to reviewers")

    for i, agent_cfg in enumerate(agent_configs):
        agent_name = agent_cfg.get("name", f"Agent-{i+1}")
        console.print(f"  Running {agent_name} ({agent_cfg['provider']}/{agent_cfg['model']})...")
        try:
            agent_review = _run_agent_reviewer(
                agent_cfg, paper_latex, venue, workspace_context,
            )
            agent_review["source"] = f"agent:{agent_name}"
            agent_review["agent_config"] = {
                "provider": agent_cfg["provider"],
                "model": agent_cfg["model"],
            }
            all_reviews.append(agent_review)
            console.print(
                f"    Score: {agent_review.get('overall_score', 'N/A')}, "
                f"Decision: {agent_review.get('decision', 'N/A')}"
            )
            integrity = agent_review.get("integrity_assessment", "")
            if integrity:
                console.print(f"    Integrity: {integrity[:80]}")
        except Exception as e:
            console.print(f"    [red]Failed: {e}[/]")

    if not all_reviews:
        console.print("[red]No reviews obtained from any source.[/]")
        return ReviewResult(
            reviews=[],
            avg_score=0,
            decision="reject",
            aggregated_feedback="All review sources failed.",
        )

    # ── Aggregate scores ──
    scores = [r["overall_score"] for r in all_reviews if r.get("overall_score") is not None]
    avg_score = sum(scores) / len(scores) if scores else 5.0

    if has_fake_refs:
        final_score = 0.0
        decision = "reject"
        console.print(
            f"  Reviewer avg: {avg_score:.1f}, but forced to 0 (fake references)"
        )
    else:
        final_score = avg_score
        decision = _score_to_decision(final_score, accept_threshold)

    _display_review_summary(all_reviews, final_score, decision)

    aggregated = _aggregate_feedback(all_reviews)
    if ref_feedback:
        aggregated = ref_feedback + "\n\n" + aggregated

    return ReviewResult(
        reviews=all_reviews,
        avg_score=final_score,
        decision=decision,
        aggregated_feedback=aggregated,
    )


# ── Workspace context collection ─────────────────────────────────────────


def _collect_workspace_context(workspace: Path) -> str:
    """Collect experiment code, logs, and results from workspace for reviewers.

    Reads all relevant files and concatenates them into a single string
    that gets included in the reviewer prompt.
    """
    parts = []
    MAX_FILE_SIZE = 5000  # chars per file to avoid token limits

    # results.json
    results_path = workspace / "results.json"
    if results_path.exists():
        content = results_path.read_text(errors="replace")[:MAX_FILE_SIZE]
        parts.append(f"=== results.json ===\n{content}")

    # Experiment code (.py files, excluding hidden/venv)
    py_files = []
    for f in sorted(workspace.rglob("*.py")):
        rel = f.relative_to(workspace)
        if any(p.startswith(".") or p in ("logs", "__pycache__") for p in rel.parts):
            continue
        py_files.append(f)

    for f in py_files[:5]:  # limit to 5 code files
        rel = f.relative_to(workspace)
        content = f.read_text(errors="replace")[:MAX_FILE_SIZE]
        parts.append(f"=== {rel} ===\n{content}")

    # Stdout log (training evidence)
    stdout_log = workspace / "logs" / "agent_stdout.txt"
    if stdout_log.exists():
        content = stdout_log.read_text(errors="replace")
        # Take last 3000 chars (most relevant part of training)
        if len(content) > MAX_FILE_SIZE:
            content = f"[...truncated first {len(content) - MAX_FILE_SIZE} chars...]\n" + content[-MAX_FILE_SIZE:]
        parts.append(f"=== logs/agent_stdout.txt ===\n{content}")

    # Stderr log
    stderr_log = workspace / "logs" / "agent_stderr.txt"
    if stderr_log.exists():
        content = stderr_log.read_text(errors="replace")[-2000:]
        if content.strip():
            parts.append(f"=== logs/agent_stderr.txt (last 2000 chars) ===\n{content}")

    return "\n\n".join(parts)


# ── paperreview.ai ───────────────────────────────────────────────────────


def _run_paperreview(
    paper_pdf_path: str | Path | None,
    config: dict,
    venue: str,
) -> dict | None:
    """Submit to paperreview.ai and return normalized review dict."""
    if not config.get("email"):
        console.print("  [yellow]No email configured. Skipping paperreview.ai.[/]")
        return None

    if not paper_pdf_path or not Path(paper_pdf_path).exists():
        console.print(f"  [yellow]PDF not available at {paper_pdf_path}. Skipping.[/]")
        return None

    try:
        from autoresearch.utils.paperreview import submit_and_wait

        result = submit_and_wait(
            pdf_path=paper_pdf_path,
            email_address=config["email"],
            email_password=config["email_password"],
            imap_server=config.get("imap_server", "imap.gmail.com"),
            venue=venue.lower(),
            poll_interval=config.get("poll_interval", 60),
            max_wait=config.get("max_wait", 3600),
        )

        overall = result.overall_score or 5.0
        return {
            "source": "paperreview.ai",
            "scores": result.dimensions,
            "overall_score": overall,
            "decision": _score_to_decision(overall),
            "summary": result.summary,
            "strengths": [result.strengths] if result.strengths else [],
            "weaknesses": [result.weaknesses] if result.weaknesses else [],
            "detailed_feedback": result.detailed_comments or result.overall_assessment,
            "questions_for_authors": [result.questions] if result.questions else [],
        }
    except Exception as e:
        console.print(f"  [red]paperreview.ai error: {e}[/]")
        return None


# ── Agent reviewers ──────────────────────────────────────────────────────


def _run_agent_reviewer(
    agent_cfg: dict,
    paper_latex: str,
    venue: str,
    workspace_context: str,
) -> dict:
    """Run a single agent reviewer with full workspace access."""
    llm = LLMClient(
        provider=agent_cfg["provider"],
        model=agent_cfg["model"],
        temperature=agent_cfg.get("temperature", 0.3),
        max_tokens=agent_cfg.get("max_tokens", 8192),
    )

    system = _load_reviewer_system_prompt(venue)

    user_msg = f"Paper to review:\n\n{paper_latex}"

    if workspace_context:
        user_msg += (
            "\n\n"
            "========================================\n"
            "EXPERIMENT WORKSPACE (from the agent's Docker container)\n"
            "========================================\n\n"
            "The following files come from the same isolated Docker container "
            "where the research agent ran. The code, logs, and results were all "
            "produced inside that container — they cannot be selectively curated.\n\n"
            "Verify the chain: code → logs → results.json → paper.\n"
            "If any link is broken or results appear fabricated, score 0.\n\n"
            f"{workspace_context}"
        )

    return llm.generate_json(system, user_msg)


# ── Display & aggregation ────────────────────────────────────────────────


def _display_review_summary(reviews: list[dict], avg_score: float, decision: str):
    table = Table(title="Review Summary")
    table.add_column("Source", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Decision")
    table.add_column("Integrity")
    table.add_column("Key Weaknesses")

    for r in reviews:
        source = r.get("source", "unknown")
        score = r.get("overall_score", "N/A")
        dec = r.get("decision", "N/A")

        integrity = ""
        integrity_score = r.get("scores", {}).get("results_integrity")
        if integrity_score is not None:
            integrity = f"{integrity_score}/10"

        weaknesses = r.get("weaknesses", [])
        if isinstance(weaknesses, list):
            w_str = "; ".join(str(w)[:50] for w in weaknesses[:2])
        else:
            w_str = str(weaknesses)[:100]

        table.add_row(source, f"{score}", dec, integrity, w_str)

    console.print(table)
    console.print(f"[bold]Aggregate: avg={avg_score:.1f}/10, decision={decision}[/]")


def _aggregate_feedback(reviews: list[dict]) -> str:
    parts = []
    for i, r in enumerate(reviews):
        source = r.get("source", "unknown")
        parts.append(f"--- Review {i+1} ({source}, score: {r.get('overall_score', 'N/A')}) ---")

        weaknesses = r.get("weaknesses", [])
        if isinstance(weaknesses, list):
            for w in weaknesses:
                parts.append(f"  Weakness: {w}")
        else:
            parts.append(f"  Weakness: {weaknesses}")

        feedback = r.get("detailed_feedback", "")
        if feedback:
            parts.append(f"  Feedback: {feedback}")

        integrity = r.get("integrity_assessment", "")
        if integrity:
            parts.append(f"  Integrity: {integrity}")

        parts.append("")

    return "\n".join(parts)


def _score_to_decision(score: float, accept_threshold: float = 6.0) -> str:
    if score >= accept_threshold + 1:
        return "accept"
    elif score >= accept_threshold:
        return "weak_accept"
    elif score >= accept_threshold - 1:
        return "borderline"
    elif score >= accept_threshold - 2:
        return "weak_reject"
    else:
        return "reject"


def save_reviews(result: ReviewResult, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "reviews.json"
    clean_reviews = []
    for r in result.reviews:
        clean = {k: v for k, v in r.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        clean_reviews.append(clean)
    data = {
        "reviews": clean_reviews,
        "avg_score": result.avg_score,
        "decision": result.decision,
        "aggregated_feedback": result.aggregated_feedback,
    }
    path.write_text(json.dumps(data, indent=2))
    return path
