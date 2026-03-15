"""Stage 6: Multi-source automated paper review.

Before sending to reviewers, we verify all references in the paper against
Semantic Scholar and CrossRef. Papers with high fake-reference rates get
penalized and the feedback includes which references need fixing.

Review sources (fully autonomous):
  1. Reference check  — verify citations are real (Semantic Scholar + CrossRef)
  2. paperreview.ai   — external online review (Stanford Agentic Reviewer)
  3. Agent reviewers  — independent CLI-agent LLMs (different models/providers)

Scores are aggregated and a decision is made automatically based on thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from autoresearch.utils.llm import LLMClient

console = Console()

AGENT_REVIEWER_SYSTEM_PROMPT = """\
You are an expert reviewer for a top-tier ML conference ({venue}). \
Review the submitted paper rigorously and fairly.

Evaluate on these criteria (each scored 1-10):
1. Novelty: Is the contribution new and non-trivial?
2. Soundness: Are the methods and experiments technically correct?
3. Significance: Does this work matter to the community?
4. Clarity: Is the paper well-written and easy to follow?
5. Reproducibility: Could someone replicate the experiments?
6. Experimental rigor: Are baselines fair? Are results statistically significant?

Return a JSON object:
{{
    "scores": {{
        "novelty": <int>,
        "soundness": <int>,
        "significance": <int>,
        "clarity": <int>,
        "reproducibility": <int>,
        "experimental_rigor": <int>
    }},
    "overall_score": <float>,
    "decision": "accept" | "weak_accept" | "borderline" | "weak_reject" | "reject",
    "summary": "<2-3 sentence summary>",
    "strengths": ["<strength1>", ...],
    "weaknesses": ["<weakness1>", ...],
    "detailed_feedback": "<paragraph of actionable feedback for revision>",
    "questions_for_authors": ["<question1>", ...]
}}
"""


@dataclass
class ReviewResult:
    reviews: list[dict]          # all individual reviews (agent + paperreview)
    avg_score: float             # average across all reviews
    decision: str                # automated decision based on score threshold
    aggregated_feedback: str     # all feedback combined for pipeline backtracking


# ── Main entry point ─────────────────────────────────────────────────────


def review_paper(
    paper_latex: str,
    paper_pdf_path: str | Path | None,
    agent_configs: list[dict],
    paperreview_config: dict,
    venue: str = "NeurIPS",
    accept_threshold: float = 6.0,
) -> ReviewResult:
    """Run all review sources and aggregate scores automatically.

    Args:
        paper_latex: LaTeX source (for agent reviewers)
        paper_pdf_path: Compiled PDF path (for paperreview.ai)
        agent_configs: List of agent reviewer configs, each with:
            - provider: "anthropic" or "openai"
            - model: model name
            - name: human-readable name (e.g. "Claude Sonnet", "GPT-4o")
            - temperature: (optional, default 0.3)
            - max_tokens: (optional, default 8192)
        paperreview_config: Config for paperreview.ai (email, password, etc.)
        venue: Target venue name
        accept_threshold: Score threshold for acceptance

    Returns:
        ReviewResult with aggregated scores and automated decision
    """
    all_reviews = []

    # ── Pre-check: Reference verification ──
    console.print("\n[bold]Pre-check: Reference verification[/]")
    from autoresearch.utils.reference_checker import check_references, save_reference_check

    ref_result = check_references(paper_latex)
    # Save to PDF directory if available, otherwise to a temp location
    ref_save_dir = Path(paper_pdf_path).parent if paper_pdf_path else None
    if ref_save_dir:
        save_reference_check(ref_result, ref_save_dir)

    # Any fake reference = automatic rejection
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

    # ── Source 2: Agent reviewers ──
    console.print(f"\n[bold]Review Source 2: Agent reviewers ({len(agent_configs)} agents)[/]")
    for i, agent_cfg in enumerate(agent_configs):
        agent_name = agent_cfg.get("name", f"Agent-{i+1}")
        console.print(f"  Running {agent_name} ({agent_cfg['provider']}/{agent_cfg['model']})...")
        try:
            agent_review = _run_agent_reviewer(agent_cfg, paper_latex, venue)
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

    # Fake references = automatic rejection, score zeroed
    if has_fake_refs:
        final_score = 0.0
        decision = "reject"
        console.print(
            f"  Reviewer avg: {avg_score:.1f}, but forced to 0 (fake references)"
        )
    else:
        final_score = avg_score
        decision = _score_to_decision(final_score, accept_threshold)

    # ── Display summary ──
    _display_review_summary(all_reviews, final_score, decision)

    # ── Aggregate feedback for pipeline backtracking ──
    aggregated = _aggregate_feedback(all_reviews)
    if ref_feedback:
        aggregated = ref_feedback + "\n\n" + aggregated

    return ReviewResult(
        reviews=all_reviews,
        avg_score=final_score,
        decision=decision,
        aggregated_feedback=aggregated,
    )


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


def _run_agent_reviewer(agent_cfg: dict, paper_latex: str, venue: str) -> dict:
    """Run a single agent reviewer and return its review dict."""
    llm = LLMClient(
        provider=agent_cfg["provider"],
        model=agent_cfg["model"],
        temperature=agent_cfg.get("temperature", 0.3),
        max_tokens=agent_cfg.get("max_tokens", 8192),
    )

    system = AGENT_REVIEWER_SYSTEM_PROMPT.format(venue=venue)
    user_msg = f"Paper to review:\n\n{paper_latex}"

    return llm.generate_json(system, user_msg)


# ── Display & aggregation ────────────────────────────────────────────────


def _display_review_summary(reviews: list[dict], avg_score: float, decision: str):
    """Print a summary table of all reviews."""
    table = Table(title="Review Summary")
    table.add_column("Source", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Decision")
    table.add_column("Key Weaknesses")

    for r in reviews:
        source = r.get("source", "unknown")
        score = r.get("overall_score", "N/A")
        dec = r.get("decision", "N/A")

        weaknesses = r.get("weaknesses", [])
        if isinstance(weaknesses, list):
            w_str = "; ".join(str(w)[:60] for w in weaknesses[:2])
        else:
            w_str = str(weaknesses)[:120]

        table.add_row(source, f"{score}", dec, w_str)

    console.print(table)
    console.print(f"[bold]Aggregate: avg={avg_score:.1f}/10, decision={decision}[/]")


def _aggregate_feedback(reviews: list[dict]) -> str:
    """Combine all review feedback for pipeline use."""
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
    # Filter out non-serializable fields from reviews
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
