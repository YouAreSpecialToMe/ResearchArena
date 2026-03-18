"""Stage 4: Multi-source automated paper review.

Reviewer agents run as CLI agents with read-only workspace access.
They can read all code, logs, results, and the paper itself.

When evaluating multiple CLI agents (claude, codex, kimi, minimax), the agents
NOT under test serve as reviewers. E.g., if claude is the researcher, codex,
kimi, and minimax review.

Review sources (fully autonomous):
  1. Reference check  — verify citations are real (Semantic Scholar + CrossRef)
  2. paperreview.ai   — external online review (Stanford Agentic Reviewer)
  3. CLI agent reviewers — other agents with read-only workspace access
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_DOMAINS_WITH_TEMPLATES = {"ml", "systems", "databases", "pl", "theory", "security"}


def _get_reviewer_guidelines_path(domain: str = "ml") -> Path:
    """Return the path to reviewer guidelines, domain-specific if available."""
    if domain in _DOMAINS_WITH_TEMPLATES:
        domain_path = _TEMPLATES_DIR / domain / "reviewer_guidelines.md"
        if domain_path.exists():
            return domain_path
    return _TEMPLATES_DIR / "ml" / "reviewer_guidelines.md"

# Output format appended to every reviewer prompt
_REVIEW_OUTPUT_FORMAT = """
Output your review as a JSON object to stdout. Print ONLY the JSON:
{
    "scores": {
        "novelty": <int 1-10>,
        "soundness": <int 1-10>,
        "significance": <int 1-10>,
        "clarity": <int 1-10>,
        "reproducibility": <int 1-10>,
        "experimental_rigor": <int 1-10>,
        "references": <int 1-10>,
        "results_integrity": <int 1-10>
    },
    "overall_score": <int, one of: 0, 2, 4, 6, 8, 10>,
    "decision": "accept" | "reject",
    "summary": "<2-3 sentence summary of the paper>",
    "novelty_assessment": "<what you found when searching for existing work online>",
    "strengths": ["<strength1>", ...],
    "weaknesses": ["<weakness1>", ...],
    "detailed_feedback": "<paragraph of actionable feedback for revision>",
    "questions_for_authors": ["<question1>", ...],
    "integrity_check": "<brief sanity check on results consistency>"
}

overall_score uses ICLR scale: 10=seminal, 8=clear accept, 6=marginal (revision),
4=reject, 2=strong reject, 0=fabricated/trivial. Threshold is 8.
"""


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
    reviewer_agents: list[dict],
    paperreview_config: dict,
    venue: str = "NeurIPS",
    accept_threshold: float = 8.0,
    workspace: Path | None = None,
    docker_image: str = "researcharena/agent:latest",
    tracker=None,
    runtime: str = "docker",
    domain: str = "ml",
) -> ReviewResult:
    """Run all review sources and aggregate scores automatically.

    Args:
        paper_latex: LaTeX source (unused when reviewers are CLI agents with workspace)
        paper_pdf_path: Compiled PDF path (for paperreview.ai)
        reviewer_agents: List of reviewer agent configs, each with:
            - type: "claude", "codex", "kimi", "minimax"
            - name: human-readable label
            - model: (optional) model override
        paperreview_config: Config for paperreview.ai
        venue: Target venue name
        accept_threshold: Score threshold for acceptance
        workspace: Full agent workspace (mounted read-only for reviewers)
        docker_image: Docker image to use for reviewer containers

    Returns:
        ReviewResult with aggregated scores and automated decision
    """
    all_reviews = []

    # ── Pre-check: Reference verification ──
    console.print("\n[bold]Pre-check: Reference verification[/]")
    from researcharena.utils.reference_checker import check_references, save_reference_check

    if tracker:
        tracker.begin_action(stage="review", action="reference_check")
    ref_result = check_references(paper_latex, workspace=workspace)
    if workspace:
        save_reference_check(ref_result, workspace)
    if tracker:
        tracker.end_action(
            outcome="success",
            details=f"{ref_result.verified}/{ref_result.total} verified, {ref_result.unverified} fake",
        )

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
    if tracker:
        tracker.begin_action(stage="review", action="paperreview_ai")
    pr_review = _run_paperreview(paper_pdf_path, paperreview_config, venue)
    if pr_review:
        # Use the first available reviewer agent to score the qualitative review
        if reviewer_agents and workspace:
            console.print("  Scoring paperreview.ai review with agent...")
            scored = _score_qualitative_review(
                pr_review, reviewer_agents[0], workspace, venue, runtime,
                timeout=paperreview_config.get("scoring_timeout", 120),
                max_turns=paperreview_config.get("scoring_max_turns", 5),
            )
            if scored is not None:
                pr_review["overall_score"] = scored
                pr_review["decision"] = _score_to_decision(scored, accept_threshold)
                console.print(f"  Agent scored paperreview.ai: {scored}/10")

        all_reviews.append(pr_review)
        console.print(f"  Score: {pr_review.get('overall_score', 'N/A')}")
        if tracker:
            tracker.end_action(outcome="success", details=f"score={pr_review.get('overall_score')}")
    else:
        console.print("  [yellow]paperreview.ai review unavailable.[/]")
        if tracker:
            tracker.end_action(outcome="skipped", details="unavailable or not configured")

    # ── Source 2: CLI agent reviewers (independent, no prior context) ──
    console.print(
        f"\n[bold]Review Source 2: CLI agent reviewers "
        f"({len(reviewer_agents)} agents)[/]"
    )

    if not workspace:
        console.print("  [yellow]No workspace — skipping agent reviewers.[/]")
    else:
        for i, agent_cfg in enumerate(reviewer_agents):
            agent_name = agent_cfg.get("name", agent_cfg.get("type", f"Agent-{i+1}"))
            agent_type = agent_cfg["type"]
            console.print(f"  Running {agent_name} ({agent_type})...")

            if tracker:
                tracker.begin_action(
                    stage="review",
                    action=f"agent_review:{agent_name}",
                    agent_type=agent_type,
                    model=agent_cfg.get("model"),
                )

            try:
                agent_review, agent_result = _run_cli_reviewer(
                    agent_cfg=agent_cfg,
                    workspace=workspace,
                    venue=venue,
                    docker_image=docker_image,
                    runtime=runtime,
                    domain=domain,
                )
                reviewer_log_files = agent_result.log_files if agent_result else None

                if agent_review:
                    agent_review["source"] = f"agent:{agent_name}"
                    all_reviews.append(agent_review)
                    console.print(
                        f"    Score: {agent_review.get('overall_score', 'N/A')}, "
                        f"Decision: {agent_review.get('decision', 'N/A')}"
                    )
                    integrity = agent_review.get("integrity_check", "")
                    if integrity:
                        console.print(f"    Integrity: {integrity[:80]}")
                    if tracker:
                        tracker.end_action(
                            outcome="success",
                            details=f"score={agent_review.get('overall_score')}, decision={agent_review.get('decision')}",
                            log_files=reviewer_log_files,
                        )
                else:
                    console.print(f"    [red]No review produced.[/]")
                    if tracker:
                        tracker.end_action(
                            outcome="failure",
                            details="No review produced",
                            log_files=reviewer_log_files,
                        )
            except Exception as e:
                console.print(f"    [red]Failed: {e}[/]")
                if tracker:
                    tracker.end_action(outcome="failure", details=str(e)[:100])

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


# ── CLI agent reviewers ──────────────────────────────────────────────────


def _run_cli_reviewer(
    agent_cfg: dict,
    workspace: Path,
    venue: str,
    docker_image: str,
    runtime: str = "docker",
    domain: str = "ml",
) -> tuple[dict | None, object]:
    """Run a CLI agent as a reviewer with read-only workspace.

    The reviewer agent gets the full workspace (code, logs, results, paper)
    as read-only. It can read everything but cannot modify anything.
    The review is parsed from stdout (JSON output).

    Returns:
        Tuple of (parsed review dict or None, AgentResult).
    """
    from researcharena.utils.agent_runner import invoke_agent

    # Load domain-specific reviewer guidelines
    reviewer_guidelines_path = _get_reviewer_guidelines_path(domain)
    guidelines = ""
    if reviewer_guidelines_path.exists():
        guidelines = reviewer_guidelines_path.read_text()

    task = (
        f"You are a reviewer for {venue}. The workspace contains a research "
        f"paper and supporting materials.\n\n"
        f"Available files:\n"
        f"- paper.tex — the paper to review\n"
        f"- experiment code (.py files)\n"
        f"- logs/ — experiment execution logs\n"
        f"- results.json — raw experiment results\n"
        f"- figures/ — generated figures\n\n"
        f"FIRST: Read reviewer_guidelines.md for detailed review instructions.\n\n"
        f"Your job:\n"
        f"1. Read the paper and evaluate its scientific contribution\n"
        f"2. Search online (arXiv, Semantic Scholar, Google Scholar) to verify "
        f"the claimed novelty — check if similar work already exists\n"
        f"3. Check experiment code and logs as a sanity check on results\n"
        f"4. Write your review\n\n"
        f"{guidelines}\n\n"
        f"{_REVIEW_OUTPUT_FORMAT}"
    )

    # Copy reviewer guidelines into workspace (it's read-only so we need
    # to do this before mounting — the workspace already has idea_guidelines,
    # we add reviewer_guidelines alongside it)
    reviewer_guide_dest = workspace / "reviewer_guidelines.md"
    if not reviewer_guide_dest.exists() and reviewer_guidelines_path.exists():
        try:
            shutil.copy2(reviewer_guidelines_path, reviewer_guide_dest)
        except OSError:
            pass  # workspace might already be read-only from a previous reviewer

    agent_type = agent_cfg["type"]
    reviewer_config = {
        **agent_cfg,
        "docker_image": docker_image,
        "runtime": runtime,
    }

    result = invoke_agent(
        agent_type=agent_type,
        task=task,
        workspace=workspace,
        timeout=agent_cfg.get("review_timeout", 1800),
        agent_config=reviewer_config,
        readonly=True,
    )

    # Parse the review from stdout (since workspace is read-only, agent
    # can't write review.json — we parse it from the output instead)
    return _parse_review_from_output(result.stdout), result


def _parse_review_from_output(stdout: str) -> dict | None:
    """Extract the review JSON from the agent's stdout.

    Handles two output formats:
    1. Stream-json: review JSON is inside {"type":"assistant","message":{"content":[{"text":"..."}]}}
    2. Plain text: review JSON appears directly in stdout

    Searches for the last valid JSON containing 'overall_score' and 'decision'.
    """
    if not stdout:
        return None

    # Step 1: Extract all text content from stream-json events
    # This handles the case where the review is embedded in a text field
    text_content = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            # Extract text from assistant messages
            if event.get("type") == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        text_content.append(block["text"])
            # Also check result text
            elif event.get("type") == "result":
                result_text = event.get("result", "")
                if result_text:
                    text_content.append(result_text)
        except json.JSONDecodeError:
            # Not JSON — treat as plain text
            text_content.append(line)

    combined = "\n".join(text_content) if text_content else stdout

    # Step 2: Find review JSON in the extracted text using brace matching
    candidates = []
    brace_depth = 0
    json_start = None
    for i, c in enumerate(combined):
        if c == '{':
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif c == '}':
            brace_depth -= 1
            if brace_depth == 0 and json_start is not None:
                candidate = combined[json_start:i + 1]
                try:
                    data = json.loads(candidate)
                    if "overall_score" in data and "decision" in data:
                        candidates.append(data)
                except json.JSONDecodeError:
                    pass
                json_start = None

    # Return the LAST match (most likely the actual review, not an example)
    return candidates[-1] if candidates else None


# ── Helpers ──────────────────────────────────────────────────────────────


def _to_str(value) -> str:
    """Convert a value to string — joins lists with newlines."""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value) if value else ""


# ── Score qualitative reviews ─────────────────────────────────────────────


def _score_qualitative_review(
    pr_review: dict,
    agent_cfg: dict,
    workspace: Path,
    venue: str,
    runtime: str = "docker",
    timeout: int = 120,
    max_turns: int = 5,
) -> float | None:
    """Use a CLI agent to assign a numeric ICLR score to a qualitative review.

    The agent reads the review text and outputs a JSON with overall_score.
    This is a short, focused task — no web search or code reading needed.
    """
    from researcharena.utils.agent_runner import invoke_agent

    # Build review text from all available sections
    strengths = _to_str(pr_review.get("strengths", ""))
    weaknesses = _to_str(pr_review.get("weaknesses", ""))
    questions = _to_str(pr_review.get("questions_for_authors", ""))

    review_text = (
        f"Summary:\n{pr_review.get('summary', '')}\n\n"
        f"Strengths:\n{strengths}\n\n"
        f"Weaknesses:\n{weaknesses}\n\n"
        f"Detailed feedback:\n{pr_review.get('detailed_feedback', '')}\n\n"
        f"Questions for authors:\n{questions}\n\n"
        f"Overall assessment:\n{pr_review.get('overall_assessment', '')}\n"
    )

    task = (
        f"You are a meta-reviewer for {venue}. An external review system "
        f"(paperreview.ai) produced the following qualitative review of a "
        f"research paper, but did NOT assign a numeric score.\n\n"
        f"Read the review carefully and assign a score.\n\n"
        f"--- EXTERNAL REVIEW ---\n{review_text}\n--- END REVIEW ---\n\n"
        f"Based on this review, assign an overall score on the ICLR scale:\n"
        f"  10 = seminal, top 5%\n"
        f"  8 = clear accept, strong contribution\n"
        f"  6 = marginal, needs revision\n"
        f"  4 = below threshold, reject\n"
        f"  2 = strong reject\n"
        f"  0 = fabricated or trivial\n\n"
        f"Output ONLY a JSON object:\n"
        f'{{"overall_score": <int, one of 0/2/4/6/8/10>, '
        f'"reasoning": "<1-2 sentence justification>"}}\n'
    )

    agent_type = agent_cfg["type"]
    scorer_config = {
        **agent_cfg,
        "runtime": runtime,
        "max_turns": max_turns,
    }

    result = invoke_agent(
        agent_type=agent_type,
        task=task,
        workspace=workspace,
        timeout=timeout,
        agent_config=scorer_config,
        readonly=True,
    )

    # Parse score from stdout.
    # The output may be stream-json (nested inside event wrappers) or plain text.
    # In stream-json, quotes are escaped as \" so we check both forms.
    if result.stdout:
        import re
        # Strategy 1: find overall_score": N or overall_score\": N anywhere
        score_match = re.search(
            r'overall_score\\?"\s*:\s*(\d+)',
            result.stdout,
        )
        if score_match:
            score = int(score_match.group(1))
            if 0 <= score <= 10:
                return float(score)

        # Strategy 2: use the general review parser (extracts text from stream-json)
        parsed = _parse_review_from_output(result.stdout)
        if parsed and parsed.get("overall_score") is not None:
            return float(parsed["overall_score"])

    return None


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
        from researcharena.utils.paperreview import submit_and_wait

        result = submit_and_wait(
            pdf_path=paper_pdf_path,
            email_address=config["email"],
            email_password=config["email_password"],
            imap_server=config.get("imap_server", "imap.gmail.com"),
            venue=venue.lower(),
            poll_interval=config.get("poll_interval", 60),
            max_wait=config.get("max_wait", 3600),
        )

        overall = result.overall_score
        return {
            "source": "paperreview.ai",
            "scores": result.dimensions,
            "overall_score": overall,
            "decision": _score_to_decision(overall) if overall else "unknown",
            "summary": result.summary,
            "strengths": [result.strengths] if result.strengths else [],
            "weaknesses": [result.weaknesses] if result.weaknesses else [],
            "detailed_feedback": result.detailed_comments or "",
            "overall_assessment": result.overall_assessment or "",
            "questions_for_authors": [result.questions] if result.questions else [],
        }
    except Exception as e:
        console.print(f"  [red]paperreview.ai error: {e}[/]")
        return None


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

        w_str = _to_str(r.get("weaknesses", ""))[:100]

        table.add_row(source, f"{score}", dec, integrity, w_str)

    console.print(table)
    console.print(f"[bold]Aggregate: avg={avg_score:.1f}/10, decision={decision}[/]")


def _aggregate_feedback(reviews: list[dict]) -> str:
    parts = []
    for i, r in enumerate(reviews):
        source = r.get("source", "unknown")
        parts.append(f"--- Review {i+1} ({source}, score: {r.get('overall_score', 'N/A')}) ---")

        parts.append(f"  Weaknesses: {_to_str(r.get('weaknesses', ''))}")

        feedback = r.get("detailed_feedback", "")
        if feedback:
            parts.append(f"  Feedback: {feedback}")

        integrity = r.get("integrity_check", "")
        if integrity:
            parts.append(f"  Integrity: {integrity}")

        parts.append("")

    return "\n".join(parts)


def _score_to_decision(score: float, accept_threshold: float = 8.0) -> str:
    """Map score to accept/reject using ICLR-style threshold."""
    if score >= accept_threshold:
        return "accept"
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
