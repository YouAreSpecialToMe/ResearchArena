"""Pipeline orchestrator for autonomous research by CLI agents.

The pipeline tests whether a CLI agent (Claude Code, Codex, etc.) can conduct
end-to-end research autonomously. We provide minimal scaffolding — the agent
does the actual thinking, coding, and writing.

Three stages (each is a single CLI agent invocation):
  1. IDEATION     — agent gets seed topic, produces idea.json
  2. EXPERIMENTS  — agent gets idea.json, produces results.json + figures/
  3. PAPER        — agent gets idea + results, produces paper.tex

Then we evaluate externally (paperreview.ai + agent reviewers).

Iteration: if the paper doesn't pass review, we can either:
  - Retry paper writing with feedback
  - Retry experiments with error context
  - Abandon idea and try a new one
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autoresearch.stages import (
    experiment_design as experiments,
    ideation,
    paper_writing,
    review,
)
from autoresearch.utils.action_parser import annotate_with_workspace_diff, parse_agent_stdout
from autoresearch.utils.tracker import RunTracker, TokenUsage
from autoresearch.utils.workspace_diff import snapshot as ws_snapshot, diff as ws_diff

console = Console()


class Stage(Enum):
    IDEATION = "ideation"
    EXPERIMENTS = "experiments"
    PAPER = "paper"
    REVIEW = "review"
    ACCEPTED = "accepted"
    FAILED = "failed"


@dataclass
class BestPaper:
    """Snapshot of the highest-scored paper across the entire run."""
    score: float = 0.0
    idea: dict | None = None
    paper_pdf_path: Path | None = None
    review_result: review.ReviewResult | None = None
    workspace: Path | None = None


@dataclass
class PipelineState:
    """Tracks the full state of the pipeline across iterations."""
    stage: Stage = Stage.IDEATION
    global_step: int = 0

    # Current artifacts
    idea: dict | None = None
    results: dict | None = None
    review_result: review.ReviewResult | None = None
    workspace: Path | None = None  # current idea's workspace

    # Best paper across all ideas (never reset)
    best: BestPaper = field(default_factory=BestPaper)

    # Failure tracking
    idea_history: list[dict] = field(default_factory=list)
    experiment_errors: list[str] = field(default_factory=list)

    # Per-idea counters
    experiment_attempts: int = 0
    paper_revision_attempts: int = 0

    # Per-seed counter
    idea_attempts: int = 0

    # Limits
    max_experiment_retries: int = 3
    max_paper_revisions: int = 2
    max_ideas_per_seed: int = 5
    max_global_steps: int = 30


class Pipeline:
    """Orchestrates the research pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.agent_type = config["agent"]["type"]
        self.agent_config = config["agent"]

        self.base_dir = Path(config["experiment"]["workspace"])
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = RunTracker()

        self.state = PipelineState(
            max_experiment_retries=config["experiment"].get("max_experiment_retries_per_idea", 3),
            max_paper_revisions=config["paper"].get("max_revisions", 2),
            max_ideas_per_seed=config["pipeline"]["max_ideas_per_seed"],
            max_global_steps=config["pipeline"]["max_global_steps"],
        )

    def run(self) -> dict:
        seed_topic = self.config["seed_topic"]
        accept_threshold = self.config["review"]["accept_threshold"]

        console.print(Panel(
            f"[bold]AutoResearch — CLI Agent Benchmark[/]\n"
            f"Agent: {self.agent_type} ({self.agent_config.get('model', 'default')})\n"
            f"Seed: {seed_topic}\n"
            f"Accept threshold: {accept_threshold}/10\n"
            f"Max ideas: {self.state.max_ideas_per_seed}",
            style="green",
        ))

        self.tracker.start_run()
        start_time = time.time()

        while self.state.stage not in (Stage.ACCEPTED, Stage.FAILED):
            if self.state.global_step >= self.state.max_global_steps:
                console.print("[red]Hit global step limit. Stopping.[/]")
                self.state.stage = Stage.FAILED
                break

            self.state.global_step += 1
            stage = self.state.stage
            console.print(f"\n[bold cyan]Step {self.state.global_step}: {stage.value}[/]")

            if stage == Stage.IDEATION:
                self._run_ideation(seed_topic)
            elif stage == Stage.EXPERIMENTS:
                self._run_experiments()
            elif stage == Stage.PAPER:
                self._run_paper()
            elif stage == Stage.REVIEW:
                self._run_review(accept_threshold)

        self.tracker.end_run()
        elapsed = time.time() - start_time
        self._print_summary(elapsed)

        # Save tracker data
        self.tracker.save(self.base_dir)

        return self._build_summary(elapsed)

    # ── Stage handlers ──────────────────────────────────────────────────

    def _run_ideation(self, seed_topic: str):
        if self.state.idea_attempts >= self.state.max_ideas_per_seed:
            console.print(f"[red]Exhausted idea budget ({self.state.max_ideas_per_seed}).[/]")
            self.state.stage = Stage.FAILED
            return

        self.state.idea_attempts += 1
        workspace = self.base_dir / f"idea_{self.state.idea_attempts:02d}"
        console.print(f"  Idea {self.state.idea_attempts}/{self.state.max_ideas_per_seed}")

        self.tracker.begin_action(
            stage="ideation",
            action="generate_idea",
            agent_type=self.agent_type,
            model=self.agent_config.get("model"),
            attempt=self.state.idea_attempts,
        )

        before = ws_snapshot(workspace)

        idea, agent_result = ideation.run(
            agent_type=self.agent_type,
            seed_topic=seed_topic,
            workspace=workspace,
            history=self.state.idea_history or None,
            timeout=self.config["agent"].get("ideation_timeout", 1800),
            agent_config=self.agent_config,
        )

        tokens, log_files, sub_actions, diff_dict, fail_cat = self._extract_tracking(
            agent_result, ws_before=before, workspace=workspace,
        )

        if idea is None:
            console.print("  [red]Agent failed to produce idea.json.[/]")
            self.tracker.end_action(
                outcome="failure",
                details="No valid idea.json produced",
                tokens=tokens,
                log_files=log_files,
                sub_actions=sub_actions,
                workspace_diff=diff_dict,
                failure_category=fail_cat,
            )
            self.state.idea_history.append({
                "idea": {"title": "(no idea produced)"},
                "failure_stage": "ideation",
                "failure_reason": "Agent did not produce a valid idea.json",
                "best_score": None,
            })
            self.state.stage = Stage.IDEATION  # try again
            return

        title = idea.get('title', 'N/A')
        console.print(f"  Title: [green]{title}[/]")
        self.tracker.end_action(
            outcome="success",
            details=title[:80],
            tokens=tokens,
            log_files=log_files,
            sub_actions=sub_actions,
            workspace_diff=diff_dict,
        )

        # Reset per-idea state
        self.state.idea = idea
        self.state.results = None
        self.state.review_result = None
        self.state.workspace = workspace
        self.state.experiment_errors = []
        self.state.experiment_attempts = 0
        self.state.paper_revision_attempts = 0

        self.state.stage = Stage.EXPERIMENTS

    def _run_experiments(self):
        self.state.experiment_attempts += 1

        if self.state.experiment_attempts > self.state.max_experiment_retries:
            console.print("  [red]Experiment budget exhausted. Abandoning idea.[/]")
            self._abandon_idea("experiments", (
                f"Experiments failed after {self.state.max_experiment_retries} attempts."
            ))
            return

        console.print(
            f"  Experiment attempt {self.state.experiment_attempts}/"
            f"{self.state.max_experiment_retries}"
        )

        self.tracker.begin_action(
            stage="experiments",
            action="run_experiments",
            agent_type=self.agent_type,
            model=self.agent_config.get("model"),
            attempt=self.state.experiment_attempts,
        )

        before = ws_snapshot(self.state.workspace)

        results, agent_result = experiments.run(
            agent_type=self.agent_type,
            workspace=self.state.workspace,
            timeout=self.config["experiment"].get("max_gpu_hours", 4) * 3600,
            agent_config=self.agent_config,
            prior_errors=self.state.experiment_errors or None,
        )

        tokens, log_files, sub_actions, diff_dict, fail_cat = self._extract_tracking(
            agent_result, ws_before=before, workspace=self.state.workspace,
        )

        if results is None:
            error_log = self._collect_error_log()
            self.state.experiment_errors.append(error_log)
            console.print("  [red]Agent failed to produce results.json.[/]")
            self.tracker.end_action(
                outcome="failure",
                details="No results.json produced",
                tokens=tokens,
                log_files=log_files,
                sub_actions=sub_actions,
                workspace_diff=diff_dict,
                failure_category=fail_cat,
            )
            self.state.stage = Stage.EXPERIMENTS  # retry
            return

        console.print("  [green]Experiments succeeded.[/]")
        self.tracker.end_action(
            outcome="success",
            details="Experiments completed, results.json written",
            tokens=tokens,
            log_files=log_files,
            sub_actions=sub_actions,
            workspace_diff=diff_dict,
        )
        self.state.results = results
        self.state.stage = Stage.PAPER

    def _run_paper(self):
        console.print("  Writing paper...")

        revision_feedback = None
        if self.state.review_result and self.state.paper_revision_attempts > 0:
            revision_feedback = self.state.review_result.aggregated_feedback

        is_revision = self.state.paper_revision_attempts > 0
        self.tracker.begin_action(
            stage="paper",
            action="revise_paper" if is_revision else "write_paper",
            agent_type=self.agent_type,
            model=self.agent_config.get("model"),
            attempt=self.state.paper_revision_attempts + 1,
        )

        before = ws_snapshot(self.state.workspace)

        success, agent_result = paper_writing.run(
            agent_type=self.agent_type,
            workspace=self.state.workspace,
            venue=self.config["paper"].get("template", "neurips"),
            timeout=self.config["agent"].get("paper_timeout", 3600),
            agent_config=self.agent_config,
            revision_feedback=revision_feedback,
        )

        tokens, log_files, sub_actions, diff_dict, fail_cat = self._extract_tracking(
            agent_result, ws_before=before, workspace=self.state.workspace,
        )

        if not success:
            console.print("  [red]Agent failed to produce paper.tex.[/]")
            self.tracker.end_action(
                outcome="failure",
                details="No paper.tex produced",
                tokens=tokens,
                log_files=log_files,
                sub_actions=sub_actions,
                workspace_diff=diff_dict,
                failure_category=fail_cat,
            )
            self._abandon_idea("paper", "Agent did not produce paper.tex.")
            return

        console.print("  [green]Paper written.[/]")
        self.tracker.end_action(
            outcome="success",
            details="revision" if is_revision else "initial draft",
            tokens=tokens,
            log_files=log_files,
            sub_actions=sub_actions,
            workspace_diff=diff_dict,
        )
        self.state.stage = Stage.REVIEW

    def _run_review(self, accept_threshold: float):
        console.print("  Collecting reviews...")

        paper_tex = self.state.workspace / "paper.tex"
        paper_pdf = self.state.workspace / "paper.pdf"
        latex = paper_tex.read_text(errors="replace") if paper_tex.exists() else ""

        # Auto-select reviewer agents: all configured agents EXCEPT the one under test
        reviewer_agents = [
            a for a in self.config["review"].get("agents", [])
            if a.get("type") != self.agent_type
        ]
        console.print(
            f"  Researcher: {self.agent_type} → "
            f"Reviewers: {[a.get('name', a.get('type')) for a in reviewer_agents]}"
        )

        # Review sub-actions (reference_check, paperreview_ai, agent_review)
        # are individually tracked inside review.review_paper via the tracker.
        result = review.review_paper(
            paper_latex=latex,
            paper_pdf_path=paper_pdf if paper_pdf.exists() else None,
            reviewer_agents=reviewer_agents,
            paperreview_config=self.config["review"].get("paperreview", {}),
            venue=self.config["paper"].get("template", "neurips"),
            accept_threshold=accept_threshold,
            workspace=self.state.workspace,
            docker_image=self.agent_config.get("docker_image", "autoresearch/agent:latest"),
            tracker=self.tracker,
        )
        review.save_reviews(result, self.state.workspace)
        self.state.review_result = result

        # Track best
        if result.avg_score > self.state.best.score:
            self.state.best = BestPaper(
                score=result.avg_score,
                idea=self.state.idea,
                paper_pdf_path=paper_pdf if paper_pdf.exists() else None,
                review_result=result,
                workspace=self.state.workspace,
            )
            console.print(f"  [green]New best! Score: {result.avg_score:.1f}/10[/]")

        console.print(f"  Score: {result.avg_score:.1f}/10, Decision: {result.decision}")

        if result.avg_score >= accept_threshold:
            console.print(Panel("[bold green]ACCEPTED![/]", style="green"))
            self.state.stage = Stage.ACCEPTED

        elif result.avg_score < accept_threshold - 2:
            console.print("  [yellow]→ Score far below threshold. Abandoning idea.[/]")
            self._abandon_idea("review", (
                f"Score {result.avg_score:.1f} (threshold {accept_threshold}). "
                f"Feedback: {result.aggregated_feedback[:500]}"
            ))

        elif self.state.paper_revision_attempts < self.state.max_paper_revisions:
            self.state.paper_revision_attempts += 1
            console.print(
                f"  [yellow]→ Revising paper "
                f"({self.state.paper_revision_attempts}/{self.state.max_paper_revisions})[/]"
            )
            self.state.stage = Stage.PAPER

        else:
            console.print("  [yellow]→ Revisions exhausted. Abandoning idea.[/]")
            self._abandon_idea("review", (
                f"Score {result.avg_score:.1f} after {self.state.paper_revision_attempts} revisions. "
                f"Feedback: {result.aggregated_feedback[:500]}"
            ))

    # ── Helpers ──────────────────────────────────────────────────────────

    def _abandon_idea(self, failure_stage: str, failure_reason: str):
        best_score = None
        if self.state.review_result:
            best_score = self.state.review_result.avg_score

        self.state.idea_history.append({
            "idea": self.state.idea or {"title": "(none)"},
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
            "best_score": best_score,
        })
        self.state.stage = Stage.IDEATION

    def _extract_tracking(self, agent_result, ws_before=None, workspace=None):
        """Extract tokens, log_files, sub_actions, workspace_diff, and failure_category."""
        if not agent_result:
            return TokenUsage(), None, None, None, None
        tokens = RunTracker.parse_tokens_from_stdout(agent_result.stdout)
        log_files = agent_result.log_files
        events_path = log_files.get("events") if log_files else None
        sub_actions_raw = parse_agent_stdout(
            self.agent_type, agent_result.stdout, events_path=events_path,
        )

        # Compute workspace diff
        diff_dict = None
        if ws_before is not None and workspace is not None:
            ws_after = ws_snapshot(workspace)
            d = ws_diff(ws_before, ws_after)
            if d.total_changes > 0:
                diff_dict = d.to_dict()

        # Cross-reference: annotate sub-actions with actual file changes
        if sub_actions_raw and diff_dict:
            annotate_with_workspace_diff(sub_actions_raw, diff_dict)

        sub_actions = [s.to_dict() for s in sub_actions_raw] if sub_actions_raw else None
        failure_category = agent_result.failure_category

        return tokens, log_files, sub_actions, diff_dict, failure_category

    def _collect_error_log(self) -> str:
        log_dir = self.state.workspace / "logs"
        if not log_dir.exists():
            return "No error log found"
        # Find the most recent stderr file (named {agent}_{timestamp}_stderr.txt)
        stderr_files = sorted(log_dir.glob("*_stderr.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if stderr_files:
            return stderr_files[0].read_text()[-2000:]
        return "No error log found"

    def _print_summary(self, elapsed: float):
        table = Table(title="Pipeline Summary")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Agent", f"{self.agent_type} ({self.agent_config.get('model', 'default')})")
        table.add_row("Final state", self.state.stage.value)
        table.add_row("Total steps", str(self.state.global_step))
        table.add_row("Ideas tried", f"{self.state.idea_attempts}/{self.state.max_ideas_per_seed}")
        table.add_row("Wall time", f"{elapsed:.0f}s")

        total_tokens = self.tracker.total_tokens
        if total_tokens.total:
            table.add_row("Total tokens", f"{total_tokens.total:,} (in: {total_tokens.input_tokens:,}, out: {total_tokens.output_tokens:,})")
        if self.tracker.total_cost > 0:
            table.add_row("Est. cost", f"${self.tracker.total_cost:.2f}")

        best = self.state.best
        if best.idea:
            table.add_row("Best paper", best.idea.get("title", "N/A"))
            table.add_row("Best score", f"{best.score:.1f}/10")
            table.add_row("Best workspace", str(best.workspace))
        else:
            table.add_row("Best paper", "[red]None — no paper produced[/]")

        console.print(table)

        # Print tracker tables
        self.tracker.print_action_log()
        self.tracker.print_stage_summary()

        if self.state.idea_history:
            hist = Table(title="Idea History")
            hist.add_column("#")
            hist.add_column("Title")
            hist.add_column("Failed at")
            hist.add_column("Score")
            for i, h in enumerate(self.state.idea_history):
                score = h.get("best_score")
                hist.add_row(
                    str(i + 1),
                    h["idea"].get("title", "N/A")[:50],
                    h["failure_stage"],
                    f"{score:.1f}" if score is not None else "-",
                )
            console.print(hist)

    def _build_summary(self, elapsed: float) -> dict:
        best = self.state.best
        return {
            "agent": self.agent_type,
            "agent_model": self.agent_config.get("model"),
            "status": self.state.stage.value,
            "total_steps": self.state.global_step,
            "ideas_tried": self.state.idea_attempts,
            "wall_time_seconds": round(elapsed),
            "tracker": self.tracker.to_dict(),
            "best_paper": {
                "title": best.idea.get("title"),
                "score": best.score,
                "workspace": str(best.workspace),
            } if best.idea else None,
            "idea_history": [
                {
                    "title": h["idea"].get("title"),
                    "failure_stage": h["failure_stage"],
                    "best_score": h.get("best_score"),
                }
                for h in self.state.idea_history
            ],
        }
