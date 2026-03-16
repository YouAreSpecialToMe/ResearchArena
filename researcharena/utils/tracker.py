"""Run tracker for recording time, token usage, and actions across the pipeline.

Tracks every stage invocation with:
  - Wall time per action and per stage
  - Token usage (parsed from CLI agent stdout when available)
  - What the agent did (stage, attempt number, outcome)
  - Cumulative totals and cost estimates

Saves a structured tracker.json alongside summary.json.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# Approximate pricing per 1M tokens (USD).
#
# For CLI agents running on subscriptions (Claude Code on Max plan, Codex on
# ChatGPT Pro), the actual cost is the subscription fee, not per-token.
# These API prices are included for reference and for agents that DO use
# per-token billing (Kimi, MiniMax via API keys).
#
# Token counts are always tracked regardless — they measure efficiency.
_PRICING = {
    # Anthropic (API pricing — Claude Code subscription is flat-rate)
    "claude-opus-4-6":   {"input": 15.0, "output": 75.0},
    "claude-opus-4-5":   {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0,  "output": 15.0},
    "claude-sonnet-4-5": {"input": 3.0,  "output": 15.0},
    "claude-haiku-4-5":  {"input": 0.80, "output": 4.0},
    # OpenAI (API pricing — Codex subscription is flat-rate)
    "gpt-4o":            {"input": 2.50, "output": 10.0},
    "gpt-4o-mini":       {"input": 0.15, "output": 0.60},
    "o3":                {"input": 10.0, "output": 40.0},
    "o3-mini":           {"input": 1.10, "output": 4.40},
    # Kimi (Moonshot AI) — per-token billing
    "moonshot-v1-auto":    {"input": 0.84, "output": 0.84},
    "moonshot-v1-8k":     {"input": 0.84, "output": 0.84},
    "moonshot-v1-32k":    {"input": 1.68, "output": 1.68},
    "moonshot-v1-128k":   {"input": 4.20, "output": 4.20},
    "kimi-k2":            {"input": 1.0,  "output": 3.0},
    # MiniMax — per-token billing
    "MiniMax-Text-01":    {"input": 1.0,  "output": 5.50},
    "abab6.5s-chat":      {"input": 0.70, "output": 0.70},
    "abab6.5-chat":       {"input": 2.10, "output": 2.10},
}

# Agents that run on flat-rate subscriptions. Cost estimate is shown as
# "API-equivalent" for comparison, but the actual cost is the subscription fee.
_SUBSCRIPTION_AGENTS = {"claude", "codex"}


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total,
        }


@dataclass
class ActionRecord:
    """A single tracked action (one agent invocation or review call)."""
    stage: str
    action: str
    agent_type: str | None = None
    model: str | None = None
    attempt: int | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed_seconds: float = 0.0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    outcome: str = ""  # "success", "failure", "timeout", etc.
    failure_category: str | None = None  # "oom", "timeout", "crash", "rate_limit", etc.
    details: str = ""  # brief description of what happened
    cost_usd: float = 0.0
    log_files: dict[str, str] | None = None

    def to_dict(self) -> dict:
        is_sub = self.agent_type in _SUBSCRIPTION_AGENTS if self.agent_type else False
        d = {
            "stage": self.stage,
            "action": self.action,
            "agent_type": self.agent_type,
            "model": self.model,
            "attempt": self.attempt,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "tokens": self.tokens.to_dict(),
            "outcome": self.outcome,
            "details": self.details,
            "cost_usd": round(self.cost_usd, 4),
            "billing": "subscription" if is_sub else "api",
        }
        if self.failure_category:
            d["failure_category"] = self.failure_category
        if self.log_files:
            d["log_files"] = self.log_files
        return d


class RunTracker:
    """Tracks all actions, timing, and token usage for a pipeline run."""

    def __init__(self):
        self.actions: list[ActionRecord] = []
        self.run_start: float = 0.0
        self.run_end: float = 0.0
        self._current: ActionRecord | None = None

    def start_run(self):
        self.run_start = time.time()

    def end_run(self):
        self.run_end = time.time()

    # ── Action lifecycle ──────────────────────────────────────────────

    def begin_action(
        self,
        stage: str,
        action: str,
        agent_type: str | None = None,
        model: str | None = None,
        attempt: int | None = None,
    ) -> ActionRecord:
        record = ActionRecord(
            stage=stage,
            action=action,
            agent_type=agent_type,
            model=model,
            attempt=attempt,
            start_time=time.time(),
        )
        self._current = record
        return record

    def end_action(
        self,
        outcome: str,
        details: str = "",
        tokens: TokenUsage | None = None,
        log_files: dict[str, str] | None = None,
        failure_category: str | None = None,
    ):
        if self._current is None:
            return
        record = self._current
        record.end_time = time.time()
        record.elapsed_seconds = record.end_time - record.start_time
        record.outcome = outcome
        record.details = details
        if tokens:
            record.tokens = tokens
        if log_files:
            record.log_files = log_files
        if failure_category:
            record.failure_category = failure_category
        record.cost_usd = self._estimate_cost(record)
        self.actions.append(record)
        self._current = None

    # ── Token parsing from CLI agent output ───────────────────────────

    @staticmethod
    def parse_tokens_from_stdout(stdout: str) -> TokenUsage:
        """Extract token usage from CLI agent stdout.

        Claude Code prints a usage summary like:
            Total tokens: 123456
            Input tokens: 80000
            Output tokens: 43456

        Codex prints similar info. We try multiple patterns.
        """
        usage = TokenUsage()
        if not stdout:
            return usage

        # Look at last 5000 chars where usage summaries typically appear
        tail = stdout[-5000:]

        # Pattern: "Input tokens: 12345" or "input_tokens: 12345"
        m = re.search(r'[Ii]nput.?tokens[:=]\s*([\d,]+)', tail)
        if m:
            usage.input_tokens = int(m.group(1).replace(',', ''))

        m = re.search(r'[Oo]utput.?tokens[:=]\s*([\d,]+)', tail)
        if m:
            usage.output_tokens = int(m.group(1).replace(',', ''))

        # Fallback: "Total tokens: 12345" without input/output breakdown
        if usage.total == 0:
            m = re.search(r'[Tt]otal.?tokens[:=]\s*([\d,]+)', tail)
            if m:
                total = int(m.group(1).replace(',', ''))
                # Rough split: assume 60% input, 40% output
                usage.input_tokens = int(total * 0.6)
                usage.output_tokens = total - usage.input_tokens

        # Pattern: "Cost: $1.23" — extract cost directly
        # (we compute our own cost estimate, but this is useful as a sanity check)

        return usage

    # ── Cost estimation ───────────────────────────────────────────────

    def _estimate_cost(self, record: ActionRecord) -> float:
        """Estimate cost from token usage.

        For subscription agents (Claude Code, Codex), this is the API-equivalent
        cost for comparison purposes — the actual cost is the subscription fee.
        For API-billed agents (Kimi, MiniMax), this is the real cost.
        """
        if not record.model or record.tokens.total == 0:
            return 0.0

        pricing = _PRICING.get(record.model)
        if not pricing:
            return 0.0

        input_cost = (record.tokens.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (record.tokens.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @staticmethod
    def is_subscription_agent(agent_type: str | None) -> bool:
        """Check if an agent runs on a flat-rate subscription."""
        return agent_type in _SUBSCRIPTION_AGENTS

    # ── Aggregation ───────────────────────────────────────────────────

    @property
    def total_elapsed(self) -> float:
        if self.run_end:
            return self.run_end - self.run_start
        return time.time() - self.run_start if self.run_start else 0.0

    @property
    def total_tokens(self) -> TokenUsage:
        total = TokenUsage()
        for a in self.actions:
            total += a.tokens
        return total

    @property
    def total_cost(self) -> float:
        return sum(a.cost_usd for a in self.actions)

    def stage_summary(self) -> dict[str, dict]:
        """Aggregate time, tokens, and cost per stage."""
        stages: dict[str, dict] = {}
        for a in self.actions:
            if a.stage not in stages:
                stages[a.stage] = {
                    "elapsed_seconds": 0.0,
                    "tokens": TokenUsage(),
                    "cost_usd": 0.0,
                    "actions": 0,
                    "successes": 0,
                    "failures": 0,
                }
            s = stages[a.stage]
            s["elapsed_seconds"] += a.elapsed_seconds
            s["tokens"] += a.tokens
            s["cost_usd"] += a.cost_usd
            s["actions"] += 1
            if a.outcome == "success":
                s["successes"] += 1
            elif a.outcome in ("failure", "timeout"):
                s["failures"] += 1
            if a.failure_category:
                cats = s.setdefault("failure_categories", {})
                cats[a.failure_category] = cats.get(a.failure_category, 0) + 1
        return stages

    # ── Display ───────────────────────────────────────────────────────

    def print_action_log(self):
        """Print a table of all actions."""
        table = Table(title="Action Log")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Stage")
        table.add_column("Action")
        table.add_column("Agent")
        table.add_column("Time", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Outcome")
        table.add_column("Details", max_width=40)

        for i, a in enumerate(self.actions):
            tokens_str = f"{a.tokens.total:,}" if a.tokens.total else "-"
            # Mark subscription agent costs as API-equivalent
            if a.cost_usd > 0:
                if self.is_subscription_agent(a.agent_type):
                    cost_str = f"~${a.cost_usd:.2f}"  # ~ means API-equivalent
                else:
                    cost_str = f"${a.cost_usd:.2f}"
            else:
                cost_str = "-"
            time_str = _format_duration(a.elapsed_seconds)
            outcome_style = "green" if a.outcome == "success" else "red" if a.outcome in ("failure", "timeout") else ""
            table.add_row(
                str(i + 1),
                a.stage,
                a.action,
                a.agent_type or "-",
                time_str,
                tokens_str,
                cost_str,
                f"[{outcome_style}]{a.outcome}[/]" if outcome_style else a.outcome,
                (a.details[:40] + "..." if len(a.details) > 40 else a.details) if a.details else "",
            )

        console.print(table)

    def print_stage_summary(self):
        """Print per-stage aggregation."""
        table = Table(title="Stage Summary")
        table.add_column("Stage")
        table.add_column("Actions", justify="right")
        table.add_column("OK/Fail", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Cost", justify="right")

        for stage, s in self.stage_summary().items():
            table.add_row(
                stage,
                str(s["actions"]),
                f"{s['successes']}/{s['failures']}",
                _format_duration(s["elapsed_seconds"]),
                f"{s['tokens'].total:,}" if s["tokens"].total else "-",
                f"${s['cost_usd']:.2f}" if s["cost_usd"] > 0 else "-",
            )

        # Totals row
        total_tokens = self.total_tokens
        table.add_row(
            "[bold]TOTAL[/]",
            f"[bold]{len(self.actions)}[/]",
            "",
            f"[bold]{_format_duration(self.total_elapsed)}[/]",
            f"[bold]{total_tokens.total:,}[/]" if total_tokens.total else "-",
            f"[bold]${self.total_cost:.2f}[/]" if self.total_cost > 0 else "-",
            style="bold",
        )

        console.print(table)

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        total_tokens = self.total_tokens
        stages = {}
        for stage, s in self.stage_summary().items():
            stage_dict = {
                "elapsed_seconds": round(s["elapsed_seconds"], 1),
                "tokens": s["tokens"].to_dict(),
                "cost_usd": round(s["cost_usd"], 4),
                "actions": s["actions"],
                "successes": s["successes"],
                "failures": s["failures"],
            }
            if s.get("failure_categories"):
                stage_dict["failure_categories"] = s["failure_categories"]
            stages[stage] = stage_dict
        return {
            "total_elapsed_seconds": round(self.total_elapsed, 1),
            "total_tokens": total_tokens.to_dict(),
            "total_cost_usd": round(self.total_cost, 4),
            "stages": stages,
            "actions": [a.to_dict() for a in self.actions],
        }

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "tracker.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m"
