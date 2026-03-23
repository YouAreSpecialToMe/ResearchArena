"""Resumable pipeline wrapper.

Extends the standard Pipeline with automatic checkpointing after each step,
and the ability to resume from a checkpoint after preemption or crash.

Usage:
    # In place of Pipeline(cfg).run():
    from researcharena.pipeline_resumable import ResumablePipeline
    pipeline = ResumablePipeline(cfg)
    result = pipeline.run()  # auto-resumes if checkpoint exists

This file does NOT modify pipeline.py — it imports and wraps it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console

from researcharena.pipeline import Pipeline, Stage, PipelineState
from researcharena.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    restore_state,
    CHECKPOINT_FILENAME,
)
from researcharena.utils.tracker import RunTracker, ActionRecord, TokenUsage

console = Console()


class ResumablePipeline(Pipeline):
    """Pipeline with automatic checkpoint/resume support.

    On each step, the pipeline state is saved to checkpoint.json in the
    workspace. If a checkpoint exists when run() is called, the pipeline
    resumes from that point instead of starting fresh.
    """

    def run(self) -> dict:
        seed_topic = self.config["seed_topic"]
        accept_threshold = self.config["review"]["accept_threshold"]

        # ── Try to resume from checkpoint ──
        checkpoint = load_checkpoint(self.base_dir)
        if checkpoint is not None:
            console.print("[bold yellow]Checkpoint found — resuming pipeline.[/]")
            restore_state(self.state, checkpoint)

            # Restore tracker actions from checkpoint
            self._restore_tracker(checkpoint)

            console.print(
                f"  Resuming from step {self.state.global_step}, "
                f"stage: {self.state.stage.value}, "
                f"idea: {self.state.idea_attempts}"
            )
        else:
            console.print("[bold green]No checkpoint — starting fresh.[/]")
            self.tracker.start_run()

        console.print(
            f"  Agent: {self.agent_type} "
            f"({self.agent_config.get('model', 'default')})\n"
            f"  Seed: {seed_topic}\n"
            f"  Platform: {self.platform}"
        )

        start_time = time.time()

        while self.state.stage not in (Stage.ACCEPTED, Stage.FAILED):
            if self.state.global_step >= self.state.max_global_steps:
                console.print("[red]Hit global step limit. Stopping.[/]")
                self.state.stage = Stage.FAILED
                break

            self.state.global_step += 1
            stage = self.state.stage
            console.print(
                f"\n[bold cyan]Step {self.state.global_step}: {stage.value}[/]"
            )

            # Run the stage
            if stage == Stage.IDEATION:
                self._run_ideation(seed_topic)
            elif stage == Stage.SELF_REVIEW_IDEA:
                self._run_self_review_idea()
            elif stage == Stage.EXPERIMENTS:
                self._run_experiments()
            elif stage == Stage.SELF_REVIEW_EXPERIMENT:
                self._run_self_review_experiment()
            elif stage == Stage.PAPER:
                self._run_paper()
            elif stage == Stage.SELF_REVIEW_PAPER:
                self._run_self_review_paper()
            elif stage == Stage.REVIEW:
                self._run_review(accept_threshold)

            # ── Checkpoint after every step ──
            save_checkpoint(self.state, self.base_dir, self.tracker)

        self.tracker.end_run()
        elapsed = time.time() - start_time
        self._print_summary(elapsed)

        # Save final tracker data
        self.tracker.save(self.base_dir)

        # Remove checkpoint on successful completion
        checkpoint_path = self.base_dir / CHECKPOINT_FILENAME
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            console.print("[dim]Checkpoint removed (run complete).[/]")

        return self._build_summary(elapsed)

    def _restore_tracker(self, checkpoint: dict):
        """Restore tracker actions from checkpoint data."""
        saved_actions = checkpoint.get("_tracker_actions", [])
        if not saved_actions:
            self.tracker.start_run()
            return

        # Reconstruct ActionRecord objects from saved dicts
        self.tracker.run_start = time.time()  # approximate
        for a in saved_actions:
            if isinstance(a, dict):
                tokens = a.get("tokens", {})
                record = ActionRecord(
                    stage=a.get("stage", ""),
                    action=a.get("action", ""),
                    agent_type=a.get("agent_type"),
                    model=a.get("model"),
                    attempt=a.get("attempt"),
                    start_time=0,
                    end_time=0,
                    elapsed_seconds=a.get("elapsed_seconds", 0),
                    tokens=TokenUsage(
                        input_tokens=tokens.get("input_tokens", 0),
                        output_tokens=tokens.get("output_tokens", 0),
                    ),
                    outcome=a.get("outcome", ""),
                    details=a.get("details", ""),
                    cost_usd=a.get("cost_usd", 0),
                    log_files=a.get("log_files"),
                    failure_category=a.get("failure_category"),
                )
                self.tracker.actions.append(record)

        console.print(
            f"  Restored {len(self.tracker.actions)} tracked actions "
            f"from checkpoint."
        )
