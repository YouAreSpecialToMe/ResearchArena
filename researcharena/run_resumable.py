#!/usr/bin/env python3
"""Run the research pipeline with checkpoint/resume support.

Drop-in replacement for `researcharena run` that uses ResumablePipeline
instead of Pipeline. All CLI arguments are identical.

Usage:
    python -m researcharena.run_resumable --config configs/codex_cpu.yaml \
        --seed "compiler optimization" --platform cpu \
        --workspace outputs/codex_t3_compiler_optimization

Or from a SLURM script:
    python -m researcharena.run_resumable [same args as researcharena run]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from researcharena.utils.config import load_config, merge_configs

console = Console()


def _resolve_platform_config(cfg: dict, platform: str) -> dict:
    """Merge platform-specific resources into config."""
    platforms = cfg.get("platforms", {})
    plat_cfg = platforms.get(platform, {})
    overrides = {"seed_platform": platform}
    if plat_cfg:
        if "resources" in plat_cfg:
            overrides["resources"] = plat_cfg["resources"]
        if "docker_image" in plat_cfg:
            overrides.setdefault("agent", {})["docker_image"] = plat_cfg["docker_image"]
    return merge_configs(cfg, overrides)


@click.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Config file path")
@click.option("--seed", "-s", default=None, help="Override seed topic")
@click.option("--agent", default=None, help="Agent type")
@click.option("--model", default=None, help="Agent model override")
@click.option("--max-ideas", default=None, type=int, help="Max ideas per seed")
@click.option("--platform", default=None, type=click.Choice(["gpu", "cpu"]))
@click.option("--domain", default=None, type=click.Choice(["ml", "systems", "databases", "pl", "theory", "security"]))
@click.option("--workspace", "-w", default=None, help="Override output workspace directory")
def main(config, seed, agent, model, max_ideas, platform, domain, workspace):
    """Run the research pipeline with automatic checkpoint/resume."""
    cfg = load_config(config)

    overrides = {}
    if seed:
        overrides["seed_topic"] = seed
    if agent:
        overrides.setdefault("agent", {})["type"] = agent
    if model:
        overrides.setdefault("agent", {})["model"] = model
    if max_ideas:
        overrides.setdefault("pipeline", {})["max_ideas_per_seed"] = max_ideas
    if domain:
        overrides["seed_domain"] = domain
    if workspace:
        overrides.setdefault("experiment", {})["workspace"] = workspace

    if overrides:
        cfg = merge_configs(cfg, overrides)

    resolved_platform = platform or cfg.get("seed_platform", "gpu")
    cfg = _resolve_platform_config(cfg, resolved_platform)

    from researcharena.pipeline_resumable import ResumablePipeline

    pipeline = ResumablePipeline(cfg)
    result = pipeline.run()

    summary_path = Path(cfg["experiment"]["workspace"]) / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2))
    console.print(f"\n[bold]Summary saved to {summary_path}[/]")


if __name__ == "__main__":
    main()
