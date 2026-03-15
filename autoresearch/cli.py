"""CLI entry point for autoresearch."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from autoresearch.utils.config import load_config, merge_configs

console = Console()


@click.group()
def main():
    """AutoResearch: benchmark CLI agents on autonomous research."""
    pass


@main.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Config file path")
@click.option("--seed", "-s", default=None, help="Override seed topic")
@click.option("--agent", default=None, help="Agent type (claude/codex/aider/custom)")
@click.option("--model", default=None, help="Agent model override")
@click.option("--max-ideas", default=None, type=int, help="Max ideas per seed")
def run(config, seed, agent, model, max_ideas):
    """Run the full research pipeline with a CLI agent."""
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

    if overrides:
        cfg = merge_configs(cfg, overrides)

    from autoresearch.pipeline import Pipeline

    pipeline = Pipeline(cfg)
    result = pipeline.run()

    # Save final summary
    summary_path = Path(cfg["experiment"]["workspace"]) / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2))

    console.print(f"\n[bold]Summary saved to {summary_path}[/]")
    console.print(json.dumps(result, indent=2))


@main.command()
@click.option("--config", "-c", default="configs/default.yaml")
@click.argument("workspace", type=click.Path(exists=True))
def review_only(config, workspace):
    """Run review on an existing paper (skip research stages)."""
    cfg = load_config(config)
    workspace = Path(workspace)

    from autoresearch.stages.review import review_paper, save_reviews

    paper_tex = workspace / "paper.tex"
    paper_pdf = workspace / "paper.pdf"

    if not paper_tex.exists():
        console.print(f"[red]No paper.tex found in {workspace}[/]")
        return

    result = review_paper(
        paper_latex=paper_tex.read_text(),
        paper_pdf_path=paper_pdf if paper_pdf.exists() else None,
        agent_configs=cfg["review"].get("agents", []),
        paperreview_config=cfg["review"].get("paperreview", {}),
        venue=cfg["paper"].get("template", "neurips"),
        accept_threshold=cfg["review"]["accept_threshold"],
        workspace=workspace,
    )
    save_reviews(result, workspace)
    console.print(f"Score: {result.avg_score:.1f}/10, Decision: {result.decision}")


if __name__ == "__main__":
    main()
