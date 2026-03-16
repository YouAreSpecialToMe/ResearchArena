"""CLI entry point for researcharena."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from researcharena.utils.config import load_config, merge_configs

console = Console()


@click.group()
def main():
    """ResearchArena: benchmark CLI agents on autonomous research."""
    pass


@main.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Config file path")
@click.option("--seed", "-s", default=None, help="Override seed topic")
@click.option("--agent", default=None, help="Agent type (claude/codex/kimi/minimax/custom)")
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

    from researcharena.pipeline import Pipeline

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

    from researcharena.stages.review import review_paper, save_reviews

    paper_tex = workspace / "paper.tex"
    paper_pdf = workspace / "paper.pdf"

    if not paper_tex.exists():
        console.print(f"[red]No paper.tex found in {workspace}[/]")
        return

    # Use all agents as reviewers for standalone review
    result = review_paper(
        paper_latex=paper_tex.read_text(),
        paper_pdf_path=paper_pdf if paper_pdf.exists() else None,
        reviewer_agents=cfg["review"].get("agents", []),
        paperreview_config=cfg["review"].get("paperreview", {}),
        venue=cfg["paper"].get("template", "neurips"),
        accept_threshold=cfg["review"]["accept_threshold"],
        workspace=workspace,
        docker_image=cfg["agent"].get("docker_image", "researcharena/agent:latest"),
    )
    save_reviews(result, workspace)
    console.print(f"Score: {result.avg_score:.1f}/10, Decision: {result.decision}")


@main.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Config file path")
@click.option("--seeds-file", default="configs/seeds.yaml", help="Seeds YAML file")
@click.option("--field", "-f", default=None, help="Run only this field (e.g., cv, nlp)")
@click.option("--agent", default=None, help="Agent type override")
@click.option("--model", default=None, help="Agent model override")
@click.option("--max-ideas", default=None, type=int, help="Max ideas per seed")
def bench(config, seeds_file, field, agent, model, max_ideas):
    """Run the benchmark across seed fields.

    Each seed is a field name (cv, nlp, theory, systems). The agent
    decides what specific problem to research — that's part of what
    we're testing.
    """
    cfg = load_config(config)
    seeds_cfg = load_config(seeds_file)

    all_seeds = seeds_cfg.get("seeds", [])
    if field:
        if field not in all_seeds:
            console.print(f"[red]Unknown field: {field}. Available: {all_seeds}[/]")
            return
        seeds = [field]
    else:
        seeds = all_seeds

    overrides = {}
    if agent:
        overrides.setdefault("agent", {})["type"] = agent
    if model:
        overrides.setdefault("agent", {})["model"] = model
    if max_ideas:
        overrides.setdefault("pipeline", {})["max_ideas_per_seed"] = max_ideas

    from researcharena.pipeline import Pipeline
    from rich.table import Table

    results = []
    for seed in seeds:
        console.print(f"\n[bold magenta]{'='*60}[/]")
        console.print(f"[bold magenta]Seed field: {seed}[/]")
        console.print(f"[bold magenta]{'='*60}[/]")

        run_cfg = merge_configs(cfg, {**overrides, "seed_topic": seed})
        slug = seed.replace(" ", "_").replace("/", "_").lower()
        run_cfg["experiment"]["workspace"] = f"outputs/runs/{slug}"

        pipeline = Pipeline(run_cfg)
        result = pipeline.run()
        result["seed_topic"] = seed
        results.append(result)

        # Save per-seed summary
        summary_path = Path(run_cfg["experiment"]["workspace"]) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2))

    # Print leaderboard
    console.print(f"\n[bold]{'='*60}[/]")
    table = Table(title="Benchmark Results")
    table.add_column("Seed")
    table.add_column("Status")
    table.add_column("Best Score", justify="center")
    table.add_column("Best Title")
    table.add_column("Ideas", justify="center")

    for r in results:
        best = r.get("best_paper")
        score = f"{best['score']:.1f}" if best else "-"
        title = best.get("description", best.get("title", ""))[:40] if best else "-"
        table.add_row(
            r.get("seed_topic", ""),
            r.get("status", ""),
            score,
            title,
            str(r.get("ideas_tried", 0)),
        )

    console.print(table)

    combined_path = Path("outputs/runs/benchmark_results.json")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(results, indent=2))
    console.print(f"\nResults saved to {combined_path}")


@main.command()
@click.option("--seeds-file", default="configs/seeds.yaml", help="Seeds YAML file")
def list_seeds(seeds_file):
    """List all available seed fields."""
    cfg = load_config(seeds_file)
    seeds = cfg.get("seeds", [])
    console.print(f"[bold]Available seeds ({len(seeds)}):[/]")
    for s in seeds:
        console.print(f"  - {s}")


if __name__ == "__main__":
    main()
