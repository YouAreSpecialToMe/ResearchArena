"""CLI entry point for researcharena."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from researcharena.utils.config import load_config, merge_configs

console = Console()


def _normalize_seeds(raw_seeds: list) -> list[dict]:
    """Normalize seeds to list of dicts.

    Supports both old format (list of strings) and new format (list of dicts
    with name/conferences/platform fields).
    """
    normalized = []
    for s in raw_seeds:
        if isinstance(s, str):
            normalized.append({"name": s, "conferences": [], "platform": "gpu", "domain": "ml"})
        elif isinstance(s, dict):
            normalized.append({
                "name": s.get("name", ""),
                "conferences": s.get("conferences", []),
                "platform": s.get("platform", "gpu"),
                "domain": s.get("domain", "ml"),
            })
    return normalized


def _resolve_platform_config(cfg: dict, platform: str) -> dict:
    """Merge platform-specific resources and docker_image into the config.

    Always returns a new dict — never mutates the input.
    """
    platforms = cfg.get("platforms", {})
    plat_cfg = platforms.get(platform, {})

    overrides = {"seed_platform": platform}

    if plat_cfg:
        if "resources" in plat_cfg:
            overrides["resources"] = plat_cfg["resources"]
        if "docker_image" in plat_cfg:
            overrides.setdefault("agent", {})["docker_image"] = plat_cfg["docker_image"]

    return merge_configs(cfg, overrides)


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
@click.option("--platform", default=None, type=click.Choice(["gpu", "cpu"]), help="Platform (gpu/cpu)")
@click.option("--domain", default=None, type=click.Choice(["ml", "systems", "databases", "pl", "theory", "security"]), help="Domain (selects guideline templates)")
@click.option("--workspace", "-w", default=None, help="Override output workspace directory")
@click.option("--resume", "-r", default=None, type=click.Path(exists=True), help="Resume from existing idea workspace (e.g., outputs/kimi/run_3/computer_vision/idea_01)")
def run(config, seed, agent, model, max_ideas, platform, domain, workspace, resume):
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
    if domain:
        overrides["seed_domain"] = domain
    if workspace:
        overrides.setdefault("experiment", {})["workspace"] = workspace

    if overrides:
        cfg = merge_configs(cfg, overrides)

    # Resolve platform (CLI flag > config > default "gpu")
    resolved_platform = platform or cfg.get("seed_platform", "gpu")
    cfg = _resolve_platform_config(cfg, resolved_platform)

    from researcharena.pipeline import Pipeline

    pipeline = Pipeline(cfg)

    if resume:
        result = pipeline.resume(resume)
    else:
        result = pipeline.run()

    # Save final summary
    summary_path = Path(cfg["experiment"]["workspace"]) / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2))

    console.print(f"\n[bold]Summary saved to {summary_path}[/]")
    console.print(json.dumps(result, indent=2))


@main.command()
@click.option("--config", "-c", default="configs/default.yaml")
@click.option("--domain", default=None, type=click.Choice(["ml", "systems", "databases", "pl", "theory", "security"]), help="Domain (selects reviewer guidelines)")
@click.argument("workspace", type=click.Path(exists=True))
def review_only(config, domain, workspace):
    """Run review on an existing paper (skip research stages)."""
    cfg = load_config(config)
    workspace = Path(workspace)

    from researcharena.stages.review import review_paper, save_reviews

    paper_tex = workspace / "paper.tex"
    paper_pdf = workspace / "paper.pdf"

    if not paper_tex.exists():
        console.print(f"[red]No paper.tex found in {workspace}[/]")
        return

    resolved_domain = domain or cfg.get("seed_domain", "ml")

    # Use all agents as reviewers for standalone review
    result = review_paper(
        paper_latex=paper_tex.read_text(),
        paper_pdf_path=paper_pdf if paper_pdf.exists() else None,
        reviewer_agents=cfg["review"].get("agents", []),
        paperreview_config=cfg["review"].get("paperreview", {}),
        venue=cfg.get("seed_conferences", [None])[0] or cfg["paper"].get("template") or {"ml": "neurips", "systems": "osdi", "databases": "sigmod", "pl": "pldi", "theory": "stoc", "security": "ccs"}.get(resolved_domain, "neurips"),
        accept_threshold=cfg["review"]["accept_threshold"],
        workspace=workspace,
        docker_image=cfg["agent"].get("docker_image", "researcharena/agent:latest"),
        runtime=cfg["agent"].get("runtime", "docker"),
        domain=resolved_domain,
    )
    save_reviews(result, workspace)
    console.print(f"Score: {result.avg_score:.1f}/10, Decision: {result.decision}")


@main.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Config file path")
@click.option("--seeds-file", default="configs/seeds.yaml", help="Seeds YAML file")
@click.option("--field", "-f", default=None, help="Run only this field (e.g., 'computer vision')")
@click.option("--agent", default=None, help="Agent type override")
@click.option("--model", default=None, help="Agent model override")
@click.option("--max-ideas", default=None, type=int, help="Max ideas per seed")
@click.option("--conference", default=None, help="Filter seeds by conference (e.g., sigmod, iclr)")
@click.option("--platform", default=None, type=click.Choice(["gpu", "cpu"]), help="Filter seeds by platform")
def bench(config, seeds_file, field, agent, model, max_ideas, conference, platform):
    """Run the benchmark across seed fields.

    Each seed is a field name with a platform (gpu/cpu) and conference tags.
    The agent decides what specific problem to research — that's part of what
    we're testing.

    Examples:
      researcharena bench --agent claude                      # all seeds
      researcharena bench --agent claude --platform gpu       # GPU seeds only
      researcharena bench --agent claude --conference sigmod  # SIGMOD seeds
      researcharena bench --agent claude --field "query optimization"
    """
    cfg = load_config(config)
    seeds_cfg = load_config(seeds_file)

    all_seeds = _normalize_seeds(seeds_cfg.get("seeds", []))

    # Apply filters
    seeds = all_seeds
    if field:
        seeds = [s for s in seeds if s["name"] == field]
        if not seeds:
            all_names = [s["name"] for s in all_seeds]
            console.print(f"[red]Unknown field: {field}.[/]")
            # Show close matches
            close = [n for n in all_names if field.lower() in n.lower()]
            if close:
                console.print(f"[yellow]Did you mean: {close}[/]")
            return

    if conference:
        conf_lower = conference.lower()
        seeds = [s for s in seeds if conf_lower in [c.lower() for c in s["conferences"]]]
        if not seeds:
            console.print(f"[red]No seeds found for conference: {conference}[/]")
            return

    if platform:
        seeds = [s for s in seeds if s["platform"] == platform]
        if not seeds:
            console.print(f"[red]No seeds found for platform: {platform}[/]")
            return

    console.print(f"[bold]Running {len(seeds)} seed(s)[/]")
    if conference:
        console.print(f"  Conference filter: {conference}")
    if platform:
        console.print(f"  Platform filter: {platform}")

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
        seed_name = seed["name"]
        seed_platform = seed["platform"]

        console.print(f"\n[bold magenta]{'='*60}[/]")
        console.print(
            f"[bold magenta]Seed: {seed_name}  |  "
            f"Platform: {seed_platform}  |  "
            f"Conferences: {', '.join(seed['conferences'])}[/]"
        )
        console.print(f"[bold magenta]{'='*60}[/]")

        run_cfg = merge_configs(cfg, {
            **overrides,
            "seed_topic": seed_name,
            "seed_domain": seed.get("domain", "ml"),
            "seed_conferences": seed.get("conferences", []),
        })
        run_cfg = _resolve_platform_config(run_cfg, seed_platform)

        slug = seed_name.replace(" ", "_").replace("/", "_").lower()
        run_cfg["experiment"]["workspace"] = f"outputs/runs/{slug}"

        pipeline = Pipeline(run_cfg)
        result = pipeline.run()
        result["seed_topic"] = seed_name
        result["platform"] = seed_platform
        result["conferences"] = seed["conferences"]
        results.append(result)

        # Save per-seed summary
        summary_path = Path(run_cfg["experiment"]["workspace"]) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2))

    # Print leaderboard
    console.print(f"\n[bold]{'='*60}[/]")
    table = Table(title="Benchmark Results")
    table.add_column("Seed")
    table.add_column("Platform")
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
            r.get("platform", ""),
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
@click.option("--conference", default=None, help="Filter by conference")
@click.option("--platform", default=None, type=click.Choice(["gpu", "cpu"]), help="Filter by platform")
def list_seeds(seeds_file, conference, platform):
    """List all available seed fields."""
    cfg = load_config(seeds_file)
    all_seeds = _normalize_seeds(cfg.get("seeds", []))

    seeds = all_seeds
    if conference:
        conf_lower = conference.lower()
        seeds = [s for s in seeds if conf_lower in [c.lower() for c in s["conferences"]]]
    if platform:
        seeds = [s for s in seeds if s["platform"] == platform]

    gpu_seeds = [s for s in seeds if s["platform"] == "gpu"]
    cpu_seeds = [s for s in seeds if s["platform"] == "cpu"]

    console.print(f"[bold]Available seeds ({len(seeds)} total, {len(gpu_seeds)} GPU, {len(cpu_seeds)} CPU):[/]")

    if gpu_seeds:
        console.print(f"\n[bold green]GPU platform ({len(gpu_seeds)}):[/]")
        for s in gpu_seeds:
            confs = ", ".join(s["conferences"]) if s["conferences"] else "—"
            console.print(f"  - {s['name']}  [dim]({confs})[/]")

    if cpu_seeds:
        console.print(f"\n[bold blue]CPU platform ({len(cpu_seeds)}):[/]")
        for s in cpu_seeds:
            confs = ", ".join(s["conferences"]) if s["conferences"] else "—"
            console.print(f"  - {s['name']}  [dim]({confs})[/]")


if __name__ == "__main__":
    main()
