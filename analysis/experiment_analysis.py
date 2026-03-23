"""
ResearchArena Experiment Analysis
=================================
Analyze all experiment results across agents, seeds, platforms, and trials.

Usage:
    python analysis/experiment_analysis.py
    # Or run in Jupyter: copy cells into a notebook
"""

import json
import glob
import os
import math
from pathlib import Path
from collections import defaultdict
import sys

# ── Data Collection ──────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent


def find_all_results():
    """Collect all experiment results into a structured dict."""
    results = []

    # results/ directory (organized trials)
    for agent_dir in sorted((BASE / "results").iterdir()):
        if not agent_dir.is_dir():
            continue
        agent = agent_dir.name  # "kimi", "codex", "claude"

        for seed_dir in sorted(agent_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            seed_name = seed_dir.name  # e.g., "ai_for_biology_gpu"
            platform = "gpu" if seed_name.endswith("_gpu") else "cpu"
            seed = seed_name.replace("_gpu", "").replace("_cpu", "").replace("_", " ")

            for trial_dir in sorted(seed_dir.iterdir()):
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial"):
                    continue
                trial_num = int(trial_dir.name.replace("trial", ""))

                result = extract_trial_data(agent, seed, platform, trial_num, trial_dir)
                if result:
                    results.append(result)

    # outputs/ directory (run_N structure for GPU)
    for agent in ["kimi", "codex", "claude"]:
        for run_dir in sorted((BASE / "outputs" / agent).glob("run_*")):
            run_num = int(run_dir.name.split("_")[1])
            for seed_dir in sorted(run_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                seed = seed_dir.name.replace("_", " ")
                result = extract_output_data(agent, seed, "gpu", run_num, seed_dir)
                if result:
                    # Avoid duplicates with results/ dir
                    dup = any(r["agent"] == agent and r["seed"] == seed
                             and r["trial"] == run_num and r["platform"] == "gpu"
                             for r in results)
                    if not dup:
                        results.append(result)

    # outputs/ directory (timestamp structure for GPU)
    for agent in ["kimi", "codex", "claude"]:
        agent_dir = BASE / "outputs" / agent
        if not agent_dir.exists():
            continue
        for ws_dir in sorted(agent_dir.iterdir()):
            if not ws_dir.is_dir() or ws_dir.name.startswith("run_") or ws_dir.name == "logs":
                continue
            # Parse seed from timestamp dir name
            parts = ws_dir.name.rsplit("_", 2)
            if len(parts) >= 3 and parts[-2].isdigit():
                seed = "_".join(parts[:-2]).replace("_", " ")
            else:
                continue
            result = extract_output_data(agent, seed, "gpu", 1, ws_dir)
            if result:
                dup = any(r["agent"] == agent and r["seed"] == seed
                         and r["trial"] == 1 and r["platform"] == "gpu"
                         for r in results)
                if not dup:
                    results.append(result)

    return results


def extract_trial_data(agent, seed, platform, trial_num, trial_dir):
    """Extract data from results/ trial directory."""
    data = {
        "agent": agent, "seed": seed, "platform": platform,
        "trial": trial_num, "source": "results",
    }

    # Reviews
    review_files = sorted(trial_dir.glob("code/idea_*/reviews.json"))
    if review_files:
        try:
            reviews = json.loads(review_files[-1].read_text())
            data["avg_score"] = reviews.get("avg_score", 0)
            data["decision"] = reviews.get("decision", "unknown")
            data["per_reviewer"] = {}
            for r in reviews.get("reviews", []):
                name = r.get("source", r.get("reviewer", "unknown"))
                data["per_reviewer"][name] = {
                    "overall": r.get("overall_score"),
                    "scores": r.get("scores", {}),
                }
        except:
            pass

    # Tracker
    tracker_file = trial_dir / "tracker" / "tracker.json"
    if tracker_file.exists():
        try:
            tracker = json.loads(tracker_file.read_text())
            data["tracker"] = extract_tracker_stats(tracker)
        except:
            pass

    # Summary
    summary_file = trial_dir / "tracker" / "summary.json"
    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text())
            data["status"] = summary.get("status", "unknown")
            data["total_steps"] = summary.get("total_steps", 0)
            data["ideas_tried"] = summary.get("ideas_tried", 0)
            data["wall_time"] = summary.get("wall_time_seconds", 0)
        except:
            pass

    if "avg_score" in data:
        return data
    return None


def extract_output_data(agent, seed, platform, trial_num, workspace_dir):
    """Extract data from outputs/ workspace directory."""
    data = {
        "agent": agent, "seed": seed, "platform": platform,
        "trial": trial_num, "source": "outputs",
    }

    # Find latest idea with reviews
    idea_dirs = sorted(workspace_dir.glob("idea_*/reviews.json"))
    if not idea_dirs:
        return None

    try:
        reviews = json.loads(idea_dirs[-1].read_text())
        data["avg_score"] = reviews.get("avg_score", 0)
        data["decision"] = reviews.get("decision", "unknown")
        data["per_reviewer"] = {}
        for r in reviews.get("reviews", []):
            name = r.get("source", r.get("reviewer", "unknown"))
            data["per_reviewer"][name] = {
                "overall": r.get("overall_score"),
                "scores": r.get("scores", {}),
            }
    except:
        return None

    # Tracker
    tracker_file = workspace_dir / "tracker.json"
    if tracker_file.exists():
        try:
            tracker = json.loads(tracker_file.read_text())
            data["tracker"] = extract_tracker_stats(tracker)
        except:
            pass

    # Summary
    summary_file = workspace_dir / "summary.json"
    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text())
            data["status"] = summary.get("status", "unknown")
            data["total_steps"] = summary.get("total_steps", 0)
            data["ideas_tried"] = summary.get("ideas_tried", 0)
            data["wall_time"] = summary.get("wall_time_seconds", 0)
        except:
            pass

    return data


def extract_tracker_stats(tracker):
    """Extract timing and token stats from tracker data."""
    actions = tracker.get("actions", [])
    stats = {
        "total_actions": len(actions),
        "stages": defaultdict(lambda: {"count": 0, "total_time": 0, "outcomes": defaultdict(int)}),
    }

    for a in actions:
        stage = a.get("stage", "unknown")
        elapsed = a.get("elapsed_seconds", 0)
        outcome = a.get("outcome", "unknown")
        stats["stages"][stage]["count"] += 1
        stats["stages"][stage]["total_time"] += elapsed
        stats["stages"][stage]["outcomes"][outcome] += 1

    # Convert defaultdicts for JSON serialization
    stats["stages"] = {
        k: {"count": v["count"], "total_time": v["total_time"],
            "outcomes": dict(v["outcomes"])}
        for k, v in stats["stages"].items()
    }
    return stats


# ── Analysis Functions ───────────────────────────────────────────────


def mean_se(values):
    """Compute mean and standard error."""
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, None
    std = (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5
    se = std / math.sqrt(n)
    return mean, se


def print_agent_summary(results):
    """Print summary table per agent."""
    agents = sorted(set(r["agent"] for r in results))

    for agent in agents:
        agent_results = [r for r in results if r["agent"] == agent]
        platforms = sorted(set(r["platform"] for r in agent_results))

        for platform in platforms:
            plat_results = [r for r in agent_results if r["platform"] == platform]
            seeds = sorted(set(r["seed"] for r in plat_results))

            print(f"\n{'='*70}")
            print(f"  {agent.upper()} — {platform.upper()} ({len(plat_results)} trials across {len(seeds)} seeds)")
            print(f"{'='*70}")
            print(f"{'Seed':<40} | {'N':>3} | {'Mean':>6} | {'SE':>6} | {'Scores'}")
            print("-" * 40 + "-|" + "-" * 5 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 20)

            all_means = []
            for seed in seeds:
                scores = [r["avg_score"] for r in plat_results if r["seed"] == seed]
                m, se = mean_se(scores)
                all_means.append(m)
                score_str = ", ".join(f"{s:.1f}" for s in scores)
                se_str = f"{se:.2f}" if se is not None else "n/a"
                print(f"{seed:<40} | {len(scores):>3} | {m:6.2f} | {se_str:>6} | {score_str}")

            m, se = mean_se(all_means)
            se_str = f"{se:.2f}" if se is not None else "n/a"
            print(f"{'AVERAGE':<40} | {len(all_means):>3} | {m:6.2f} | {se_str:>6} |")


def print_reviewer_analysis(results):
    """Analyze per-reviewer scoring patterns."""
    reviewer_scores = defaultdict(list)

    for r in results:
        for name, data in r.get("per_reviewer", {}).items():
            if data.get("overall") is not None:
                reviewer_scores[name].append(data["overall"])

    print(f"\n{'='*70}")
    print("  REVIEWER SCORING PATTERNS")
    print(f"{'='*70}")
    print(f"{'Reviewer':<30} | {'N':>5} | {'Mean':>6} | {'SE':>6} | {'Min':>4} | {'Max':>4}")
    print("-" * 30 + "-|" + "-" * 7 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 6 + "|" + "-" * 6)

    for name in sorted(reviewer_scores.keys()):
        scores = reviewer_scores[name]
        m, se = mean_se(scores)
        se_str = f"{se:.2f}" if se is not None else "n/a"
        print(f"{name:<30} | {len(scores):>5} | {m:6.2f} | {se_str:>6} | {min(scores):>4} | {max(scores):>4}")


def print_timing_analysis(results):
    """Analyze time spent per stage."""
    stage_times = defaultdict(list)

    for r in results:
        tracker = r.get("tracker", {})
        for stage, stats in tracker.get("stages", {}).items():
            if stats["total_time"] > 0:
                stage_times[stage].append(stats["total_time"])

    print(f"\n{'='*70}")
    print("  TIME PER STAGE (seconds)")
    print(f"{'='*70}")
    print(f"{'Stage':<30} | {'N':>5} | {'Mean':>8} | {'SE':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 30 + "-|" + "-" * 7 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10)

    for stage in sorted(stage_times.keys()):
        times = stage_times[stage]
        m, se = mean_se(times)
        se_str = f"{se:.0f}" if se is not None else "n/a"
        print(f"{stage:<30} | {len(times):>5} | {m:8.0f} | {se_str:>8} | {min(times):8.0f} | {max(times):8.0f}")


def print_self_review_impact(results):
    """Analyze self-review outcomes and their impact."""
    sr_stats = defaultdict(lambda: defaultdict(int))

    for r in results:
        tracker = r.get("tracker", {})
        for stage, stats in tracker.get("stages", {}).items():
            if "self_review" in stage:
                for outcome, count in stats.get("outcomes", {}).items():
                    sr_stats[stage][outcome] += count
                    sr_stats[stage]["total"] += count

    print(f"\n{'='*70}")
    print("  SELF-REVIEW OUTCOMES")
    print(f"{'='*70}")
    print(f"{'Gate':<30} | {'Total':>6} | {'Pass':>6} | {'Revise':>6} | {'Skip':>6} | {'Abandon':>6} | {'Pass%':>6}")
    print("-" * 30 + "-|" + ("-" * 8 + "|") * 6)

    for gate in sorted(sr_stats.keys()):
        s = dict(sr_stats[gate])
        total = s.get("total", 0)
        passed = s.get("success", 0)
        revised = s.get("revision", 0)
        skipped = s.get("skipped", 0)
        abandoned = s.get("abandoned", 0)
        pass_rate = f"{100 * passed / total:.0f}%" if total > 0 else "n/a"
        print(f"{gate:<30} | {total:>6} | {passed:>6} | {revised:>6} | {skipped:>6} | {abandoned:>6} | {pass_rate:>6}")


def print_wall_time_analysis(results):
    """Analyze total wall time per run."""
    agent_times = defaultdict(list)

    for r in results:
        wt = r.get("wall_time", 0)
        if wt > 0:
            agent_times[f"{r['agent']}_{r['platform']}"].append(wt / 3600)

    print(f"\n{'='*70}")
    print("  WALL TIME PER RUN (hours)")
    print(f"{'='*70}")
    print(f"{'Agent/Platform':<30} | {'N':>5} | {'Mean':>8} | {'SE':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 30 + "-|" + "-" * 7 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10)

    for key in sorted(agent_times.keys()):
        times = agent_times[key]
        m, se = mean_se(times)
        se_str = f"{se:.1f}" if se is not None else "n/a"
        print(f"{key:<30} | {len(times):>5} | {m:8.1f} | {se_str:>8} | {min(times):8.1f} | {max(times):8.1f}")


def print_dimension_analysis(results):
    """Analyze per-dimension scores across all reviews."""
    dim_scores = defaultdict(lambda: defaultdict(list))

    for r in results:
        agent = r["agent"]
        for reviewer_name, reviewer_data in r.get("per_reviewer", {}).items():
            for dim, score in reviewer_data.get("scores", {}).items():
                if isinstance(score, (int, float)):
                    dim_scores[dim][agent].append(score)

    print(f"\n{'='*70}")
    print("  PER-DIMENSION SCORES (averaged across all reviews)")
    print(f"{'='*70}")

    agents = sorted(set(r["agent"] for r in results))
    header = f"{'Dimension':<25}"
    for agent in agents:
        header += f" | {agent:>8}"
    print(header)
    print("-" * 25 + ("-|" + "-" * 10) * len(agents))

    for dim in sorted(dim_scores.keys()):
        row = f"{dim:<25}"
        for agent in agents:
            scores = dim_scores[dim].get(agent, [])
            if scores:
                m, _ = mean_se(scores)
                row += f" | {m:8.1f}"
            else:
                row += f" | {'n/a':>8}"
        print(row)


# ── Main ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("Collecting results...")
    results = find_all_results()
    print(f"Found {len(results)} trial results\n")

    print_agent_summary(results)
    print_reviewer_analysis(results)
    print_dimension_analysis(results)
    print_self_review_impact(results)
    print_timing_analysis(results)
    print_wall_time_analysis(results)
