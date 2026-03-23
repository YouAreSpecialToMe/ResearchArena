#!/usr/bin/env python3
"""Analyze all 45 CPU experiment results (3 agents × 3 trials × 5 seeds).

Run from the repo root or analysis/ directory:
    python3 analysis/cpu_analysis.py

Generates figures to analysis/figures/ and prints summary tables.
"""

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["figure.figsize"] = (14, 7)
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["savefig.dpi"] = 150
matplotlib.rcParams["savefig.bbox"] = "tight"

# ── Resolve paths ──
BASE = Path(__file__).parent.parent if (Path(__file__).parent.parent / "results").exists() else Path(".")
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

AGENT_COLORS = {"claude": "#6B4CE6", "kimi": "#FF6B35", "codex": "#2196F3"}
AGENT_LABELS = {"claude": "Claude", "kimi": "Kimi", "codex": "Codex"}

# ── 1. Data Collection ──────────────────────────────────────────────────

def collect_data():
    rows = []
    for agent in ["claude", "kimi", "codex"]:
        agent_dir = BASE / "results" / agent
        if not agent_dir.exists():
            continue
        for seed_dir in sorted(agent_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.endswith("_cpu"):
                continue
            seed_raw = seed_dir.name.replace("_cpu", "")
            seed = seed_raw.replace("_", " ").title()

            for trial_dir in sorted(seed_dir.iterdir()):
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial"):
                    continue
                trial_num = int(trial_dir.name.replace("trial", ""))

                review_files = sorted(trial_dir.glob("code/idea_*/reviews.json"))
                if not review_files:
                    continue
                try:
                    reviews = json.loads(review_files[-1].read_text())
                except Exception:
                    continue

                row = {
                    "agent": agent,
                    "seed": seed,
                    "trial": trial_num,
                    "avg_score": reviews.get("avg_score", 0),
                }

                for rev in reviews.get("reviews", []):
                    name = rev.get("source", "?").replace("agent:", "")
                    row[f"reviewer_{name}"] = rev.get("overall_score")
                    for dim, val in rev.get("scores", {}).items():
                        if isinstance(val, (int, float)):
                            row[f"dim_{dim}_{name}"] = val

                tracker_file = trial_dir / "tracker" / "tracker.json"
                if tracker_file.exists():
                    try:
                        tracker = json.loads(tracker_file.read_text())
                        actions = tracker.get("actions", [])
                        for a in actions:
                            stage = a.get("stage", "")
                            if "self_review" in stage:
                                key = f"sr_{stage}_count"
                                row[key] = row.get(key, 0) + 1
                                if a.get("outcome") == "success":
                                    row[f"sr_{stage}_pass"] = row.get(f"sr_{stage}_pass", 0) + 1
                            elapsed = a.get("elapsed_seconds", 0)
                            if elapsed > 0:
                                key = f"time_{stage}"
                                row[key] = row.get(key, 0) + elapsed
                    except Exception:
                        pass

                summary_file = trial_dir / "tracker" / "summary.json"
                if summary_file.exists():
                    try:
                        s = json.loads(summary_file.read_text())
                        row["wall_time_h"] = s.get("wall_time_seconds", 0) / 3600
                        row["total_steps"] = s.get("total_steps", 0)
                        row["ideas_tried"] = s.get("ideas_tried", 0)
                    except Exception:
                        pass

                rows.append(row)

    return pd.DataFrame(rows)


# ── 2. Peer Review Scores by Agent ─────────────────────────────────────

def plot_scores_by_agent(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    agents = ["claude", "kimi", "codex"]
    positions = []
    for i, agent in enumerate(agents):
        data = df[df.agent == agent]["avg_score"].dropna()
        bp = ax.boxplot(
            data, positions=[i], widths=0.6,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white", markersize=6),
        )
        bp["boxes"][0].set_facecolor(AGENT_COLORS[agent])
        bp["boxes"][0].set_alpha(0.7)
        # Jitter points
        jitter = np.random.normal(0, 0.08, len(data))
        ax.scatter(np.full(len(data), i) + jitter, data, alpha=0.5, s=20, c=AGENT_COLORS[agent], zorder=3)
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels([AGENT_LABELS[a] for a in agents])
    ax.set_ylabel("Average Peer Review Score")
    ax.set_title("Peer Review Scores by Agent (CPU, 15 trials each)")
    ax.axhline(y=8, color="green", linestyle="--", alpha=0.5, label="Accept (≥8)")
    ax.axhline(y=6, color="orange", linestyle="--", alpha=0.5, label="Revision (≥6)")
    ax.legend()
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "scores_by_agent.png")
    plt.close()
    print("Saved: scores_by_agent.png")


# ── 3. Scores by Seed ──────────────────────────────────────────────────

def plot_scores_by_seed(df):
    seeds = sorted(df.seed.unique())
    agents = ["claude", "kimi", "codex"]
    x = np.arange(len(seeds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, agent in enumerate(agents):
        means = []
        stds = []
        for seed in seeds:
            vals = df[(df.agent == agent) & (df.seed == seed)]["avg_score"]
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)
        ax.bar(x + i * width, means, width, yerr=stds, label=AGENT_LABELS[agent],
               color=AGENT_COLORS[agent], alpha=0.8, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace(" And ", " & ") for s in seeds], rotation=15, ha="right")
    ax.set_ylabel("Avg Peer Review Score")
    ax.set_title("Peer Review Scores by Seed Topic (mean ± std across 3 trials)")
    ax.legend()
    ax.axhline(y=8, color="green", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "scores_by_seed.png")
    plt.close()
    print("Saved: scores_by_seed.png")


# ── 4. Reviewer Bias ───────────────────────────────────────────────────

def plot_reviewer_bias(df):
    reviewer_cols = [c for c in df.columns if c.startswith("reviewer_")]
    reviewer_data = {}
    for col in reviewer_cols:
        name = col.replace("reviewer_", "")
        vals = df[col].dropna()
        if len(vals) > 0:
            reviewer_data[name] = vals

    if not reviewer_data:
        print("No reviewer data found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    names = sorted(reviewer_data.keys())
    for i, name in enumerate(names):
        data = reviewer_data[name]
        bp = ax.boxplot(data, positions=[i], widths=0.6, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white", markersize=6))
        bp["boxes"][0].set_alpha(0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Score Given")
    ax.set_title("Reviewer Bias: Score Distribution per Reviewer")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "reviewer_bias.png")
    plt.close()
    print("Saved: reviewer_bias.png")


# ── 5. Per-Dimension Radar ─────────────────────────────────────────────

def plot_dimension_radar(df):
    dimensions = [
        "novelty", "soundness", "significance", "clarity",
        "reproducibility", "experimental_rigor", "references",
        "reference_integrity", "results_integrity",
    ]
    agents = ["claude", "kimi", "codex"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    for agent in agents:
        agent_df = df[df.agent == agent]
        means = []
        for dim in dimensions:
            dim_cols = [c for c in agent_df.columns if c.startswith(f"dim_{dim}_")]
            vals = []
            for col in dim_cols:
                vals.extend(agent_df[col].dropna().tolist())
            means.append(np.mean(vals) if vals else 0)
        means += means[:1]
        ax.plot(angles, means, "o-", label=AGENT_LABELS[agent], color=AGENT_COLORS[agent], linewidth=2)
        ax.fill(angles, means, alpha=0.1, color=AGENT_COLORS[agent])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace("_", "\n") for d in dimensions], size=9)
    ax.set_ylim(0, 10)
    ax.set_title("Per-Dimension Scores (averaged across all reviewers & trials)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.savefig(FIG_DIR / "dimension_radar.png")
    plt.close()
    print("Saved: dimension_radar.png")


# ── 6. Wall Time Comparison ────────────────────────────────────────────

def plot_wall_time(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    agents = ["claude", "kimi", "codex"]
    for i, agent in enumerate(agents):
        data = df[df.agent == agent]["wall_time_h"].dropna()
        bp = ax.boxplot(
            data, positions=[i], widths=0.6,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white", markersize=6),
        )
        bp["boxes"][0].set_facecolor(AGENT_COLORS[agent])
        bp["boxes"][0].set_alpha(0.7)
        jitter = np.random.normal(0, 0.08, len(data))
        ax.scatter(np.full(len(data), i) + jitter, data, alpha=0.5, s=20, c=AGENT_COLORS[agent], zorder=3)

    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels([AGENT_LABELS[a] for a in agents])
    ax.set_ylabel("Wall Time (hours)")
    ax.set_title("Wall Time by Agent (CPU runs)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "wall_time.png")
    plt.close()
    print("Saved: wall_time.png")


# ── 7. Time Breakdown by Stage ─────────────────────────────────────────

def plot_time_breakdown(df):
    stages = ["ideation", "self_review_idea", "experiments", "self_review_experiment",
              "paper", "self_review_paper", "review"]
    agents = ["claude", "kimi", "codex"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(agents))
    bottom = np.zeros(len(agents))

    colors = plt.cm.Set3(np.linspace(0, 1, len(stages)))
    for j, stage in enumerate(stages):
        means = []
        for agent in agents:
            col = f"time_{stage}"
            if col in df.columns:
                vals = df[df.agent == agent][col].dropna()
                means.append(vals.mean() / 60 if len(vals) > 0 else 0)
            else:
                means.append(0)
        bars = ax.bar(x, means, 0.6, bottom=bottom, label=stage.replace("_", " ").title(), color=colors[j])
        bottom += means

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[a] for a in agents])
    ax.set_ylabel("Time (minutes)")
    ax.set_title("Average Time Breakdown by Pipeline Stage")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.savefig(FIG_DIR / "time_breakdown.png")
    plt.close()
    print("Saved: time_breakdown.png")


# ── 8. Self-Review Iterations ──────────────────────────────────────────

def plot_self_review_iterations(df):
    gates = ["self_review_idea", "self_review_experiment", "self_review_paper"]
    agents = ["claude", "kimi", "codex"]
    x = np.arange(len(gates))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, agent in enumerate(agents):
        means = []
        for gate in gates:
            col = f"sr_{gate}_count"
            vals = df[df.agent == agent][col].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
        ax.bar(x + i * width, means, width, label=AGENT_LABELS[agent], color=AGENT_COLORS[agent], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Idea", "Experiment", "Paper"])
    ax.set_ylabel("Avg Self-Review Iterations")
    ax.set_title("Self-Review Iterations per Gate")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "self_review_iterations.png")
    plt.close()
    print("Saved: self_review_iterations.png")


# ── 9. Score Heatmap ───────────────────────────────────────────────────

def plot_score_heatmap(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    agents = ["claude", "kimi", "codex"]

    for ax, agent in zip(axes, agents):
        adf = df[df.agent == agent]
        pivot = adf.pivot_table(values="avg_score", index="seed", columns="trial", aggfunc="first")
        pivot = pivot.sort_index()

        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=10, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"Trial {c}" for c in pivot.columns])
        if ax == axes[0]:
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([s.replace(" And ", "\n& ") for s in pivot.index], fontsize=9)
        else:
            ax.set_yticks([])
        ax.set_title(AGENT_LABELS[agent])

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=10,
                            color="white" if val < 4 else "black")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Avg Review Score")
    fig.suptitle("Peer Review Score Heatmap (Seed × Trial)", fontsize=14, y=1.02)
    fig.savefig(FIG_DIR / "score_heatmap.png")
    plt.close()
    print("Saved: score_heatmap.png")


# ── 10. Ideas Tried ────────────────────────────────────────────────────

def plot_ideas_tried(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    agents = ["claude", "kimi", "codex"]
    for i, agent in enumerate(agents):
        data = df[df.agent == agent]["ideas_tried"].dropna()
        jitter = np.random.normal(0, 0.1, len(data))
        ax.scatter(np.full(len(data), i) + jitter, data, alpha=0.6, s=40,
                   c=AGENT_COLORS[agent], label=AGENT_LABELS[agent])
        ax.scatter([i], [data.mean()], marker="D", s=100, c=AGENT_COLORS[agent],
                   edgecolors="black", zorder=5)

    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels([AGENT_LABELS[a] for a in agents])
    ax.set_ylabel("Ideas Tried")
    ax.set_title("Number of Ideas Tried per Trial")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "ideas_tried.png")
    plt.close()
    print("Saved: ideas_tried.png")


# ── 11. Weakness Analysis ──────────────────────────────────────────────

def analyze_weaknesses(df):
    weakness_counts = defaultdict(lambda: Counter())

    for agent in ["claude", "kimi", "codex"]:
        agent_dir = BASE / "results" / agent
        if not agent_dir.exists():
            continue
        for reviews_file in sorted(agent_dir.rglob("reviews.json")):
            try:
                reviews = json.loads(reviews_file.read_text())
            except Exception:
                continue
            for rev in reviews.get("reviews", []):
                for w in rev.get("weaknesses", []):
                    if not isinstance(w, str) or len(w) < 10:
                        continue
                    w_lower = w.lower()
                    if "novelty" in w_lower or "incremental" in w_lower:
                        weakness_counts[agent]["Limited novelty"] += 1
                    elif "baseline" in w_lower or "comparison" in w_lower:
                        weakness_counts[agent]["Missing baselines"] += 1
                    elif "statistic" in w_lower or "error bar" in w_lower or "significance" in w_lower:
                        weakness_counts[agent]["No statistical tests"] += 1
                    elif "citation" in w_lower or "reference" in w_lower or "fabricat" in w_lower:
                        weakness_counts[agent]["Reference issues"] += 1
                    elif "reproducib" in w_lower or "detail" in w_lower:
                        weakness_counts[agent]["Reproducibility gaps"] += 1
                    elif "scale" in w_lower or "benchmark" in w_lower or "dataset" in w_lower:
                        weakness_counts[agent]["Limited scale/datasets"] += 1
                    elif "code" in w_lower and ("mismatch" in w_lower or "discrepan" in w_lower):
                        weakness_counts[agent]["Code-paper mismatch"] += 1

    categories = sorted(set(c for agent_counts in weakness_counts.values() for c in agent_counts))
    if not categories:
        print("No weaknesses found")
        return

    agents = ["claude", "kimi", "codex"]
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, agent in enumerate(agents):
        vals = [weakness_counts[agent].get(c, 0) for c in categories]
        ax.bar(x + i * width, vals, width, label=AGENT_LABELS[agent], color=AGENT_COLORS[agent], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Common Weaknesses Identified by Reviewers")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "weaknesses.png")
    plt.close()
    print("Saved: weaknesses.png")


# ── 12. Summary Table ──────────────────────────────────────────────────

def print_summary(df):
    print("\n" + "=" * 70)
    print("CPU EXPERIMENT SUMMARY — 3 Agents × 5 Seeds × 3 Trials = 45 Runs")
    print("=" * 70)

    for agent in ["claude", "kimi", "codex"]:
        adf = df[df.agent == agent]
        print(f"\n--- {AGENT_LABELS[agent]} ---")
        print(f"  Trials: {len(adf)}")
        print(f"  Avg score: {adf.avg_score.mean():.2f} ± {adf.avg_score.std():.2f}")
        print(f"  Best score: {adf.avg_score.max():.1f}")
        print(f"  Worst score: {adf.avg_score.min():.1f}")
        wt = adf["wall_time_h"].dropna()
        if len(wt) > 0:
            print(f"  Avg wall time: {wt.mean():.1f}h (range: {wt.min():.1f}–{wt.max():.1f}h)")
        ideas = adf["ideas_tried"].dropna()
        if len(ideas) > 0:
            print(f"  Avg ideas tried: {ideas.mean():.1f}")

    # Per-seed comparison
    print(f"\n--- Per-Seed Average Scores ---")
    pivot = df.pivot_table(values="avg_score", index="seed", columns="agent", aggfunc="mean")
    pivot["best"] = pivot.idxmax(axis=1)
    print(pivot.round(2).to_string())

    # Per-reviewer average
    print(f"\n--- Per-Reviewer Average Score Given ---")
    reviewer_cols = [c for c in df.columns if c.startswith("reviewer_")]
    for col in sorted(reviewer_cols):
        name = col.replace("reviewer_", "")
        for agent in ["claude", "kimi", "codex"]:
            vals = df[df.agent == agent][col].dropna()
            if len(vals) > 0:
                print(f"  {name} reviewing {AGENT_LABELS[agent]}: {vals.mean():.2f} (n={len(vals)})")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    df = collect_data()
    print(f"Collected {len(df)} trials: {df.agent.value_counts().to_dict()}")

    print_summary(df)

    print("\nGenerating figures...")
    plot_scores_by_agent(df)
    plot_scores_by_seed(df)
    plot_reviewer_bias(df)
    plot_dimension_radar(df)
    plot_wall_time(df)
    plot_time_breakdown(df)
    plot_self_review_iterations(df)
    plot_score_heatmap(df)
    plot_ideas_tried(df)
    analyze_weaknesses(df)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
