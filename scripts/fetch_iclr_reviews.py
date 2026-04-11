#!/usr/bin/env python3
"""Fetch Stanford AI Reviewer scores for ICLR 2025 baseline papers.

Polls email for review tokens, fetches scores, and produces summary statistics.

Usage:
    python scripts/fetch_iclr_reviews.py [--collect-tokens] [--fetch-scores] [--summarize]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
LOG_DIR = BASE / "analysis" / "iclr2025_baseline" / "stanford_reviews"
MANIFEST = BASE / "analysis" / "iclr2025_baseline" / "manifest_100.json"


def ensure_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "-q"])
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])


def fetch_score(token):
    """Fetch a review score by token."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"https://paperreview.ai/review?token={token}", timeout=60000)
        page.wait_for_timeout(8000)
        body = page.text_content("body") or ""
        browser.close()

    m = re.search(r"Estimated Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", body)
    score = float(m.group(1)) if m else None
    return score, body


def collect_tokens():
    """Extract review tokens from submission log / email."""
    token_file = LOG_DIR / "tokens.json"
    # Tokens come from email confirmations — collect from submission log
    log_file = LOG_DIR.parent / "stanford_reviews" / "submission_log.jsonl"
    if not log_file.exists():
        log_file = BASE / "analysis" / "stanford_reviews" / "submission_log.jsonl"

    tokens = {}
    if log_file.exists():
        for line in log_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("agent") == "iclr2025":
                conf = entry.get("confirmation", "")
                m = re.search(r"token=([a-zA-Z0-9_-]+)", conf)
                if m:
                    tokens[entry["index"]] = m.group(1)

    token_file.write_text(json.dumps(tokens, indent=2))
    print(f"Collected {len(tokens)} tokens")
    return tokens


def fetch_scores():
    """Fetch all review scores."""
    ensure_playwright()
    token_file = LOG_DIR / "tokens.json"
    if not token_file.exists():
        print("No tokens found. Run --collect-tokens first.")
        return

    tokens = json.loads(token_file.read_text())
    results_dir = LOG_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    for idx_str, token in sorted(tokens.items(), key=lambda x: int(x[0])):
        idx = int(idx_str)
        out_file = results_dir / f"review_{idx:03d}.json"
        if out_file.exists():
            continue

        score, body = fetch_score(token)
        result = {"index": idx, "token": token, "score": score, "raw_text": body[:5000]}
        out_file.write_text(json.dumps(result, indent=2))
        print(f"[{idx:3d}] score={score}")
        time.sleep(2)


def summarize():
    """Produce summary statistics."""
    import numpy as np

    manifest = json.loads(MANIFEST.read_text())
    results_dir = LOG_DIR / "results"

    scores = []
    accepted_scores = []
    rejected_scores = []

    for p in manifest:
        idx = p["index"]
        result_file = results_dir / f"review_{idx:03d}.json"
        if not result_file.exists():
            continue
        result = json.loads(result_file.read_text())
        if result["score"] is None:
            continue

        scores.append(result["score"])
        if p["decision"].startswith("ICLR"):
            accepted_scores.append(result["score"])
        elif p["decision"] == "Rejected":
            rejected_scores.append(result["score"])

    scores = np.array(scores)
    accepted_scores = np.array(accepted_scores)
    rejected_scores = np.array(rejected_scores)

    print("=" * 60)
    print("ICLR 2025 BASELINE — Stanford AI Reviewer Scores")
    print("=" * 60)
    print(f"\nAll papers:      n={len(scores):3d}, mean={scores.mean():.2f}, std={scores.std():.2f}")
    if len(accepted_scores) > 0:
        print(f"Accepted papers: n={len(accepted_scores):3d}, mean={accepted_scores.mean():.2f}, std={accepted_scores.std():.2f}")
    if len(rejected_scores) > 0:
        print(f"Rejected papers: n={len(rejected_scores):3d}, mean={rejected_scores.mean():.2f}, std={rejected_scores.std():.2f}")

    print(f"\n>= 5.0: {(scores >= 5.0).mean():.0%}")
    print(f">= 6.0: {(scores >= 6.0).mean():.0%}")
    print(f">= 7.0: {(scores >= 7.0).mean():.0%}")

    # Compare with our agents
    print("\n--- Comparison ---")
    print(f"ICLR 2025 (human):  {scores.mean():.2f} (n={len(scores)})")
    print(f"Claude Code:        5.45 (n=39)")
    print(f"FARS:               5.06 (n=102)")
    print(f"Codex:              4.93 (n=39)")
    print(f"Kimi Code:          4.24 (n=39)")

    # Save summary
    summary = {
        "total": len(scores),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "accepted_mean": float(accepted_scores.mean()) if len(accepted_scores) > 0 else None,
        "rejected_mean": float(rejected_scores.mean()) if len(rejected_scores) > 0 else None,
        "pct_above_5": float((scores >= 5.0).mean()),
        "pct_above_6": float((scores >= 6.0).mean()),
    }
    (LOG_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {LOG_DIR / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-tokens", action="store_true")
    parser.add_argument("--fetch-scores", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.collect_tokens:
        collect_tokens()
    if args.all or args.fetch_scores:
        fetch_scores()
    if args.all or args.summarize:
        summarize()


if __name__ == "__main__":
    main()
