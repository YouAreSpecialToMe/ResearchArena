#!/usr/bin/env python3
"""Match ICLR 2025 papers to arXiv IDs and download PDFs.

Produces 3 groups:
  - accepted: 100 accepted papers
  - rejected: 100 rejected papers
  - random:   100 randomly sampled (mix of accepted + rejected)

Usage:
    python scripts/match_iclr_arxiv.py [--download] [--download-only]
"""

import argparse
import csv
import io
import json
import random
import re
import requests
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "analysis" / "iclr2025_baseline"
MATCH_FILE = OUTDIR / "arxiv_matched_all.json"
PDF_DIR = OUTDIR / "pdfs"

CSV_URL = "https://huggingface.co/datasets/cuijiaxing/ICLR_2025_Accepted_Papers/resolve/main/ICLR_2025_results.csv"

TARGET_PER_GROUP = 110  # Find 110 to have buffer, then pick 100


def load_papers():
    """Download and parse the ICLR 2025 CSV."""
    r = requests.get(CSV_URL, timeout=60)
    reader = csv.DictReader(io.StringIO(r.text))
    all_papers = list(reader)

    accepted = [p for p in all_papers if p["Decision"].startswith("ICLR 2025")]
    rejected = [p for p in all_papers if p["Decision"] == "Rejected"]

    random.seed(42)
    random.shuffle(accepted)
    random.shuffle(rejected)

    print(f"Total: {len(all_papers)} (accepted={len(accepted)}, rejected={len(rejected)})")
    return accepted, rejected


def search_arxiv(title, retries=3):
    """Search arXiv by title keywords. Returns arxiv_id or None."""
    words = [w for w in re.split(r"\W+", title) if len(w) > 3][:6]
    if len(words) < 3:
        return None

    query = "ti:" + "+AND+ti:".join(urllib.parse.quote(w) for w in words)
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results=5"

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
        except Exception:
            time.sleep(10)
            continue

        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f"    Rate limited, waiting {wait}s...", flush=True)
            time.sleep(wait)
            continue

        if resp.status_code != 200 or not resp.text.strip():
            return None

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            return None

        for entry in root.findall("atom:entry", ns):
            arxiv_title = (
                entry.find("atom:title", ns)
                .text.strip()
                .replace("\n", " ")
                .replace("  ", " ")
            )
            t1 = arxiv_title.lower().strip().rstrip(".")
            t2 = title.lower().strip().rstrip(".")
            if t1 == t2 or (len(t1) >= 30 and len(t2) >= 30 and t1[:30] == t2[:30]):
                arxiv_id = entry.find("atom:id", ns).text.strip().split("/abs/")[-1]
                return arxiv_id
        return None

    return None


def match_group(papers, group_name, target, existing_matches):
    """Match papers from a list until we reach target count."""
    seen_titles = {m["title"] for m in existing_matches}
    found = [m for m in existing_matches if m.get("group") == group_name]
    print(f"\n=== Matching {group_name} (have {len(found)}, need {target}) ===", flush=True)

    for i, p in enumerate(papers):
        if len(found) >= target:
            break
        if p["Title"] in seen_titles:
            continue

        arxiv_id = search_arxiv(p["Title"])
        if arxiv_id:
            entry = {
                "title": p["Title"],
                "avg_score": float(p["Average Score"]),
                "decision": p["Decision"],
                "area": p["Author-defined Area"],
                "arxiv_id": arxiv_id,
                "group": group_name,
            }
            found.append(entry)
            seen_titles.add(p["Title"])
            print(
                f"  [{group_name}] ({len(found):3d}/{target}) {arxiv_id:20s} | {p['Title'][:50]}",
                flush=True,
            )

            if len(found) % 10 == 0:
                print(f"  --- {group_name}: {len(found)}/{target} ---", flush=True)

        time.sleep(5)

    print(f"  {group_name} done: {len(found)} matches", flush=True)
    return found


def match_phase():
    """Phase 1: Match papers for all 3 groups."""
    accepted, rejected = load_papers()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load existing matches
    if MATCH_FILE.exists():
        all_matches = json.loads(MATCH_FILE.read_text())
        print(f"Resuming from {len(all_matches)} existing matches", flush=True)
    else:
        all_matches = []

    # Group 1: Accepted papers
    acc_matches = match_group(accepted, "accepted", TARGET_PER_GROUP, all_matches)

    # Group 2: Rejected papers
    rej_matches = match_group(rejected, "rejected", TARGET_PER_GROUP, all_matches)

    # Group 3: Random (from both pools, excluding already matched)
    acc_titles = {m["title"] for m in acc_matches}
    rej_titles = {m["title"] for m in rej_matches}

    # Build random pool: mix remaining accepted + rejected
    remaining_acc = [p for p in accepted if p["Title"] not in acc_titles]
    remaining_rej = [p for p in rejected if p["Title"] not in rej_titles]
    random_pool = remaining_acc + remaining_rej
    random.seed(123)
    random.shuffle(random_pool)

    rand_matches = match_group(random_pool, "random", TARGET_PER_GROUP, all_matches)

    # Combine and save
    all_matches = acc_matches + rej_matches + rand_matches
    MATCH_FILE.write_text(json.dumps(all_matches, indent=2))

    for grp in ["accepted", "rejected", "random"]:
        n = sum(1 for m in all_matches if m["group"] == grp)
        print(f"  {grp}: {n} matches")
    print(f"\nTotal: {len(all_matches)} matches saved to {MATCH_FILE}", flush=True)


def _download_one(args):
    """Download a single PDF. Used by ThreadPoolExecutor."""
    group, i, arxiv_id, title, pdf_path = args
    pdf_path = Path(pdf_path)

    if pdf_path.exists() and pdf_path.stat().st_size > 10000:
        return group, i, True, 0, title

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and "pdf" in r.headers.get("content-type", ""):
                pdf_path.write_bytes(r.content)
                size_mb = len(r.content) / 1024 / 1024
                return group, i, True, size_mb, title
            elif r.status_code == 429:
                time.sleep(30 * (attempt + 1))
        except Exception:
            time.sleep(5)
    return group, i, False, 0, title


def download_phase():
    """Phase 2: Download PDFs for all 3 groups (100 per group), concurrently."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not MATCH_FILE.exists():
        print("No matches found. Run matching phase first.")
        sys.exit(1)

    all_matches = json.loads(MATCH_FILE.read_text())
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Build download tasks for all 3 groups
    tasks = []
    group_selected = {}
    for group in ["accepted", "rejected", "random"]:
        group_matches = [m for m in all_matches if m["group"] == group]
        random.seed({"accepted": 100, "rejected": 200, "random": 300}[group])
        selected = random.sample(group_matches, min(100, len(group_matches)))
        for i, p in enumerate(selected):
            p["index"] = i
            pdf_path = PDF_DIR / f"{group}_{i:03d}.pdf"
            tasks.append((group, i, p["arxiv_id"], p["title"], str(pdf_path)))
        group_selected[group] = selected

    print(f"Downloading {len(tasks)} PDFs with 20 concurrent threads...", flush=True)

    results = {g: {"ok": 0, "fail": 0} for g in ["accepted", "rejected", "random"]}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_download_one, t): t for t in tasks}
        for future in as_completed(futures):
            group, i, ok, size_mb, title = future.result()
            if ok:
                results[group]["ok"] += 1
                # Mark pdf_path on the selected entry
                for p in group_selected[group]:
                    if p["index"] == i:
                        p["pdf_path"] = str(PDF_DIR / f"{group}_{i:03d}.pdf")
                        break
            else:
                results[group]["fail"] += 1
                print(f"  [{group}_{i:03d}] FAIL | {title[:45]}", flush=True)

            total_done = sum(r["ok"] + r["fail"] for r in results.values())
            if total_done % 50 == 0:
                print(f"  Progress: {total_done}/{len(tasks)} done", flush=True)

    # Save manifests
    for group in ["accepted", "rejected", "random"]:
        final = [p for p in group_selected[group] if "pdf_path" in p]
        manifest_path = OUTDIR / f"manifest_{group}.json"
        manifest_path.write_text(json.dumps(final, indent=2))
        print(f"  {group}: {results[group]['ok']} OK, {results[group]['fail']} failed -> {manifest_path}", flush=True)

    print(f"\n=== Download Summary ===")
    for grp, r in results.items():
        print(f"  {grp}: {r['ok']} PDFs downloaded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    if not args.download_only:
        match_phase()

    if args.download or args.download_only:
        download_phase()


if __name__ == "__main__":
    main()
