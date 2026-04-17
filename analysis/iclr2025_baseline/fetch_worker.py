#!/usr/bin/env python3
"""Fetch ICLR review overall assessments from paperreview.ai.

Usage: python fetch_worker.py <worker_id> <total_workers>
  Splits 300 papers across workers. Each worker uses 8 threads.
"""
import json, re, sys, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKER_ID = int(sys.argv[1])
TOTAL_WORKERS = int(sys.argv[2])

BASE = Path('/home/zz865/pythonProject/autoresearch/analysis/iclr2025_baseline/stanford_reviews')
tokens = json.load(open(BASE / 'iclr_tokens.json'))
output_file = BASE / f'assessments_w{WORKER_ID:02d}.json'

# Split keys across workers
all_keys = sorted(tokens.keys())
my_keys = [k for i, k in enumerate(all_keys) if i % TOTAL_WORKERS == WORKER_ID]
print(f"Worker {WORKER_ID}/{TOTAL_WORKERS}: {len(my_keys)} papers", flush=True)

# Load existing results
results = json.load(open(output_file)) if output_file.exists() else {}
remaining = [k for k in my_keys if k not in results or not results[k].get('assessment')]
print(f"  Done: {len(my_keys)-len(remaining)}, remaining: {len(remaining)}", flush=True)

if not remaining:
    print("All done!")
    sys.exit(0)


def fetch_one(key):
    from playwright.sync_api import sync_playwright
    token = tokens[key]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"https://paperreview.ai/review?token={token}", timeout=60000)
            page.wait_for_timeout(8000)
            body = page.text_content("body") or ""
            browser.close()

        m = re.search(r"Estimated Score[:\s]*(\d+(?:\.\d+)?)\s*/\s*10", body)
        score = float(m.group(1)) if m else None

        oa_match = re.search(
            r'Estimated Score.*?Calibrated to ICLR scale\)?\s*(.+?)(?:We Value Your Feedback|Submit Your Review|$)',
            body, re.DOTALL
        )
        assessment = oa_match.group(1).strip()[:2000] if oa_match else ""

        return key, {"score": score, "assessment": assessment}
    except Exception as e:
        return key, {"score": None, "assessment": "", "error": str(e)}


THREADS = 8
done = 0
with ThreadPoolExecutor(max_workers=THREADS) as pool:
    futures = {pool.submit(fetch_one, k): k for k in remaining}
    for future in as_completed(futures):
        key, result = future.result()
        results[key] = result
        done += 1
        score = result.get('score')
        alen = len(result.get('assessment', ''))
        if done <= 3 or done % 10 == 0:
            print(f"  [{done}/{len(remaining)}] {key}: score={score}, len={alen}", flush=True)
        if done % 20 == 0:
            output_file.write_text(json.dumps(results, indent=2))

output_file.write_text(json.dumps(results, indent=2))
print(f"Worker {WORKER_ID} done. Saved {len(results)} results.", flush=True)
