#!/usr/bin/env python3
"""Fetch all ICLR review scores from paperreview.ai using Playwright."""
import json
import re
import sys
from pathlib import Path

BASE = Path('/home/zz865/pythonProject/autoresearch')
token_file = BASE / 'analysis' / 'iclr2025_baseline' / 'stanford_reviews' / 'iclr_tokens.json'
score_file = BASE / 'analysis' / 'iclr2025_baseline' / 'stanford_reviews' / 'iclr_scores_combined.json'

tokens = json.loads(token_file.read_text())

# Load existing
if score_file.exists():
    results = json.loads(score_file.read_text())
else:
    results = {}

remaining = {k: v for k, v in tokens.items() if k not in results or results[k].get('score') is None}
print(f'Total: {len(tokens)}, done: {len(tokens)-len(remaining)}, remaining: {len(remaining)}', flush=True)

if not remaining:
    print('All scores already fetched!')
    sys.exit(0)

from playwright.sync_api import sync_playwright

for i, (key, token) in enumerate(sorted(remaining.items()), 1):
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
        results[key] = {"score": score}
        print(f"  [{i}/{len(remaining)}] {key}: score={score}", flush=True)
    except Exception as e:
        results[key] = {"score": None, "error": str(e)}
        print(f"  [{i}/{len(remaining)}] {key}: ERROR {e}", flush=True)
    
    if i % 20 == 0:
        score_file.write_text(json.dumps(results, indent=2))

score_file.write_text(json.dumps(results, indent=2))
valid = sum(1 for v in results.values() if v['score'] is not None)
print(f'\nDone. {valid}/{len(results)} valid scores saved.', flush=True)
