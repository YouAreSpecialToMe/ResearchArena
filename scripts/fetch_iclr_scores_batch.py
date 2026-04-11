#!/usr/bin/env python3
import json
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = Path('/home/zz865/pythonProject/autoresearch')
token_file = BASE / 'analysis' / 'iclr2025_baseline' / 'stanford_reviews' / 'iclr_tokens.json'
score_file = BASE / 'analysis' / 'iclr2025_baseline' / 'stanford_reviews' / 'iclr_scores.json'
tokens = json.loads(token_file.read_text())

# Load existing scores
if score_file.exists():
    scores = json.loads(score_file.read_text())
else:
    scores = {}

remaining = {k: v for k, v in tokens.items() if k not in scores or scores[k].get('score') is None}
print(f'Total: {len(tokens)}, already fetched: {len(tokens)-len(remaining)}, remaining: {len(remaining)}', flush=True)

from playwright.sync_api import sync_playwright

def fetch_one(key_token):
    key, token = key_token
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"https://paperreview.ai/review?token={token}", timeout=60000)
            page.wait_for_timeout(10000)
            body = page.text_content("body") or ""
            browser.close()
        
        m = re.search(r"Estimated Score[:\s]*([\d.]+)\s*/\s*10", body)
        score = float(m.group(1)) if m else None
        return key, score, body[:500]
    except Exception as e:
        return key, None, str(e)

# Process sequentially but save incrementally
done = 0
for key, token in sorted(remaining.items()):
    k, score, snippet = fetch_one((key, token))
    scores[k] = {"score": score, "snippet": snippet[:200]}
    done += 1
    print(f"  [{done}/{len(remaining)}] {k}: score={score}", flush=True)
    
    if done % 10 == 0:
        score_file.write_text(json.dumps(scores, indent=2))

score_file.write_text(json.dumps(scores, indent=2))
print(f"\nDone. Scores saved to {score_file}", flush=True)

# Summary
valid = [v["score"] for v in scores.values() if v["score"] is not None]
print(f"Valid scores: {len(valid)}/{len(scores)}", flush=True)
if valid:
    import numpy as np
    arr = np.array(valid)
    print(f"Mean: {arr.mean():.2f}, Std: {arr.std():.2f}", flush=True)
