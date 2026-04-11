#!/usr/bin/env python3
"""Submit a single paper to paperreview.ai (Stanford Agentic Reviewer).

Usage:
    python scripts/stanford_submit_one.py <pdf_path> <index> [--venue iclr]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EMAIL = "zhxzhang4ever@gmail.com"
LOG_DIR = Path(__file__).resolve().parent.parent / "analysis" / "stanford_reviews"


def ensure_playwright():
    """Install playwright + chromium if needed."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "-q"])
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])


def submit(pdf_path, venue="iclr", max_retries=3):
    from playwright.sync_api import sync_playwright

    VENUE_MAP = {
        "neurips": "NeurIPS", "icml": "ICML", "iclr": "ICLR",
        "cvpr": "CVPR", "aaai": "AAAI", "acl": "ACL",
    }
    venue_value = VENUE_MAP.get(venue.lower(), "ICLR")

    for attempt in range(max_retries):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("https://paperreview.ai", timeout=60000)

            page.fill('input[name="email"]', EMAIL)
            page.select_option('select[name="venue"]', value=venue_value)
            page.set_input_files('input[name="pdf"]', str(pdf_path))
            page.click('button[type="submit"]', timeout=10000)
            page.wait_for_timeout(10000)
            body = page.text_content("body")
            browser.close()

        if "Submission Successful" in body or "successful" in body.lower():
            return body, True
        elif "Rate Limit" in body or "rate limit" in body.lower():
            wait = 300 * (attempt + 1)
            print(f"  Rate limited (attempt {attempt+1}/{max_retries}). Waiting {wait}s...")
            time.sleep(wait)
            continue
        else:
            return body, False

    return body, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("index", type=int)
    parser.add_argument("--venue", default="iclr")
    parser.add_argument("--agent", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--trial", default="")
    args = parser.parse_args()

    import socket
    print(f"Node: {socket.gethostname()}")
    print(f"Submitting: {args.pdf_path}")
    print(f"Email: {EMAIL}, Venue: {args.venue}")

    ensure_playwright()

    confirmation, success = submit(args.pdf_path, venue=args.venue)
    status = "submitted" if success else "failed"
    print(f"Result: {status}")

    # Append to shared log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "index": args.index,
        "agent": args.agent,
        "seed": args.seed,
        "trial": args.trial,
        "pdf_path": args.pdf_path,
        "email": EMAIL,
        "status": status,
        "success": success,
        "node": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log_path = LOG_DIR / "submission_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Logged to {log_path}")


if __name__ == "__main__":
    main()
