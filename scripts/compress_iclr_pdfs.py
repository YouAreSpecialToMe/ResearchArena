#!/usr/bin/env python3
"""Compress oversized ICLR PDFs (>10MB) using ghostscript, then resubmit."""

import json
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PDF_DIR = BASE / "analysis" / "iclr2025_baseline" / "pdfs"
LOG_FILE = BASE / "analysis" / "stanford_reviews" / "submission_log.jsonl"

MAX_SIZE_MB = 10


def get_failed_indices():
    """Get indices of failed submissions from log."""
    failed = set()
    if not LOG_FILE.exists():
        return failed
    for line in LOG_FILE.read_text().strip().split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("status") == "failed" and "iclr2025" in d.get("agent", ""):
            failed.add((d["agent"], d["index"]))
    return failed


def compress_pdf(input_path, output_path):
    """Compress PDF with ghostscript."""
    cmd = [
        "gs", "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/ebook",
        "-dNOPAUSE", "-dBATCH", "-dQUIET",
        "-dColorImageResolution=150",
        "-dGrayImageResolution=150",
        "-dMonoImageResolution=300",
        f"-sOutputFile={output_path}",
        str(input_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    # Find all PDFs > 10MB
    oversized = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        size_mb = pdf.stat().st_size / 1024 / 1024
        if size_mb > MAX_SIZE_MB:
            oversized.append((pdf, size_mb))

    print(f"Found {len(oversized)} PDFs > {MAX_SIZE_MB}MB")

    compressed = 0
    still_too_big = []
    for pdf, orig_size in oversized:
        compressed_path = pdf.with_suffix(".compressed.pdf")
        ok = compress_pdf(pdf, compressed_path)

        if ok and compressed_path.exists():
            new_size = compressed_path.stat().st_size / 1024 / 1024
            if new_size <= MAX_SIZE_MB:
                # Replace original with compressed
                compressed_path.rename(pdf)
                compressed += 1
                print(f"  {pdf.name}: {orig_size:.1f}MB -> {new_size:.1f}MB OK")
            else:
                # Try more aggressive compression
                aggressive_path = pdf.with_suffix(".aggressive.pdf")
                cmd2 = [
                    "gs", "-sDEVICE=pdfwrite",
                    "-dCompatibilityLevel=1.4",
                    "-dPDFSETTINGS=/screen",
                    "-dNOPAUSE", "-dBATCH", "-dQUIET",
                    "-dColorImageResolution=72",
                    "-dGrayImageResolution=72",
                    f"-sOutputFile={aggressive_path}",
                    str(pdf),
                ]
                subprocess.run(cmd2, capture_output=True)
                if aggressive_path.exists():
                    agg_size = aggressive_path.stat().st_size / 1024 / 1024
                    if agg_size <= MAX_SIZE_MB:
                        aggressive_path.rename(pdf)
                        compressed += 1
                        print(f"  {pdf.name}: {orig_size:.1f}MB -> {agg_size:.1f}MB OK (aggressive)")
                    else:
                        still_too_big.append((pdf.name, orig_size, agg_size))
                        print(f"  {pdf.name}: {orig_size:.1f}MB -> {agg_size:.1f}MB STILL TOO BIG")
                        aggressive_path.unlink(missing_ok=True)
                compressed_path.unlink(missing_ok=True)
        else:
            print(f"  {pdf.name}: compression failed")

    print(f"\nCompressed: {compressed}/{len(oversized)}")
    if still_too_big:
        print(f"Still too big: {len(still_too_big)}")
        for name, orig, final in still_too_big:
            print(f"  {name}: {final:.1f}MB")


if __name__ == "__main__":
    main()
