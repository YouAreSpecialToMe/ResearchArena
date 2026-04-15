from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import ACCESS_PAGE_SIZE, CALIBRATION_ROOT, PILOT_SEED, WINDOW, ensure_layout, write_json
from .workloads import build_sqlite_db, generate_sqlite_traces, make_synthetic_traces, realized_overlap


def _active_pages(events: list[dict[str, Any]]) -> int:
    if not events:
        return 1
    counts = []
    for start in range(0, len(events), WINDOW):
        window = events[start: start + WINDOW]
        counts.append(len({(e["file_id"], e["page_id"]) for e in window}))
    return max(1, int(np.mean(counts)))


def _ensure_backing_file(path: Path, pages: int) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        payload = b"x" * ACCESS_PAGE_SIZE
        for _ in range(pages):
            handle.write(payload)
    return path


def _probe_latencies(path: Path, samples: int = 30) -> tuple[float, float]:
    fd = os.open(path, os.O_RDONLY)
    hit = []
    miss = []
    try:
        page_count = max(1, os.path.getsize(path) // ACCESS_PAGE_SIZE)
        for idx in range(samples):
            page = (idx * 997) % page_count
            offset = page * ACCESS_PAGE_SIZE
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, offset, ACCESS_PAGE_SIZE, os.POSIX_FADV_DONTNEED)
            start = time.perf_counter_ns()
            os.pread(fd, ACCESS_PAGE_SIZE, offset)
            miss.append((time.perf_counter_ns() - start) / 1000.0)
            start = time.perf_counter_ns()
            os.pread(fd, ACCESS_PAGE_SIZE, offset)
            hit.append((time.perf_counter_ns() - start) / 1000.0)
    finally:
        os.close(fd)
    return float(np.median(hit)), float(np.median(miss))


def _isolated_avg_latency(events: list[dict[str, Any]], hit_us: float, miss_us: float, tenant_count: int) -> dict[str, float]:
    out = {}
    for tenant in range(tenant_count):
        seen = set()
        total = 0.0
        refs = 0
        for event in events:
            if event["tenant_id"] != tenant:
                continue
            refs += 1
            key = (event["file_id"], event["page_id"])
            if key in seen:
                total += hit_us
            else:
                total += miss_us
                seen.add(key)
        out[str(tenant)] = float(total / max(1, refs))
    return out


def prepare_and_calibrate() -> dict[str, Any]:
    ensure_layout()
    synthetic = make_synthetic_traces()
    sqlite = generate_sqlite_traces()
    traces = synthetic + sqlite
    families = []
    latency_by_family: dict[str, dict[str, float]] = {}
    isolated_avg_latency: dict[str, dict[str, float]] = {}
    sqlite_db = build_sqlite_db()
    for trace in traces:
        if trace["seed"] != PILOT_SEED:
            continue
        family = trace["family"]
        events = trace["events"]
        tenant_count = trace["tenant_count"]
        if family.startswith("SQLiteTraceMix"):
            probe_file = sqlite_db
        else:
            probe_file = _ensure_backing_file(CALIBRATION_ROOT / f"{family}.bin", trace["meta"]["unique_pages"] + 32)
        hit_us, miss_us = _probe_latencies(probe_file)
        active_pages = _active_pages(events)
        overlap = realized_overlap(events, tenant_count)
        tenant_penalties = {str(t): float((miss_us - hit_us) * (1.0 + 0.04 * t)) for t in range(tenant_count)}
        families.append(
            {
                "workload_family": family,
                "tenant_count": tenant_count,
                "active_pages": active_pages,
                "overlap_ratio": overlap["overlap_ratio"],
                "hit_latency_us": hit_us,
                "miss_latency_us": miss_us,
                "family_penalty": float(miss_us - hit_us),
                "tenant_penalties": tenant_penalties,
                "budgets": {
                    "tight": max(16, int(round(active_pages * 0.35))),
                    "medium": max(16, int(round(active_pages * 0.55))),
                    "loose": max(16, int(round(active_pages * 0.75))),
                },
            }
        )
        latency_by_family[family] = {"hit_latency_us": hit_us, "miss_latency_us": miss_us}
        isolated_avg_latency[family] = _isolated_avg_latency(events, hit_us, miss_us, tenant_count)
    manifest = {
        "families": families,
        "manifest": {row["workload_family"]: row for row in families},
        "latency_by_family": latency_by_family,
        "isolated_avg_latency": isolated_avg_latency,
    }
    write_json(CALIBRATION_ROOT / "manifest.json", manifest)
    return manifest
