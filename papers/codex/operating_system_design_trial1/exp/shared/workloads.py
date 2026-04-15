from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import ARTIFACTS, CACHE_RATIOS, TRACE_LENGTH, dump_json, ensure_dirs


UNIVERSE = 200_000


def zipf_ids(rng: np.random.Generator, count: int, domain: np.ndarray, exponent: float) -> np.ndarray:
    weights = 1.0 / np.power(np.arange(1, domain.size + 1, dtype=np.float64), exponent)
    weights /= weights.sum()
    idx = rng.choice(domain.size, size=count, p=weights)
    return domain[idx]


def phase_loop(seed: int, length: int) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    seg = length // 6
    scan_pages = np.arange(160_000, dtype=np.int32)
    loop_pages = np.arange(160_000, 200_000, dtype=np.int32)
    hot_pages = rng.choice(np.arange(20_000, 200_000, dtype=np.int32), size=20_000, replace=False)
    parts = []
    boundaries = []
    labels = []
    for repeat in range(2):
        scan = scan_pages[np.arange(seg) % scan_pages.size]
        loop = loop_pages[np.arange(seg) % loop_pages.size]
        hot = zipf_ids(rng, seg, hot_pages, 1.2)
        for label, block in [("scan", scan), ("loop", loop), ("hotspot", hot)]:
            parts.append(block)
            boundaries.append(sum(len(p) for p in parts))
            labels.append({"label": label, "end": boundaries[-1], "repeat": repeat})
    trace = np.concatenate(parts).astype(np.int32, copy=False)
    tenants = np.zeros(trace.size, dtype=np.int8)
    return trace, tenants, {"phase_boundaries": labels}


def two_tenant_mix(seed: int, length: int) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    tenant_a_domain = rng.choice(np.arange(UNIVERSE, dtype=np.int32), size=24_000, replace=False)
    tenant_b_scan = rng.choice(np.arange(UNIVERSE, dtype=np.int32), size=180_000, replace=False)
    tenant_b_loop = rng.choice(np.arange(UNIVERSE, dtype=np.int32), size=80_000, replace=False)
    trace = np.empty(length, dtype=np.int32)
    tenant_a_mask = rng.random(length) < 0.7
    tenants = np.zeros(length, dtype=np.int8)
    tenants[~tenant_a_mask] = 1
    trace[tenant_a_mask] = zipf_ids(rng, int(tenant_a_mask.sum()), tenant_a_domain, 1.1)
    modes = []
    current_scan_idx = 0
    current_loop_idx = 0
    for block_start in range(0, length, 300_000):
        block_end = min(block_start + 300_000, length)
        block = block_start // 300_000
        b_mode = "scan" if block % 2 == 0 else "loop"
        modes.append({"label": b_mode, "end": block_end})
        b_mask = ~tenant_a_mask[block_start:block_end]
        b_count = int(b_mask.sum())
        if b_count == 0:
            continue
        if b_mode == "scan":
            values = tenant_b_scan[np.arange(current_scan_idx, current_scan_idx + b_count) % tenant_b_scan.size]
            current_scan_idx += b_count
        else:
            values = tenant_b_loop[np.arange(current_loop_idx, current_loop_idx + b_count) % tenant_b_loop.size]
            current_loop_idx += b_count
        trace[block_start:block_end][b_mask] = values
    return trace, tenants, {"phase_boundaries": modes, "tenant_mix": {"A": 0.7, "B": 0.3}}


def skew_shift(seed: int, length: int) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    exponents = [0.8, 1.3, 0.9, 1.4]
    seg = length // 4
    parts = []
    labels = []
    for exponent in exponents:
        domain = rng.choice(np.arange(UNIVERSE, dtype=np.int32), size=40_000, replace=False)
        parts.append(zipf_ids(rng, seg, domain, exponent))
        labels.append({"label": f"zipf_{exponent}", "end": sum(len(p) for p in parts)})
    trace = np.concatenate(parts).astype(np.int32, copy=False)
    tenants = np.zeros(trace.size, dtype=np.int8)
    return trace, tenants, {"phase_boundaries": labels}


def stationary_zipf(seed: int, length: int) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    domain = np.arange(UNIVERSE, dtype=np.int32)
    trace = zipf_ids(rng, length, domain, 1.1).astype(np.int32, copy=False)
    tenants = np.zeros(trace.size, dtype=np.int8)
    return trace, tenants, {"phase_boundaries": [{"label": "stationary", "end": length}]}


GENERATORS = {
    "PhaseLoop": phase_loop,
    "TwoTenantMix": two_tenant_mix,
    "SkewShift": skew_shift,
    "StationaryZipf": stationary_zipf,
}


def trace_path(family: str, seed: int) -> Path:
    return ARTIFACTS / "traces" / f"{family}_seed{seed}.npy"


def tenant_path(family: str, seed: int) -> Path:
    return ARTIFACTS / "traces" / f"{family}_seed{seed}_tenants.npy"


def manifest_path() -> Path:
    return ARTIFACTS / "data_manifest.json"


def generate_trace(family: str, seed: int, length: int = TRACE_LENGTH) -> dict:
    ensure_dirs()
    trace, tenants, meta = GENERATORS[family](seed, length)
    path = trace_path(family, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, trace)
    np.save(tenant_path(family, seed), tenants)
    first_window = min(300_000, trace.size)
    working_set = int(np.unique(trace[:first_window]).size)
    capacities = {str(ratio): max(16, int(working_set * ratio)) for ratio in CACHE_RATIOS}
    return {
        "family": family,
        "seed": seed,
        "trace_path": str(path),
        "tenant_path": str(tenant_path(family, seed)),
        "trace_length": int(trace.size),
        "unique_pages": int(np.unique(trace).size),
        "working_set_estimate_first_window": working_set,
        "cache_capacities": capacities,
        **meta,
    }


def generate_all(seeds: list[int]) -> list[dict]:
    manifests = []
    for family in GENERATORS:
        for seed in seeds:
            manifests.append(generate_trace(family, seed))
    dump_json(manifest_path(), manifests)
    return manifests


def load_trace(family: str, seed: int) -> np.ndarray:
    return np.load(trace_path(family, seed), mmap_mode="r")


def load_tenants(family: str, seed: int) -> np.ndarray:
    return np.load(tenant_path(family, seed), mmap_mode="r")
