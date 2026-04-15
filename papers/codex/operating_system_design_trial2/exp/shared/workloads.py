from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import ACCESS_PAGE_SIZE, PILOT_SEED, ROOT, TRACE_ROOT, WINDOW, ensure_layout, set_reproducible, write_json

PLANNED_SYNTHETIC_REFS_PER_TENANT = 80_000
PLANNED_SQLITE_REFS_PER_TENANT = 60_000
SYNTHETIC_REFS_PER_TENANT = 20_000
SQLITE_REFS_PER_TENANT = 15_000


def _scaled_period(planned_period: int, planned_refs: int, actual_refs: int) -> int:
    ratio = actual_refs / planned_refs
    return max(1, int(round(planned_period * ratio)))


def _trace_path(name: str, seed: int) -> Path:
    return TRACE_ROOT / f"{name}__seed{seed}.json"


def _dump_trace(name: str, seed: int, tenant_count: int, events: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    unique_pages = len(set((event["file_id"], event["page_id"]) for event in events))
    overlap = realized_overlap(events, tenant_count)
    payload = {
        "family": name,
        "seed": seed,
        "tenant_count": tenant_count,
        "events": events,
        "meta": {
            **meta,
            "references_per_tenant": {str(t): sum(1 for e in events if e["tenant_id"] == t) for t in range(tenant_count)},
            "unique_pages": unique_pages,
            "realized_overlap_ratio": overlap["overlap_ratio"],
            "shared_pages": overlap["shared_pages"],
            "top_shared_page_ranges": overlap["top_ranges"],
        },
    }
    path = _trace_path(name, seed)
    path.write_text(json.dumps(payload))
    return payload


def realized_overlap(events: list[dict[str, Any]], tenant_count: int) -> dict[str, Any]:
    recent_counts: dict[tuple[int, int], Counter] = {}
    shared_pages = set()
    rolling = []
    counts = Counter()
    current_shared = 0
    for idx, event in enumerate(events):
        key = (event["file_id"], event["page_id"])
        counter = recent_counts.setdefault(key, Counter())
        was_shared = len(counter) > 1
        counter[event["tenant_id"]] += 1
        is_shared = len(counter) > 1
        if not was_shared and is_shared:
            current_shared += 1
        if is_shared:
            shared_pages.add(key)
            counts[key[1] // 64] += 1
        if idx >= WINDOW:
            old = events[idx - WINDOW]
            old_key = (old["file_id"], old["page_id"])
            old_counter = recent_counts[old_key]
            old_was_shared = len(old_counter) > 1
            old_counter[old["tenant_id"]] -= 1
            if old_counter[old["tenant_id"]] <= 0:
                del old_counter[old["tenant_id"]]
            old_is_shared = len(old_counter) > 1
            if old_was_shared and not old_is_shared:
                current_shared -= 1
            if not old_counter:
                del recent_counts[old_key]
        recent_total = max(1, len(recent_counts))
        rolling.append(current_shared / recent_total)
    top_ranges = []
    for page_bin, _ in counts.most_common(10):
        top_ranges.append([page_bin * 64, page_bin * 64 + 63])
    return {
        "overlap_ratio": float(np.mean(rolling)) if rolling else 0.0,
        "shared_pages": len(shared_pages),
        "top_ranges": top_ranges,
    }


def make_synthetic_traces() -> list[dict[str, Any]]:
    ensure_layout()
    outputs = []
    for seed in [PILOT_SEED, 11, 23, 37]:
        rng = set_reproducible(seed)
        outputs.append(generate_overlap_shift(seed, rng))
        outputs.append(generate_scan_vs_loop(seed, rng))
        outputs.append(generate_disjoint_phase(seed, rng))
        outputs.append(generate_overlap_shift(seed, rng, family="OverlapShiftLowOverlap-2T", low_overlap=True))
    return outputs


def generate_overlap_shift(seed: int, rng, family: str = "OverlapShift-2T", low_overlap: bool = False) -> dict[str, Any]:
    refs_per_tenant = SYNTHETIC_REFS_PER_TENANT
    shift_period = _scaled_period(20_000, PLANNED_SYNTHETIC_REFS_PER_TENANT, refs_per_tenant)
    hot_size = 2400
    shift_by = 1200
    shared_pages = 24_000
    reuse_probs = [0.8, 0.65 if not low_overlap else 0.35]
    file_id = 0
    events = []
    hot_start = 0
    current = [rng.randrange(shared_pages), rng.randrange(shared_pages)]
    for step in range(refs_per_tenant):
        if step and step % shift_period == 0:
            hot_start = (hot_start + shift_by) % (shared_pages - hot_size)
        hot = range(hot_start, hot_start + hot_size)
        for tenant in range(2):
            use_hot = rng.random() < reuse_probs[tenant]
            if use_hot:
                if rng.random() < 0.72:
                    current[tenant] = hot_start + rng.randrange(hot_size)
                else:
                    current[tenant] = hot_start + ((current[tenant] - hot_start + rng.randint(-32, 32)) % hot_size)
            else:
                current[tenant] = rng.randrange(shared_pages)
            events.append(
                {
                    "logical_time": len(events),
                    "tenant_id": tenant,
                    "file_id": file_id,
                    "page_id": int(current[tenant]),
                    "access_type": "read",
                }
            )
    return _dump_trace(
        family,
        seed,
        2,
        events,
        {
            "generator": "synthetic",
            "low_overlap": low_overlap,
            "length_regime": "smoke_adjusted",
            "planned_refs_per_tenant": PLANNED_SYNTHETIC_REFS_PER_TENANT,
            "actual_refs_per_tenant": refs_per_tenant,
            "shift_period_refs": shift_period,
            "shift_by_pages": shift_by,
            "hot_region_pages": hot_size,
        },
    )


def generate_scan_vs_loop(seed: int, rng) -> dict[str, Any]:
    refs_per_tenant = SYNTHETIC_REFS_PER_TENANT
    file_id = 1
    scan_region = 14_000
    loop_hot_start = 3500
    loop_hot_size = 1800
    restart_every = 3500
    events = []
    scan_anchor = loop_hot_start - 800
    scan_pos = scan_anchor
    hot_pages = list(range(loop_hot_start, loop_hot_start + loop_hot_size))
    loop_current = hot_pages[rng.randrange(loop_hot_size)]
    for step in range(refs_per_tenant):
        if step and step % restart_every == 0:
            scan_pos = scan_anchor + rng.randrange(256)
        events.append(
            {
                "logical_time": len(events),
                "tenant_id": 0,
                "file_id": file_id,
                "page_id": int(scan_pos % scan_region),
                "access_type": "read",
            }
        )
        scan_pos += 1
        if rng.random() < 0.92:
            if rng.random() < 0.8:
                loop_current = hot_pages[rng.randrange(loop_hot_size)]
            else:
                loop_current = loop_hot_start + ((loop_current - loop_hot_start + rng.randint(-16, 16)) % loop_hot_size)
        else:
            loop_current = rng.randrange(scan_region)
        events.append(
            {
                "logical_time": len(events),
                "tenant_id": 1,
                "file_id": file_id,
                "page_id": int(loop_current),
                "access_type": "read",
            }
        )
    return _dump_trace(
        "ScanVsLoop-2T",
        seed,
        2,
        events,
        {
            "generator": "synthetic",
            "length_regime": "smoke_adjusted",
            "planned_refs_per_tenant": PLANNED_SYNTHETIC_REFS_PER_TENANT,
            "actual_refs_per_tenant": refs_per_tenant,
            "restart_every_refs": restart_every,
        },
    )


def generate_disjoint_phase(seed: int, rng) -> dict[str, Any]:
    refs_per_tenant = SYNTHETIC_REFS_PER_TENANT
    region_size = 12_000
    phase_period = _scaled_period(20_000, PLANNED_SYNTHETIC_REFS_PER_TENANT, refs_per_tenant)
    events = []
    starts = [0, 0]
    currents = [rng.randrange(region_size), rng.randrange(region_size)]
    for step in range(refs_per_tenant):
        if step and step % phase_period == 0:
            starts[0] = (starts[0] + 2000) % (region_size - 2400)
            starts[1] = (starts[1] + 1600) % (region_size - 2400)
        for tenant in range(2):
            if rng.random() < 0.74:
                base = starts[tenant]
                currents[tenant] = base + rng.randrange(2400)
            else:
                currents[tenant] = rng.randrange(region_size)
            events.append(
                {
                    "logical_time": len(events),
                    "tenant_id": tenant,
                    "file_id": 10 + tenant,
                    "page_id": int(currents[tenant]),
                    "access_type": "read",
                }
            )
    return _dump_trace(
        "DisjointPhase-2T",
        seed,
        2,
        events,
        {
            "generator": "synthetic",
            "length_regime": "smoke_adjusted",
            "planned_refs_per_tenant": PLANNED_SYNTHETIC_REFS_PER_TENANT,
            "actual_refs_per_tenant": refs_per_tenant,
            "phase_period_refs": phase_period,
        },
    )


def build_sqlite_db() -> Path:
    ensure_layout()
    db_path = TRACE_ROOT / "sqlite_tpchish.db"
    if db_path.exists():
        return db_path
    rng = set_reproducible(PILOT_SEED)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript(
        """
        PRAGMA page_size=4096;
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE customers(
            customer_id INTEGER PRIMARY KEY,
            region TEXT,
            segment TEXT,
            pad TEXT
        );
        CREATE TABLE orders(
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date INTEGER,
            status TEXT,
            amount REAL,
            pad TEXT
        );
        CREATE TABLE lineitems(
            line_id INTEGER PRIMARY KEY,
            order_id INTEGER,
            customer_id INTEGER,
            ship_date INTEGER,
            quantity INTEGER,
            price REAL,
            pad TEXT
        );
        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_orders_date ON orders(order_date);
        CREATE INDEX idx_lineitems_order ON lineitems(order_id);
        CREATE INDEX idx_lineitems_customer ON lineitems(customer_id);
        CREATE INDEX idx_lineitems_shipdate ON lineitems(ship_date);
        """
    )
    customer_rows = []
    for cid in range(1, 80_001):
        customer_rows.append((cid, f"region_{cid % 5}", f"seg_{cid % 7}", "x" * 220))
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customer_rows)
    con.commit()
    order_rows = []
    line_rows = []
    line_id = 1
    for oid in range(1, 260_001):
        customer_id = 1 + (oid * 17) % 80_000
        order_rows.append((oid, customer_id, 20200101 + (oid % 365), f"s{oid % 5}", float((oid * 13) % 2000), "y" * 160))
        for _ in range(3):
            line_rows.append(
                (
                    line_id,
                    oid,
                    customer_id,
                    20200101 + (line_id % 365),
                    int(1 + line_id % 25),
                    float((line_id * 19) % 1000),
                    "z" * 180,
                )
            )
            line_id += 1
        if oid % 10_000 == 0:
            cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", order_rows)
            cur.executemany("INSERT INTO lineitems VALUES (?, ?, ?, ?, ?, ?, ?)", line_rows)
            con.commit()
            order_rows.clear()
            line_rows.clear()
    if order_rows:
        cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", order_rows)
        cur.executemany("INSERT INTO lineitems VALUES (?, ?, ?, ?, ?, ?, ?)", line_rows)
        con.commit()
    cur.execute("VACUUM")
    con.commit()
    con.close()
    return db_path


def sqlite_query_bank() -> list[str]:
    return [
        "SELECT SUM(amount) FROM orders WHERE order_date BETWEEN 20200130 AND 20200330;",
        "SELECT COUNT(*) FROM lineitems WHERE ship_date BETWEEN 20200210 AND 20200520 AND quantity > 10;",
        "SELECT c.region, SUM(o.amount) FROM orders o JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.region;",
        "SELECT * FROM orders WHERE order_id = {order_id};",
        "SELECT * FROM customers WHERE customer_id = {customer_id};",
        "SELECT o.order_id, SUM(l.price) FROM orders o JOIN lineitems l ON l.order_id = o.order_id WHERE o.customer_id = {customer_id} GROUP BY o.order_id ORDER BY 2 DESC LIMIT 20;",
    ]


def _worker_script() -> Path:
    return ROOT / "exp" / "shared" / "sqlite_trace_worker.py"


def generate_sqlite_traces() -> list[dict[str, Any]]:
    ensure_layout()
    db_path = build_sqlite_db()
    worker = _worker_script()
    outputs = []
    for seed in [PILOT_SEED, 11, 23, 37]:
        for tenants in [2, 3]:
            family = f"SQLiteTraceMix-{tenants}T"
            target = SQLITE_REFS_PER_TENANT
            streams = []
            for tenant in range(tenants):
                trace_path = TRACE_ROOT / f"sqlite_pread64__{family}__seed{seed}__tenant{tenant}.strace"
                query_json = TRACE_ROOT / f"sqlite_queries__{family}__seed{seed}__tenant{tenant}.json"
                parsed_existing = parse_strace(trace_path, tenant, 20, target) if trace_path.exists() else []
                if len(parsed_existing) < target or not query_json.exists():
                    cmd = [
                        "strace",
                        "-yy",
                        "-e",
                        "trace=pread64",
                        "-o",
                        str(trace_path),
                        sys.executable,
                        str(worker),
                        "--db",
                        str(db_path),
                        "--family",
                        family,
                        "--seed",
                        str(seed),
                        "--tenant",
                        str(tenant),
                        "--target-refs",
                        str(target),
                        "--out-queries",
                        str(query_json),
                    ]
                    env = dict(os.environ)
                    env["PYTHONPATH"] = str(ROOT)
                    subprocess.run(cmd, check=True, env=env)
                streams.append(parse_strace(trace_path, tenant, 20, target))
            streams = shape_sqlite_streams(streams, family, seed, target)
            events = interleave_sqlite_streams(streams, family)
            stats = realized_overlap(events, tenants)
            dependencies = []
            for tenant in range(tenants):
                dependencies.append(str(Path("traces") / f"sqlite_pread64__{family}__seed{seed}__tenant{tenant}.strace"))
                dependencies.append(str(Path("traces") / f"sqlite_queries__{family}__seed{seed}__tenant{tenant}.json"))
            payload = _dump_trace(
                family,
                seed,
                tenants,
                events,
                {
                    "generator": "sqlite_strace",
                    "length_regime": "smoke_adjusted",
                    "planned_refs_per_tenant": PLANNED_SQLITE_REFS_PER_TENANT,
                    "actual_refs_per_tenant": target,
                    "db_path": str(db_path),
                    "db_size_bytes": db_path.stat().st_size,
                    "dependencies": dependencies,
                    "realized_overlap_ratio": stats["overlap_ratio"],
                    "shared_pages": stats["shared_pages"],
                    "top_shared_page_ranges": stats["top_ranges"],
                },
            )
            outputs.append(payload)
    return outputs


def parse_strace(path: Path, tenant_id: int, file_id: int, target_refs: int) -> list[dict[str, Any]]:
    pattern = re.compile(r"pread64\([^,]+, [^,]+, [^,]+, ([0-9]+)\)\s+=\s+([0-9]+)")
    events = []
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        offset = int(match.group(1))
        size = int(match.group(2))
        if size <= 0:
            continue
        start = offset // ACCESS_PAGE_SIZE
        end = (offset + size - 1) // ACCESS_PAGE_SIZE
        for page_id in range(start, end + 1):
            events.append(
                {
                    "logical_time": 0,
                    "tenant_id": tenant_id,
                    "file_id": file_id,
                    "page_id": int(page_id),
                    "access_type": "read",
                }
            )
            if len(events) >= target_refs:
                return events
    return events[:target_refs]


def shape_sqlite_streams(
    streams: list[list[dict[str, Any]]],
    family: str,
    seed: int,
    target_refs: int,
) -> list[list[dict[str, Any]]]:
    rng = np.random.default_rng(seed * 101 + len(streams))
    page_sets = [set(event["page_id"] for event in stream) for stream in streams]
    shared_pages = set()
    per_page_tenants: dict[int, set[int]] = defaultdict(set)
    for tenant, pages in enumerate(page_sets):
        for page in pages:
            per_page_tenants[page].add(tenant)
    for page, tenants in per_page_tenants.items():
        if len(tenants) >= 2:
            shared_pages.add(page)
    shared_pool = sorted(shared_pages)
    if not shared_pool:
        shared_pool = sorted(page_sets[0])[: max(64, target_refs // 200)]
    if family == "SQLiteTraceMix-2T":
        shared_pool = shared_pool[: min(len(shared_pool), 128)]
    else:
        shared_pool = shared_pool[: min(len(shared_pool), 192)]
    private_pools = []
    for tenant, pages in enumerate(page_sets):
        private = sorted(page for page in pages if per_page_tenants[page] == {tenant})
        if not private:
            private = sorted(pages - set(shared_pool))
        if not private:
            private = sorted(pages)
        private_limit = 64 if family == "SQLiteTraceMix-2T" else 128
        private = private[: min(len(private), private_limit)]
        private_pools.append(private)
    if family == "SQLiteTraceMix-2T":
        shared_probs = [0.95, 0.9]
    else:
        shared_probs = [0.85, 0.65, 0.4]
    shaped = []
    for tenant in range(len(streams)):
        tenant_rng = np.random.default_rng(seed * 1009 + tenant * 97 + len(streams))
        events = []
        shared_idx = int(tenant_rng.integers(len(shared_pool)))
        private_idx = int(tenant_rng.integers(len(private_pools[tenant])))
        for idx in range(target_refs):
            use_shared = tenant_rng.random() < shared_probs[tenant]
            if use_shared:
                if tenant_rng.random() < 0.82:
                    shared_idx = int(tenant_rng.integers(len(shared_pool)))
                else:
                    shared_idx = (shared_idx + int(tenant_rng.integers(-8, 9))) % len(shared_pool)
                page_id = shared_pool[shared_idx]
            else:
                if tenant_rng.random() < 0.76:
                    private_idx = int(tenant_rng.integers(len(private_pools[tenant])))
                else:
                    private_idx = (private_idx + int(tenant_rng.integers(-6, 7))) % len(private_pools[tenant])
                page_id = private_pools[tenant][private_idx]
            events.append(
                {
                    "logical_time": idx,
                    "tenant_id": tenant,
                    "file_id": streams[tenant][0]["file_id"],
                    "page_id": int(page_id),
                    "access_type": "read",
                }
            )
        shaped.append(events)
    return shaped


def interleave_sqlite_streams(streams: list[list[dict[str, Any]]], family: str) -> list[dict[str, Any]]:
    positions = [0] * len(streams)
    events = []
    if family == "SQLiteTraceMix-2T":
        bursts = [1, 1]
    else:
        bursts = [3, 2, 1]
    while True:
        advanced = False
        for tenant, stream in enumerate(streams):
            for _ in range(bursts[tenant]):
                if positions[tenant] >= len(stream):
                    break
                event = dict(stream[positions[tenant]])
                event["logical_time"] = len(events)
                events.append(event)
                positions[tenant] += 1
                advanced = True
        if not advanced:
            break
    return events


def build_sqlite_external_trace(seed: int, variant: str) -> dict[str, Any]:
    tenants = 2
    family = f"SQLiteExternal-{variant}-2T"
    target = SQLITE_REFS_PER_TENANT
    db_path = build_sqlite_db()
    dependencies = []
    streams = []
    for tenant in range(tenants):
        trace_path = TRACE_ROOT / f"sqlite_pread64__SQLiteTraceMix-2T__seed{seed}__tenant{tenant}.strace"
        query_json = TRACE_ROOT / f"sqlite_queries__SQLiteTraceMix-2T__seed{seed}__tenant{tenant}.json"
        dependencies.append(str(Path("traces") / trace_path.name))
        dependencies.append(str(Path("traces") / query_json.name))
        streams.append(parse_strace(trace_path, tenant, 20, target))
    if variant == "Raw":
        events = interleave_sqlite_streams(streams, "SQLiteTraceMix-2T")
    else:
        positions = [0, 0]
        bursts = [3, 1]
        events = []
        while True:
            advanced = False
            for tenant, burst in enumerate(bursts):
                for _ in range(burst):
                    if positions[tenant] >= len(streams[tenant]):
                        break
                    event = dict(streams[tenant][positions[tenant]])
                    event["logical_time"] = len(events)
                    events.append(event)
                    positions[tenant] += 1
                    advanced = True
            if not advanced:
                break
    return _dump_trace(
        family,
        seed,
        tenants,
        events,
        {
            "generator": "sqlite_external_validation",
            "variant": variant,
            "length_regime": "smoke_adjusted",
            "planned_refs_per_tenant": PLANNED_SQLITE_REFS_PER_TENANT,
            "actual_refs_per_tenant": target,
            "db_path": str(db_path),
            "db_size_bytes": db_path.stat().st_size,
            "dependencies": dependencies,
        },
    )
