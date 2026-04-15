from __future__ import annotations

import time

from .policies import PageEntry, make_policy
from .utils import detect_peak_rss_mb


def simulate_policy(df, policy_name: str, capacity: int, mode: str, run_type: str) -> dict:
    policy = make_policy(policy_name, capacity, mode)
    start = time.perf_counter()
    misses = 0
    hits = 0
    dirty_evictions = 0
    clean_evictions = 0
    writebacks = 0
    eviction_ages = []

    has_dirty = "dirty_or_writeback_seen" in df.columns
    has_reclaim = "reclaim_epoch" in df.columns
    has_insert = "cache_insert_seen" in df.columns
    has_evict = "cache_evict_seen" in df.columns
    has_event_time = "event_time_us" in df.columns
    last_flush_us = 0

    for row in df.itertuples(index=False):
        page = (int(row.inode_id), int(row.page_index))
        op_class = str(row.op_class)
        seq = int(row.logical_seq)
        event_time_us = int(getattr(row, "event_time_us", seq))
        dirty = bool(getattr(row, "dirty_or_writeback_seen", 0)) if has_dirty else op_class.endswith("write")
        access_meta = {
            "reclaim_epoch": int(getattr(row, "reclaim_epoch", 0)) if has_reclaim else 0,
            "cache_insert_seen": int(getattr(row, "cache_insert_seen", 0)) if has_insert else 0,
            "cache_evict_seen": int(getattr(row, "cache_evict_seen", 0)) if has_evict else 0,
        }
        if run_type == "online":
            if has_event_time and event_time_us - last_flush_us >= 25_000:
                for entry in policy.store.values():
                    if entry.dirty and event_time_us - last_flush_us >= 25_000:
                        entry.dirty = False
                last_flush_us = event_time_us
        if page in policy.store:
            hits += 1
            policy.touch(page, seq, op_class, dirty, access_meta["reclaim_epoch"])
            continue

        misses += 1
        if len(policy.store) >= policy.capacity:
            victim = policy.choose_victim(seq, access_meta)
            removed = policy.remove(victim)
            age = seq - removed.insert_seq
            eviction_ages.append(age)
            if removed.dirty:
                dirty_evictions += 1
                writebacks += 1
            else:
                clean_evictions += 1

        entry = PageEntry(
            page=page,
            insert_seq=seq,
            last_access_seq=seq,
            hits=1,
            dirty=dirty,
            ref=1,
            op_class=op_class,
            reclaim_epoch=access_meta["reclaim_epoch"],
            insert_hinted=bool(access_meta["cache_insert_seen"]),
        )
        policy.insert(entry)

    runtime = time.perf_counter() - start
    total = hits + misses
    return {
        "policy": policy_name,
        "trace_mode": mode,
        "run_type": run_type,
        "miss_rate": misses / max(1, total),
        "hit_rate": hits / max(1, total),
        "dirty_eviction_count": dirty_evictions,
        "clean_eviction_count": clean_evictions,
        "writeback_count": writebacks,
        "mean_eviction_age": sum(eviction_ages) / max(1, len(eviction_ages)),
        "replay_runtime_s": runtime,
        "peak_rss_mb": detect_peak_rss_mb(),
    }
