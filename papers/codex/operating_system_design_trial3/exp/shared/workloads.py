from __future__ import annotations

import math
import os
import random
import sqlite3
import time
import ctypes
import mmap
from pathlib import Path

import pandas as pd

from .utils import PAGE_SIZE, pages_to_gb


def _write_file(path: Path, size_bytes: int, seed: int) -> None:
    if path.exists() and path.stat().st_size == size_bytes:
        return
    rng = random.Random(seed)
    block = bytearray(PAGE_SIZE * 16)
    with path.open("wb") as fh:
        written = 0
        while written < size_bytes:
            for idx in range(len(block)):
                block[idx] = rng.randrange(256)
            chunk = bytes(block[: min(len(block), size_bytes - written)])
            fh.write(chunk)
            written += len(chunk)


libc = ctypes.CDLL(None, use_errno=True)


class ResidencyProbe:
    def __init__(self) -> None:
        self._mappings: dict[Path, tuple[int, mmap.mmap, int, int]] = {}
        self._last_state: dict[tuple[int, int], bool] = {}
        self._last_vmstat = self._vmstat_snapshot()
        self._last_sample_ns = time.perf_counter_ns()
        self._reclaim_epoch = 0
        self._access_counter = 0
        self._dirty_changed = False

    def close(self) -> None:
        for fd, mm, _, _ in self._mappings.values():
            mm.close()
            os.close(fd)
        self._mappings.clear()

    def reset_file_cache(self, path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def _map(self, path: Path) -> tuple[int, mmap.mmap, int, int]:
        if path not in self._mappings:
            fd = os.open(path, os.O_RDONLY)
            size = os.path.getsize(path)
            if size <= 0:
                size = PAGE_SIZE
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_COPY)
            base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            self._mappings[path] = (fd, mm, size, base)
        return self._mappings[path]

    def _resident(self, path: Path, page_index: int) -> bool:
        _, _, size, base = self._map(path)
        offset = int(page_index) * PAGE_SIZE
        if offset >= size:
            return False
        vec = (ctypes.c_ubyte * 1)()
        ret = libc.mincore(ctypes.c_void_p(base + offset), ctypes.c_size_t(PAGE_SIZE), vec)
        if ret != 0:
            return False
        return bool(vec[0] & 1)

    def _vmstat_snapshot(self) -> dict[str, int]:
        fields = {"pgscan_direct": 0, "pgscan_kswapd": 0, "pgsteal_direct": 0, "pgsteal_kswapd": 0, "nr_dirty": 0, "nr_writeback": 0}
        try:
            with Path("/proc/vmstat").open("r", encoding="utf-8") as fh:
                for line in fh:
                    name, value = line.split()[:2]
                    if name in fields:
                        fields[name] = int(value)
        except Exception:
            pass
        return fields

    def _sample_vm(self) -> tuple[int, bool]:
        self._access_counter += 1
        now_ns = time.perf_counter_ns()
        if self._access_counter % 128 != 0 and now_ns - self._last_sample_ns < 100_000_000:
            return self._reclaim_epoch, self._dirty_changed
        current = self._vmstat_snapshot()
        dirty_changed = current["nr_dirty"] > self._last_vmstat["nr_dirty"] or current["nr_writeback"] > self._last_vmstat["nr_writeback"]
        reclaim_total = current["pgscan_direct"] + current["pgscan_kswapd"] + current["pgsteal_direct"] + current["pgsteal_kswapd"]
        prev_total = (
            self._last_vmstat["pgscan_direct"]
            + self._last_vmstat["pgscan_kswapd"]
            + self._last_vmstat["pgsteal_direct"]
            + self._last_vmstat["pgsteal_kswapd"]
        )
        if reclaim_total > prev_total or now_ns - self._last_sample_ns >= 100_000_000:
            self._reclaim_epoch += 1
        self._last_vmstat = current
        self._last_sample_ns = now_ns
        self._dirty_changed = dirty_changed
        return self._reclaim_epoch, self._dirty_changed

    def resident(self, path: Path, page_index: int) -> bool:
        return self._resident(path, page_index)

    def observe_transition(
        self,
        path: Path,
        inode_id: int,
        page_index: int,
        op_class: str,
        phase_id: int,
        t0_ns: int,
        pre_resident: bool,
        post_resident: bool,
    ) -> dict:
        reclaim_epoch, dirty_changed = self._sample_vm()
        page_key = (int(inode_id), int(page_index))
        previously_resident = self._last_state.get(page_key, pre_resident)
        self._last_state[page_key] = post_resident
        return {
            "event_time_us": (time.perf_counter_ns() - t0_ns) // 1000,
            "inode_id": int(inode_id),
            "page_index": int(page_index),
            "op_class": op_class,
            "phase_id": int(phase_id),
            "cache_insert_seen": int((not pre_resident) and post_resident),
            "cache_evict_seen": int(previously_resident and (not pre_resident)),
            "dirty_or_writeback_seen": int(op_class.endswith("write") or dirty_changed),
            "reclaim_epoch": int(reclaim_epoch),
        }


def _finalize(rows: list[dict], workload: str, family: str, seed: int) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"workload {workload} produced no rows")
    df = df.reset_index(drop=True)
    df["logical_seq"] = range(len(df))
    df["phase_id"] = df["phase_id"].astype(int)
    df["seed"] = seed
    df["workload"] = workload
    df["workload_family"] = family
    ordered = [
        "logical_seq",
        "event_time_us",
        "inode_id",
        "page_index",
        "op_class",
        "phase_id",
        "cache_insert_seen",
        "cache_evict_seen",
        "dirty_or_writeback_seen",
        "reclaim_epoch",
        "seed",
        "workload",
        "workload_family",
    ]
    return df[ordered]


def generate_stream_scan(seed: int, dataset_root: Path) -> pd.DataFrame:
    rng = random.Random(seed)
    workload_dir = dataset_root / "stream_scan" / f"seed_{seed}"
    workload_dir.mkdir(parents=True, exist_ok=True)
    probe = ResidencyProbe()
    dir_ids = list(range(64))
    file_paths: list[Path] = []
    for dir_id in dir_ids:
        target_dir = workload_dir / f"dir_{dir_id:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for file_idx in range(4):
            size_bytes = rng.randint(128, 384) * 1024
            path = target_dir / f"file_{file_idx:02d}.bin"
            _write_file(path, size_bytes, seed * 1000 + dir_id * 10 + file_idx)
            probe.reset_file_cache(path)
            file_paths.append(path)

    rng.shuffle(dir_ids)
    by_dir = {dir_id: sorted((workload_dir / f"dir_{dir_id:02d}").glob("*.bin")) for dir_id in range(64)}
    hot_files = rng.sample(file_paths, 24)
    rows = []
    t0 = time.perf_counter_ns()
    try:
        for phase, dir_id in enumerate(dir_ids):
            for path in by_dir[dir_id]:
                st = path.stat()
                with path.open("rb") as fh:
                    page_index = 0
                    while True:
                        pre_resident = probe.resident(path, page_index)
                        if not fh.read(PAGE_SIZE):
                            break
                        rows.append(
                            probe.observe_transition(
                                path,
                                int(st.st_ino),
                                page_index,
                                "scan_read",
                                phase,
                                t0,
                                pre_resident,
                                probe.resident(path, page_index),
                            )
                        )
                        page_index += 1
            if phase % 8 == 0:
                for hot_path in hot_files[:8]:
                    st = hot_path.stat()
                    with hot_path.open("rb") as fh:
                        for page_index in range(8):
                            pre_resident = probe.resident(hot_path, page_index)
                            if not fh.read(PAGE_SIZE):
                                break
                            rows.append(
                                probe.observe_transition(
                                    hot_path,
                                    int(st.st_ino),
                                    page_index,
                                    "scan_read",
                                    100 + phase,
                                    t0,
                                    pre_resident,
                                    probe.resident(hot_path, page_index),
                                )
                            )
    finally:
        probe.close()
    return _finalize(rows, "stream_scan", "stream_scan", seed)


def generate_sqlite_zipf(seed: int, dataset_root: Path) -> pd.DataFrame:
    rng = random.Random(seed)
    workload_dir = dataset_root / "sqlite_zipf" / f"seed_{seed}"
    workload_dir.mkdir(parents=True, exist_ok=True)
    db_path = workload_dir / "kv.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA page_size=4096")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE kv(id INTEGER PRIMARY KEY, payload BLOB NOT NULL)")
    payload = b"x" * 768
    total_rows = 120_000
    batch = [(idx, payload) for idx in range(1, total_rows + 1)]
    conn.executemany("INSERT INTO kv(id, payload) VALUES(?, ?)", batch)
    conn.commit()
    probe = ResidencyProbe()
    probe.reset_file_cache(db_path)

    rows = []
    t0 = time.perf_counter_ns()
    rows_per_page = max(1, PAGE_SIZE // len(payload))
    hot_keys = 16_000
    try:
        for op_idx in range(80_000):
            if rng.random() < 0.90:
                key = 1 + int(min(hot_keys - 1, rng.paretovariate(1.25) * 9)) % hot_keys
            else:
                key = rng.randint(1, total_rows)
            phase_id = 0 if op_idx < 25_000 else (1 if op_idx < 55_000 else 2)
            logical_page = int((key - 1) // rows_per_page)
            logical_page = min(logical_page, max(0, math.ceil(db_path.stat().st_size / PAGE_SIZE) - 1))
            pre_resident = probe.resident(db_path, logical_page)
            if rng.random() < 0.05:
                new_payload = rng.randbytes(1024)
                conn.execute("UPDATE kv SET payload = ? WHERE id = ?", (new_payload, key))
                op_class = "sqlite_write"
            else:
                conn.execute("SELECT payload FROM kv WHERE id = ?", (key,)).fetchone()
                op_class = "sqlite_read"
            if op_idx % 2000 == 0:
                conn.commit()
            rows.append(
                probe.observe_transition(
                    db_path,
                    int(db_path.stat().st_ino),
                    logical_page,
                    op_class,
                    phase_id,
                    t0,
                    pre_resident,
                    probe.resident(db_path, logical_page),
                )
            )
    finally:
        conn.commit()
        conn.close()
        probe.close()
    return _finalize(rows, "sqlite_zipf", "sqlite_zipf", seed)


def generate_filebench_fileserver(seed: int, dataset_root: Path) -> pd.DataFrame:
    rng = random.Random(seed)
    workload_dir = dataset_root / "filebench_fileserver" / f"seed_{seed}"
    workload_dir.mkdir(parents=True, exist_ok=True)
    probe = ResidencyProbe()
    files: list[Path] = []
    for dir_id in range(16):
        target_dir = workload_dir / f"dir_{dir_id:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for file_idx in range(32):
            path = target_dir / f"file_{file_idx:03d}.bin"
            size_bytes = rng.randint(48, 192) * 1024
            _write_file(path, size_bytes, seed * 2000 + dir_id * 100 + file_idx)
            probe.reset_file_cache(path)
            files.append(path)

    rows = []
    t0 = time.perf_counter_ns()
    live_files = list(files)
    try:
        for op_idx in range(36_000):
            phase_id = 0 if op_idx < 12_000 else (1 if op_idx < 24_000 else 2)
            path = rng.choice(live_files)
            st = path.stat()
            max_pages = max(1, math.ceil(st.st_size / PAGE_SIZE))
            op_roll = rng.random()
            if op_roll < 0.78:
                page_index = rng.randrange(max_pages)
                pre_resident = probe.resident(path, page_index)
                with path.open("rb") as fh:
                    fh.seek(page_index * PAGE_SIZE)
                    fh.read(PAGE_SIZE)
                op_class = "fileserver_read"
            elif op_roll < 0.95:
                page_index = rng.randrange(max_pages)
                pre_resident = probe.resident(path, page_index)
                with path.open("r+b") as fh:
                    fh.seek(page_index * PAGE_SIZE)
                    fh.write(rng.randbytes(PAGE_SIZE))
                    fh.flush()
                op_class = "fileserver_write"
            else:
                page_index = 0
                pre_resident = probe.resident(path, page_index)
                if rng.random() < 0.5 and len(live_files) > 128:
                    victim = rng.choice(live_files)
                    os.remove(victim)
                    live_files.remove(victim)
                else:
                    new_path = workload_dir / f"dynamic_{op_idx:05d}.bin"
                    _write_file(new_path, PAGE_SIZE * rng.randint(8, 40), seed + op_idx)
                    probe.reset_file_cache(new_path)
                    live_files.append(new_path)
                os.listdir(path.parent)
                op_class = "fileserver_meta"
            rows.append(
                probe.observe_transition(
                    path,
                    int(st.st_ino),
                    page_index,
                    op_class,
                    phase_id,
                    t0,
                    pre_resident,
                    probe.resident(path, page_index),
                )
            )
    finally:
        probe.close()
    return _finalize(rows, "filebench_fileserver", "filebench_fileserver", seed)


def generate_workload(workload: str, seed: int, dataset_root: Path) -> pd.DataFrame:
    if workload == "stream_scan":
        return generate_stream_scan(seed, dataset_root)
    if workload == "sqlite_zipf":
        return generate_sqlite_zipf(seed, dataset_root)
    if workload == "filebench_fileserver":
        return generate_filebench_fileserver(seed, dataset_root)
    raise ValueError(f"unknown workload {workload}")


def trace_modes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base_cols = ["logical_seq", "event_time_us", "inode_id", "page_index", "op_class", "phase_id", "seed", "workload", "workload_family"]
    access = df[base_cols].copy()
    nodirty = access.copy()
    nodirty["reclaim_epoch"] = df["reclaim_epoch"]
    compact = nodirty.copy()
    compact["dirty_or_writeback_seen"] = df["dirty_or_writeback_seen"]
    extended = compact.copy()
    extended["cache_insert_seen"] = df["cache_insert_seen"]
    extended["cache_evict_seen"] = df["cache_evict_seen"]
    noreclaim = compact.drop(columns=["reclaim_epoch"]).copy()
    return {
        "AccessOnly": access,
        "NoDirty": nodirty,
        "CompactState": compact,
        "ExtendedHinted": extended,
        "NoReclaim": noreclaim,
    }


def workload_stats(df: pd.DataFrame) -> dict[str, float | int | str]:
    page_keys = list(zip(df["inode_id"], df["page_index"]))
    unique_pages = len(set(page_keys))
    read_ops = int(df["op_class"].str.contains("read").sum())
    write_ops = int(df["op_class"].str.contains("write").sum())
    total = len(df)
    window = 2048
    reuse = []
    last_seen = {}
    for idx, key in enumerate(page_keys):
        if key in last_seen:
            reuse.append(idx - last_seen[key])
        last_seen[key] = idx
    wss = []
    for start in range(0, total, window):
        wss.append(len(set(page_keys[start : start + window])))
    return {
        "workload": str(df["workload"].iloc[0]),
        "workload_family": str(df["workload_family"].iloc[0]),
        "seed": int(df["seed"].iloc[0]),
        "total_accesses": int(total),
        "unique_pages": int(unique_pages),
        "read_write_mix": float(read_ops / max(1, read_ops + write_ops)),
        "working_set_estimate_pages": int(max(1, math.ceil(sum(wss) / max(1, len(wss))))),
        "working_set_estimate_gb": float(pages_to_gb(sum(wss) / max(1, len(wss)))),
        "mean_reuse_distance_pages": float(sum(reuse) / max(1, len(reuse))),
        "logical_dataset_gb": float(pages_to_gb(unique_pages)),
    }
