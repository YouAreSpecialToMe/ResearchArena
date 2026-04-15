from __future__ import annotations

import collections
from dataclasses import dataclass, field


POLICY_HYPERPARAMS = {
    "FIFO": {},
    "CLOCK": {
        "dirty_penalty": 2,
        "reclaim_bonus_cap": 4,
        "insert_protect_window": 32,
    },
    "LFU": {
        "reclaim_decay": 0.35,
        "dirty_bonus": 0.75,
        "insert_bonus": 0.5,
    },
    "S3FIFO": {
        "small_queue_fraction": 0.25,
        "ghost_fraction": 0.10,
        "dirty_promotion_hits": 2,
        "reclaim_promotion_gap": 2,
    },
    "Hyperbolic": {
        "score_age_scale": 256.0,
        "dirty_bonus": 1.5,
        "reclaim_bonus": 0.6,
        "insert_bonus": 0.4,
    },
    "LinuxDefault": {
        "scan_bonus": 200.0,
        "dirty_penalty": 120.0,
        "reclaim_bonus": 30.0,
        "insert_bonus": 25.0,
        "evict_bonus": 12.0,
    },
}


def _uses_dirty(mode: str) -> bool:
    return mode in {"ExtendedHinted", "CompactState"}


def _uses_reclaim(mode: str) -> bool:
    return mode in {"ExtendedHinted", "CompactState", "NoDirty"}


def _uses_insert_hints(mode: str) -> bool:
    return mode == "ExtendedHinted"


@dataclass
class PageEntry:
    page: tuple[int, int]
    insert_seq: int
    last_access_seq: int
    hits: int = 0
    dirty: bool = False
    ref: int = 1
    op_class: str = ""
    reclaim_epoch: int = 0
    tier: str = "small"
    insert_hinted: bool = False


@dataclass
class PolicyState:
    name: str
    capacity: int
    mode: str
    store: dict[tuple[int, int], PageEntry] = field(default_factory=dict)
    queue: collections.deque = field(default_factory=collections.deque)
    main: collections.deque = field(default_factory=collections.deque)
    small: collections.deque = field(default_factory=collections.deque)
    ghost: collections.deque = field(default_factory=collections.deque)
    clock: list[tuple[int, int]] = field(default_factory=list)
    clock_hand: int = 0
    active: collections.deque = field(default_factory=collections.deque)
    inactive: collections.deque = field(default_factory=collections.deque)
    score_age_scale: float = 256.0

    def touch(self, page: tuple[int, int], seq: int, op_class: str, dirty: bool, reclaim_epoch: int) -> None:
        entry = self.store[page]
        entry.last_access_seq = seq
        entry.hits += 1
        entry.ref = 1
        entry.op_class = op_class
        entry.dirty = entry.dirty or dirty
        entry.reclaim_epoch = reclaim_epoch
        if self.name == "LFU":
            return
        if self.name == "FIFO":
            return
        if self.name == "CLOCK":
            return
        if self.name == "S3FIFO":
            params = POLICY_HYPERPARAMS["S3FIFO"]
            reclaim_gap = max(0, reclaim_epoch - entry.reclaim_epoch)
            if entry.hits >= params["dirty_promotion_hits"] or (entry.dirty and _uses_dirty(self.mode)) or reclaim_gap >= params["reclaim_promotion_gap"]:
                self.store[page].tier = "main"
            return
        if self.name == "LinuxDefault":
            return

    def insert(self, entry: PageEntry) -> None:
        page = entry.page
        self.store[page] = entry
        if self.name == "S3FIFO":
            small_target = max(1, int(self.capacity * POLICY_HYPERPARAMS["S3FIFO"]["small_queue_fraction"]))
            entry.tier = "main" if page in self.ghost or len([e for e in self.store.values() if e.tier == "small"]) > small_target else "small"

    def choose_victim(self, seq: int, access_meta: dict) -> tuple[int, int]:
        if self.name == "FIFO":
            return min(self.store, key=lambda p: self.store[p].insert_seq)
        elif self.name == "CLOCK":
            params = POLICY_HYPERPARAMS["CLOCK"]
            use_dirty = _uses_dirty(self.mode)
            use_reclaim = _uses_reclaim(self.mode)
            use_insert = _uses_insert_hints(self.mode)
            candidates = []
            current_reclaim = access_meta.get("reclaim_epoch", 0)
            for page, entry in self.store.items():
                reclaim_gap = min(params["reclaim_bonus_cap"], max(0, current_reclaim - entry.reclaim_epoch)) if use_reclaim else 0
                insert_protect = int(use_insert and entry.insert_hinted and (seq - entry.insert_seq) <= params["insert_protect_window"])
                dirty_rank = int(use_dirty and entry.dirty)
                candidates.append((entry.ref, dirty_rank, -reclaim_gap, insert_protect, entry.last_access_seq, page))
                entry.ref = 0
            candidates.sort(key=lambda item: item[:-1])
            return candidates[0][-1]
        elif self.name == "LFU":
            params = POLICY_HYPERPARAMS["LFU"]
            use_dirty = _uses_dirty(self.mode)
            use_reclaim = _uses_reclaim(self.mode)
            use_insert = _uses_insert_hints(self.mode)
            current_reclaim = access_meta.get("reclaim_epoch", 0)
            return min(
                self.store,
                key=lambda p: (
                    self.store[p].hits
                    + (params["dirty_bonus"] if use_dirty and self.store[p].dirty else 0.0)
                    + (params["insert_bonus"] if use_insert and self.store[p].insert_hinted else 0.0)
                    - (params["reclaim_decay"] * max(0, current_reclaim - self.store[p].reclaim_epoch) if use_reclaim else 0.0),
                    self.store[p].last_access_seq,
                    self.store[p].insert_seq,
                ),
            )
        elif self.name == "S3FIFO":
            params = POLICY_HYPERPARAMS["S3FIFO"]
            use_dirty = _uses_dirty(self.mode)
            use_reclaim = _uses_reclaim(self.mode)
            small_pages = [p for p, e in self.store.items() if e.tier == "small"]
            if small_pages:
                victim = min(
                    small_pages,
                    key=lambda p: (
                        self.store[p].hits + (4 if use_dirty and self.store[p].dirty else 0),
                        -(max(0, access_meta.get("reclaim_epoch", 0) - self.store[p].reclaim_epoch) if use_reclaim else 0),
                        self.store[p].insert_seq,
                    ),
                )
            else:
                victim = min(
                    self.store,
                    key=lambda p: (
                        self.store[p].hits + (2 if use_dirty and self.store[p].dirty else 0),
                        -(max(0, access_meta.get("reclaim_epoch", 0) - self.store[p].reclaim_epoch) if use_reclaim else 0),
                        self.store[p].insert_seq,
                    ),
                )
            self.ghost.append(victim)
            while len(self.ghost) > max(1, int(self.capacity * params["ghost_fraction"])):
                self.ghost.popleft()
            return victim
        elif self.name == "Hyperbolic":
            params = POLICY_HYPERPARAMS["Hyperbolic"]
            use_dirty = _uses_dirty(self.mode)
            use_reclaim = _uses_reclaim(self.mode)
            use_insert = _uses_insert_hints(self.mode)
            return min(
                self.store,
                key=lambda p: (
                    (
                        self.store[p].hits
                        + (params["dirty_bonus"] if use_dirty and self.store[p].dirty else 0.0)
                        + (params["reclaim_bonus"] * max(0, access_meta.get("reclaim_epoch", 0) - self.store[p].reclaim_epoch) if use_reclaim else 0.0)
                        + (params["insert_bonus"] if use_insert and self.store[p].insert_hinted else 0.0)
                    )
                    / max(1.0, (seq - self.store[p].insert_seq) / self.score_age_scale),
                    self.store[p].last_access_seq,
                ),
            )
        elif self.name == "LinuxDefault":
            mode = self.mode
            params = POLICY_HYPERPARAMS["LinuxDefault"]
            use_dirty = _uses_dirty(mode)
            use_reclaim = _uses_reclaim(mode)
            candidates = list(self.store)

            def score(page: tuple[int, int]) -> tuple[float, int]:
                entry = self.store[page]
                age = seq - entry.last_access_seq
                penalty = 0.0
                if "scan" in entry.op_class:
                    penalty -= params["scan_bonus"]
                if use_dirty and entry.dirty:
                    penalty += params["dirty_penalty"]
                if use_reclaim:
                    penalty -= params["reclaim_bonus"] * max(0, access_meta.get("reclaim_epoch", 0) - entry.reclaim_epoch)
                if _uses_insert_hints(mode) and entry.insert_hinted:
                    penalty -= params["insert_bonus"]
                if _uses_insert_hints(mode) and access_meta.get("cache_evict_seen", 0):
                    penalty -= params["evict_bonus"]
                return (age - penalty, entry.last_access_seq)

            return max(candidates, key=score)
        return next(iter(self.store))

    def remove(self, page: tuple[int, int]) -> PageEntry:
        return self.store.pop(page)


def make_policy(name: str, capacity: int, mode: str) -> PolicyState:
    params = POLICY_HYPERPARAMS.get(name, {})
    return PolicyState(
        name=name,
        capacity=max(1, capacity),
        mode=mode,
        score_age_scale=float(params.get("score_age_scale", 256.0)),
    )
