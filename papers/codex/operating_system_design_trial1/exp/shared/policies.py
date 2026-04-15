from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from .utils import EPOCH_LENGTH, make_rng


class BasePolicy:
    name = "Base"
    metadata_bytes_per_entry = 24

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.seed = seed

    def access(self, page: int, t: int) -> bool:
        raise NotImplementedError

    def on_epoch_end(self, epoch: int) -> None:
        return None

    def metadata_bytes(self) -> int:
        return self.capacity * self.metadata_bytes_per_entry

    def snapshot_resident_pages(self) -> list[int]:
        raise NotImplementedError

    def load_resident_pages(self, pages: list[int], now: int) -> None:
        raise NotImplementedError


class LRUPolicy(BasePolicy):
    name = "LRU"
    metadata_bytes_per_entry = 20

    def __init__(self, capacity: int, seed: int = 0):
        super().__init__(capacity, seed)
        self.cache: OrderedDict[int, None] = OrderedDict()

    def access(self, page: int, t: int) -> bool:
        if page in self.cache:
            self.cache.move_to_end(page)
            return True
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[page] = None
        return False

    def snapshot_resident_pages(self) -> list[int]:
        return list(self.cache.keys())

    def load_resident_pages(self, pages: list[int], now: int) -> None:
        self.cache = OrderedDict((page, None) for page in pages[-self.capacity :])


class MRUPolicy(LRUPolicy):
    name = "MRU"

    def access(self, page: int, t: int) -> bool:
        if page in self.cache:
            self.cache.move_to_end(page)
            return True
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=True)
        self.cache[page] = None
        return False


class LFUAgingPolicy(BasePolicy):
    name = "LFU-aging"
    metadata_bytes_per_entry = 32

    def __init__(self, capacity: int, seed: int = 0):
        super().__init__(capacity, seed)
        self.page_to_freq: dict[int, int] = {}
        self.freq_to_pages: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
        self.min_freq = 1

    def _rebuild(self) -> None:
        rebuilt: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
        for page, freq in self.page_to_freq.items():
            rebuilt[freq][page] = None
        self.freq_to_pages = rebuilt
        self.min_freq = min(self.freq_to_pages) if self.freq_to_pages else 1

    def _bump(self, page: int) -> None:
        freq = self.page_to_freq[page]
        bucket = self.freq_to_pages[freq]
        bucket.pop(page, None)
        if not bucket:
            self.freq_to_pages.pop(freq, None)
            if self.min_freq == freq:
                self.min_freq += 1
        new_freq = min(255, freq + 1)
        self.page_to_freq[page] = new_freq
        self.freq_to_pages[new_freq][page] = None

    def access(self, page: int, t: int) -> bool:
        if page in self.page_to_freq:
            self._bump(page)
            return True
        if len(self.page_to_freq) >= self.capacity:
            bucket = self.freq_to_pages[self.min_freq]
            victim, _ = bucket.popitem(last=False)
            self.page_to_freq.pop(victim, None)
            if not bucket:
                self.freq_to_pages.pop(self.min_freq, None)
        self.page_to_freq[page] = 1
        self.freq_to_pages[1][page] = None
        self.min_freq = 1
        return False

    def on_epoch_end(self, epoch: int) -> None:
        if (epoch + 1) % 4 != 0:
            return
        for page, freq in list(self.page_to_freq.items()):
            self.page_to_freq[page] = max(1, freq // 2)
        self._rebuild()

    def snapshot_resident_pages(self) -> list[int]:
        return list(self.page_to_freq.keys())

    def load_resident_pages(self, pages: list[int], now: int) -> None:
        self.page_to_freq = {}
        self.freq_to_pages = defaultdict(OrderedDict)
        for page in pages[-self.capacity :]:
            self.page_to_freq[page] = 1
            self.freq_to_pages[1][page] = None
        self.min_freq = 1


class RandomSampleSet:
    def __init__(self):
        self.items: list[int] = []
        self.pos: dict[int, int] = {}

    def add(self, page: int) -> None:
        self.pos[page] = len(self.items)
        self.items.append(page)

    def remove(self, page: int) -> None:
        idx = self.pos.pop(page)
        last = self.items.pop()
        if idx < len(self.items):
            self.items[idx] = last
            self.pos[last] = idx

    def sample(self, rng, k: int) -> list[int]:
        if not self.items:
            return []
        if len(self.items) <= k:
            return list(self.items)
        return rng.sample(self.items, k)


class LHDPolicy(BasePolicy):
    name = "LHD"
    metadata_bytes_per_entry = 36

    def __init__(self, capacity: int, seed: int = 0):
        super().__init__(capacity, seed)
        self.last_access: dict[int, int] = {}
        self.hits: dict[int, int] = {}
        self.entries = RandomSampleSet()
        self.rng = make_rng("lhd", seed, capacity)

    def access(self, page: int, t: int) -> bool:
        if page in self.last_access:
            self.hits[page] = min(255, self.hits[page] + 1)
            self.last_access[page] = t
            return True
        if len(self.last_access) >= self.capacity:
            candidates = self.entries.sample(self.rng, 32)
            victim = min(candidates, key=lambda p: self._density(p, t))
            self.entries.remove(victim)
            self.last_access.pop(victim, None)
            self.hits.pop(victim, None)
        self.last_access[page] = t
        self.hits[page] = 1
        self.entries.add(page)
        return False

    def _density(self, page: int, now: int) -> float:
        age = max(1, now - self.last_access[page])
        age_bucket = min(15, age // max(1, EPOCH_LENGTH // 16))
        size_bucket = 0
        return self.hits[page] / float((age_bucket + 1) * (size_bucket + 1))

    def snapshot_resident_pages(self) -> list[int]:
        return list(self.last_access.keys())

    def load_resident_pages(self, pages: list[int], now: int) -> None:
        self.last_access = {}
        self.hits = {}
        self.entries = RandomSampleSet()
        for page in pages[-self.capacity :]:
            self.last_access[page] = now
            self.hits[page] = 1
            self.entries.add(page)


class ARCPolicy(BasePolicy):
    name = "ARC"
    metadata_bytes_per_entry = 40

    def __init__(self, capacity: int, seed: int = 0):
        super().__init__(capacity, seed)
        self.p = 0
        self.t1: OrderedDict[int, None] = OrderedDict()
        self.t2: OrderedDict[int, None] = OrderedDict()
        self.b1: OrderedDict[int, None] = OrderedDict()
        self.b2: OrderedDict[int, None] = OrderedDict()

    def _replace(self, page: int) -> None:
        if self.t1 and ((page in self.b2 and len(self.t1) == self.p) or len(self.t1) > self.p):
            old, _ = self.t1.popitem(last=False)
            self.b1[old] = None
        else:
            old, _ = self.t2.popitem(last=False)
            self.b2[old] = None

    def access(self, page: int, t: int) -> bool:
        if page in self.t1:
            self.t1.pop(page, None)
            self.t2[page] = None
            return True
        if page in self.t2:
            self.t2.move_to_end(page)
            return True
        if page in self.b1:
            delta = 1 if len(self.b1) >= len(self.b2) else len(self.b2) / max(1, len(self.b1))
            self.p = min(self.capacity, self.p + delta)
            self._replace(page)
            self.b1.pop(page, None)
            self.t2[page] = None
            return False
        if page in self.b2:
            delta = 1 if len(self.b2) >= len(self.b1) else len(self.b1) / max(1, len(self.b2))
            self.p = max(0, self.p - delta)
            self._replace(page)
            self.b2.pop(page, None)
            self.t2[page] = None
            return False
        total = len(self.t1) + len(self.t2)
        if total >= self.capacity:
            self._replace(page)
        elif total + len(self.b1) + len(self.b2) >= self.capacity:
            if total + len(self.b1) + len(self.b2) >= 2 * self.capacity and self.b2:
                self.b2.popitem(last=False)
        self.t1[page] = None
        return False

    def snapshot_resident_pages(self) -> list[int]:
        return list(self.t1.keys()) + list(self.t2.keys())

    def load_resident_pages(self, pages: list[int], now: int) -> None:
        self.p = 0
        self.t1 = OrderedDict((page, None) for page in pages[-self.capacity :])
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()


POLICY_MAP = {
    "LRU": LRUPolicy,
    "MRU": MRUPolicy,
    "LFU-aging": LFUAgingPolicy,
    "LHD": LHDPolicy,
    "ARC": ARCPolicy,
}


@dataclass
class ReplayStats:
    total_misses: int
    epoch_misses: list[int]
    miss_ratio: float
    metadata_bytes: int


def replay_trace(trace: Iterable[int], policy_name: str, capacity: int, epoch_length: int = EPOCH_LENGTH, seed: int = 0) -> ReplayStats:
    policy = POLICY_MAP[policy_name](capacity=capacity, seed=seed)
    misses = 0
    epoch_misses: list[int] = []
    epoch_count = 0
    epoch_miss = 0
    for t, page in enumerate(trace):
        hit = policy.access(int(page), t)
        if not hit:
            misses += 1
            epoch_miss += 1
        if (t + 1) % epoch_length == 0:
            epoch_misses.append(epoch_miss)
            epoch_miss = 0
            policy.on_epoch_end(epoch_count)
            epoch_count += 1
    if len(trace) % epoch_length:
        epoch_misses.append(epoch_miss)
        policy.on_epoch_end(epoch_count)
    return ReplayStats(
        total_misses=misses,
        epoch_misses=epoch_misses,
        miss_ratio=misses / max(1, len(trace)),
        metadata_bytes=policy.metadata_bytes(),
    )
