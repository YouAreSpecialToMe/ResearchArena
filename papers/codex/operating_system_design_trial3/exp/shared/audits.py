from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from .utils import read_json


REFERENCE_EXPECTED = {
    "Enoki-High-Velocity-Linux-Kernel-Scheduler-Development": {
        "title": "Enoki: High Velocity Linux Kernel Scheduler Development",
        "venue": "EuroSys",
        "year": "2024",
        "url": "https://dl.acm.org/doi/10.1145/3627703.3629578",
        "authors": ["Vakharia", "Chen"],
    },
    "PageFlex-Flexible-and-Efficient-User-space-Delegation-of-Linux-Paging-Policies-with-eBPF": {
        "title": "PageFlex: Flexible and Efficient User-space Delegation of Linux Paging Policies with eBPF",
        "venue": "USENIX ATC 25",
        "year": "2025",
        "url": "https://www.usenix.org/conference/atc25/presentation/yelam",
        "authors": ["Yelam", "Keeton"],
    },
    "cache_ext-Customizing-the-Page-Cache-with-eBPF": {
        "title": "cache_ext: Customizing the Page Cache with eBPF",
        "venue": "SOSP",
        "year": "2025",
        "url": "https://dl.acm.org/doi/10.1145/3694715.3695969",
        "authors": ["Zussman", "Cidon"],
    },
    "StreamCache-Revisiting-Page-Cache-for-File-Scanning-on-Fast-Storage-Devices": {
        "title": "StreamCache: Revisiting Page Cache for File Scanning on Fast Storage Devices",
        "venue": "USENIX ATC 24",
        "year": "2024",
        "url": "https://www.usenix.org/conference/atc24/presentation/li-zhiyue",
        "authors": ["Li", "Zhang"],
    },
    "Modeling-the-Linux-page-cache-for-accurate-simulation-of-data-intensive-applications": {
        "title": "Modeling the Linux page cache for accurate simulation of data-intensive applications",
        "venue": "CLUSTER",
        "year": "2021",
        "url": "https://ieeexplore.ieee.org/document/9556012",
        "authors": ["Do", "Casanova"],
    },
    "Cache-Modeling-and-Optimization-using-Miniature-Simulations": {
        "title": "Cache Modeling and Optimization using Miniature Simulations",
        "venue": "USENIX ATC 17",
        "year": "2017",
        "url": "https://www.usenix.org/conference/atc17/technical-sessions/presentation/waldspurger",
        "authors": ["Waldspurger", "Ahmad"],
    },
    "Efficient-MRC-Construction-with-SHARDS": {
        "title": "Efficient MRC Construction with SHARDS",
        "venue": "FAST 15",
        "year": "2015",
        "url": "https://www.usenix.org/conference/fast15/technical-sessions/presentation/waldspurger",
        "authors": ["Waldspurger", "Garthwaite"],
    },
    "Kosmo-Efficient-Online-Miss-Ratio-Curve-Generation-for-Eviction-Policy-Evaluation": {
        "title": "Kosmo: Efficient Online Miss Ratio Curve Generation for Eviction Policy Evaluation",
        "venue": "FAST 24",
        "year": "2024",
        "url": "https://www.usenix.org/conference/fast24/presentation/shakiba",
        "authors": ["Shakiba", "Stumm"],
    },
}


def _fetch_official_text(url: str) -> tuple[bool, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return True, response.read().decode("utf-8", errors="ignore")
    except Exception:
        return False, ""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def reference_audit(root: Path) -> list[dict]:
    refs = []
    ref_root = root / "references"
    for key, expected in REFERENCE_EXPECTED.items():
        meta = (ref_root / key / "meta" / "meta_info.txt").read_text(encoding="utf-8", errors="ignore")
        bib = (ref_root / key / "meta" / "bibtex.txt").read_text(encoding="utf-8", errors="ignore")
        haystack = _normalize(f"{meta}\n{bib}")
        fetched, official = _fetch_official_text(expected["url"])
        official_text = _normalize(official)
        refs.append(
            {
                "paper_key": key,
                "title_ok": expected["title"].lower() in haystack and expected["title"].lower() in official_text,
                "authors_ok": all(author.lower() in haystack and author.lower() in official_text for author in expected["authors"]),
                "venue_ok": expected["venue"].lower() in haystack and expected["venue"].lower() in official_text,
                "year_ok": expected["year"] in haystack and expected["year"] in official_text,
                "url_ok": fetched,
                "corrected": key.startswith("Enoki") or key.startswith("StreamCache"),
            }
        )
    return refs


def spec_consistency(root: Path) -> list[dict]:
    idea = read_json(root / "idea.json")
    plan = read_json(root / "plan.json")
    proposal = (root / "proposal.md").read_text(encoding="utf-8")
    idea_text = json.dumps(idea).lower()
    plan_text = json.dumps(plan).lower()
    proposal_text = proposal.lower()
    checks = [
        (
            "workload_families",
            all(name in proposal_text and name in plan_text for name in ["stream_scan", "sqlite_zipf", "filebench_fileserver"])
            and all(term in idea_text for term in ["streaming scan", "sqlite", "mixed file service"]),
        ),
        ("policy_pool", all(name.lower() in proposal_text and name.lower() in idea_text and name.lower() in plan_text for name in ["LinuxDefault", "FIFO", "CLOCK", "LFU", "S3FIFO", "Hyperbolic"])),
        ("seed_count", "[11, 17, 23]" in plan_text and "3 seeds" in proposal_text),
        ("trace_modes", all(name.lower() in proposal_text and name.lower() in idea_text and name.lower() in plan_text for name in ["ExtendedHinted", "CompactState", "NoDirty", "AccessOnly"])),
        ("online_anchors", "online anchor" in proposal_text and "online anchor" in plan_text),
        ("runtime_schedule", "8-hour" in proposal_text and "8-hour" in plan_text),
        ("narrowed_scope", "simulator-backed" in proposal_text and "simulator-backed" in plan_text),
    ]
    return [{"check": name, "ok": ok} for name, ok in checks]
