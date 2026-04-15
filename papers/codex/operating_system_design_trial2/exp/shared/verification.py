from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import ROOT, read_json, write_json


def trace_dependencies(trace_path: Path) -> list[str]:
    payload = read_json(trace_path)
    meta = payload.get("meta", {})
    deps = meta.get("dependencies", [])
    if isinstance(deps, list):
        return [str(dep) for dep in deps]
    return []


def verify_run_artifacts() -> dict[str, Any]:
    issues: list[str] = []
    manifest_path = ROOT / "calibration" / "manifest.json"
    if not manifest_path.exists():
        issues.append("missing calibration/manifest.json")
    for config_path in sorted(ROOT.glob("exp/*/runs/*/config.json")):
        try:
            cfg = read_json(config_path)
        except Exception as exc:
            issues.append(f"unreadable config {config_path.relative_to(ROOT)}: {exc}")
            continue
        run_dir = config_path.parent
        trace_path = ROOT / cfg["trace_path"]
        if not trace_path.exists():
            issues.append(f"missing trace {trace_path.relative_to(ROOT)} referenced by {config_path.relative_to(ROOT)}")
            continue
        for dep in trace_dependencies(trace_path):
            dep_path = ROOT / dep
            if not dep_path.exists():
                issues.append(
                    f"missing dependency {dep_path.relative_to(ROOT)} for trace {trace_path.relative_to(ROOT)} "
                    f"referenced by {config_path.relative_to(ROOT)}"
                )
        for dep in cfg.get("input_artifacts", []):
            dep_path = ROOT / dep
            if not dep_path.exists():
                issues.append(f"missing config dependency {dep_path.relative_to(ROOT)} in {config_path.relative_to(ROOT)}")
        results_path = run_dir / "results.json"
        if not results_path.exists():
            issues.append(f"missing results {results_path.relative_to(ROOT)}")
        log_path = ROOT / "exp" / cfg["experiment"] / "logs" / (
            f"{cfg['workload_family']}__{cfg['cache_budget']}__{cfg['method']}__seed{cfg['seed']}.log"
        )
        if not log_path.exists():
            issues.append(f"missing log {log_path.relative_to(ROOT)}")
    report = {
        "ok": not issues,
        "issue_count": len(issues),
        "issues": issues,
    }
    write_json(ROOT / "replay_results" / "verification.json", report)
    return report


def verify_or_raise() -> None:
    report = verify_run_artifacts()
    if not report["ok"]:
        joined = "\n".join(report["issues"])
        raise RuntimeError(f"artifact verification failed:\n{joined}")
