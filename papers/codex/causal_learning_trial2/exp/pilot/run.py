from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.common import ensure_dir, load_json, save_json, set_thread_env
from exp.shared.pipeline import RolloutConfig, run_rollout
from exp.shared.sem import load_instance


def run_or_load(instance, config: RolloutConfig, out: Path, seed: int, force_rerun: bool) -> dict:
    results_path = out / "results.json"
    if results_path.exists() and not force_rerun:
        return load_json(results_path)
    return run_rollout(instance, config, out, seed)


def choose_pilot_paths(aux_paths: list[str]) -> list[str]:
    parsed = []
    for path in aux_paths:
        inst = load_instance(Path(path))
        parsed.append((inst.graph_family, inst.weight_regime, inst.seed, path))
    chosen = []
    for family in ["erdos_renyi", "scale_free"]:
        for regime in ["weak", "mixed"]:
            candidates = sorted([row for row in parsed if row[0] == family and row[1] == regime], key=lambda row: row[2])
            if candidates:
                chosen.append(candidates[0][3])
    return chosen


def tune_thresholds(aux_paths: list[str], force_rerun: bool) -> dict:
    pilot_paths = choose_pilot_paths(aux_paths)
    pacer_grid = [0.20, 0.30, 0.40]
    aoed_grid = [(0.80, 0.002), (0.90, 0.002), (0.90, 0.005)]
    pacer_scores = []
    aoed_scores = []
    for eps in pacer_grid:
        vals = []
        for idx, path in enumerate(pilot_paths):
            inst = load_instance(Path(path))
            out = Path(__file__).resolve().parent / f"pacer_eps_{str(eps).replace('.', '_')}_{idx}"
            result = run_or_load(inst, RolloutConfig(method="pacer_cert", epsilon_stop=eps, shortlist_k=4), out, inst.seed + 7, force_rerun)
            vals.append(result["directed_f1"] - 0.01 * result["unused_budget"])
        pacer_scores.append({"epsilon_stop": eps, "score": float(sum(vals) / len(vals))})
    for tau, eta in aoed_grid:
        vals = []
        for idx, path in enumerate(pilot_paths):
            inst = load_instance(Path(path))
            out = Path(__file__).resolve().parent / f"aoed_tau_{str(tau).replace('.', '_')}_eta_{str(eta).replace('.', '_')}_{idx}"
            result = run_or_load(inst, RolloutConfig(method="aoed_lite", tau_stop=tau, eta_stop=eta, shortlist_k=4), out, inst.seed + 11, force_rerun)
            vals.append(result["directed_f1"] - 0.01 * result["unused_budget"])
        aoed_scores.append({"tau_stop": tau, "eta_stop": eta, "score": float(sum(vals) / len(vals))})
    best_pacer = max(pacer_scores, key=lambda row: row["score"])
    best_aoed = max(aoed_scores, key=lambda row: row["score"])
    return {"best_pacer": best_pacer, "best_aoed": best_aoed, "pacer_grid": pacer_scores, "aoed_grid": aoed_scores}


def main() -> None:
    set_thread_env()
    force_rerun = os.environ.get("FORCE_RERUN", "0") == "1"
    ensure_dir(Path(__file__).resolve().parent / "logs")
    prep = load_json(Path(__file__).resolve().parents[2] / "exp" / "data_prep" / "results.json")
    inst = load_instance(Path(sorted(prep["core"])[3]))
    methods = [
        RolloutConfig(method="pacer_cert"),
        RolloutConfig(method="pacer_no_d", disable_detectability=True),
        RolloutConfig(method="git"),
        RolloutConfig(method="aoed_lite", tau_stop=0.90, eta_stop=0.002),
    ]
    pilot_results = []
    shortlist_k = None
    for cfg in methods:
        out_dir = ensure_dir(Path(__file__).resolve().parent / f"pilot_{cfg.method}")
        result = run_or_load(inst, cfg, out_dir, inst.seed + 5, force_rerun)
        pilot_results.append({"method": cfg.method, "runtime_seconds": result["runtime_seconds"], "directed_f1": result["directed_f1"]})
    pacer_runtime = next(row["runtime_seconds"] for row in pilot_results if row["method"] == "pacer_cert")
    aoed_runtime = next(row["runtime_seconds"] for row in pilot_results if row["method"] == "aoed_lite")
    if pacer_runtime > 120 or aoed_runtime > 150:
        shortlist_k = 4
    thresholds = tune_thresholds(prep["aux"], force_rerun)
    payload = {"pilot_results": pilot_results, "shortlist_k": shortlist_k, "pilot_paths": choose_pilot_paths(prep["aux"]), **thresholds}
    save_json(Path(__file__).resolve().parent / "results.json", payload)


if __name__ == "__main__":
    main()
