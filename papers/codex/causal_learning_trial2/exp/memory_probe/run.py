from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from exp.shared.common import ensure_dir, load_json, save_json, set_thread_env
from exp.shared.discovery import build_particles, save_particles
from exp.shared.pipeline import RolloutConfig, run_rollout
from exp.shared.sem import load_instance


def main() -> None:
    set_thread_env()
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent
    ensure_dir(out_dir / "logs")
    prep = load_json(root / "exp" / "data_prep" / "results.json")
    pilot = load_json(root / "exp" / "pilot" / "results.json")
    shortlist_k = pilot["shortlist_k"]
    eps = pilot["best_pacer"]["epsilon_stop"]
    tau = pilot["best_aoed"]["tau_stop"]
    eta = pilot["best_aoed"]["eta_stop"]
    probe_path = next(path for path in prep["core"] if "core_p15_scale_free_weak_s11" in path)
    instance = load_instance(Path(probe_path))
    cache_path = root / "artifacts" / "states" / f"{Path(probe_path).stem}_memory_probe_initial.pkl"
    save_particles(cache_path, build_particles(instance.observational_data, [], instance.seed + 9000))
    methods = {
        "fges_only": RolloutConfig(method="fges_only"),
        "random_active": RolloutConfig(method="random_active"),
        "git": RolloutConfig(method="git", shortlist_k=shortlist_k),
        "aoed_lite": RolloutConfig(method="aoed_lite", tau_stop=tau, eta_stop=eta, shortlist_k=shortlist_k),
        "pacer_no_d": RolloutConfig(method="pacer_no_d", epsilon_stop=eps, disable_detectability=True, shortlist_k=shortlist_k),
        "pacer_cert": RolloutConfig(method="pacer_cert", epsilon_stop=eps, shortlist_k=shortlist_k),
        "pacer_fixed_batch": RolloutConfig(method="pacer_fixed_batch", epsilon_stop=eps, fixed_batch_size=50, shortlist_k=shortlist_k),
        "pacer_full_budget": RolloutConfig(method="pacer_cert", epsilon_stop=eps, disable_early_stop=True, shortlist_k=shortlist_k),
    }
    rows = []
    for method, config in methods.items():
        result = run_rollout(
            instance,
            config,
            out_dir / method,
            instance.seed,
            initial_particles=build_particles(instance.observational_data, [], instance.seed + 9000),
        )
        rows.append(
            {
                "method": method,
                "instance_id": instance.instance_id,
                "runtime_seconds": result["runtime_seconds"],
                "peak_rss_mb": result["peak_rss_mb"],
                "directed_f1": result["directed_f1"],
                "unused_budget": result["unused_budget"],
                "stop_flag": result["stop_flag"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "memory_probe.csv", index=False)
    save_json(
        out_dir / "results.json",
        {
            "probe_instance": instance.instance_id,
            "rows": df.to_dict(orient="records"),
            "memory_probe_path": str(out_dir / "memory_probe.csv"),
        },
    )


if __name__ == "__main__":
    main()
