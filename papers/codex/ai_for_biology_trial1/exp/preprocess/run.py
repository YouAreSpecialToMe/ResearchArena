from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared import config
from exp.shared.data import prepare_dataset_split
from exp.shared.utils import (
    Timer,
    append_jsonl,
    capture_config_snapshot,
    capture_environment_metadata,
    save_json,
)


def main() -> None:
    out_dir = ROOT / "exp" / "preprocess"
    log_dir = out_dir / "logs"
    audit_dir = out_dir / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "preprocess",
        "environment": capture_environment_metadata(),
        "config": capture_config_snapshot(config),
        "datasets": {},
    }
    for dataset in config.DATASETS:
        results["datasets"][dataset] = {}
        for seed in config.SEEDS:
            log_path = log_dir / f"{dataset}_seed{seed}.jsonl"
            with Timer() as timer:
                split = prepare_dataset_split(dataset, seed)
            payload = dict(split.audit)
            payload["runtime_minutes"] = timer.minutes
            results["datasets"][dataset][str(seed)] = payload
            save_json(audit_dir / f"{dataset}_seed{seed}.json", payload)
            append_jsonl(
                log_path,
                {
                    "event": "finish",
                    "dataset": dataset,
                    "seed": seed,
                    "runtime_minutes": timer.minutes,
                    "n_train": len(split.train_perts),
                    "n_val": len(split.val_perts),
                    "n_test": len(split.test_perts),
                },
            )
    save_json(out_dir / "results.json", results)


if __name__ == "__main__":
    main()
