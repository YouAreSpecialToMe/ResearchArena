from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.benchmark_spec import SEEDS
from exp.shared.eval_lib import aggregate_condition, bm25_rule_prediction, load_items, save_seed_outputs, score_prediction
from exp.shared.utils import ROOT, ensure_dir, timestamp, write_json


def main() -> None:
    log_dir = ensure_dir(ROOT / "exp" / "rule_baseline" / "logs")
    items = load_items()
    log_lines = [
        "experiment=rule_baseline",
        "action=run_bm25_evidence_baseline",
        "deterministic=true",
        f"replicated_seed_dirs={SEEDS}",
    ]
    for seed in SEEDS:
        predictions = []
        executions = []
        for item in items:
            start = time.time()
            pred = bm25_rule_prediction(item)
            latency = time.time() - start
            record = {"item_id": item["item_id"], "latency_sec": latency, "malformed_output": False, **pred}
            execution = score_prediction(item, pred)
            execution["latency_sec"] = latency
            execution["malformed_output"] = False
            predictions.append(record)
            executions.append(execution)
            print(f"[rule_baseline] seed={seed} item={item['item_id']} label={pred['predicted_binary_label']} repair={bool(pred['repaired_code'].strip())}", flush=True)
        save_seed_outputs("rule_baseline", seed, predictions, executions)
        log_lines.append(
            f"seed={seed} needs_update_repair_successes={sum(r['needs_update_repair_pass'] for r in executions if r['gold_label'] == 'needs_update')}"
        )

    summary = aggregate_condition("rule_baseline")
    write_json(
        ROOT / "exp" / "rule_baseline" / "results.json",
        {"experiment": "rule_baseline", "config": {"seeds": SEEDS, "deterministic": True}, "metrics": summary},
    )
    log_lines.append(f"mean_needs_update_repair_pass_rate={summary['needs_update_repair_pass_rate']['mean']}")
    (log_dir / f"{timestamp()}_run.log").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
