import argparse
import json
from pathlib import Path

import torch

from exp.evaluate_checkpoint import parse_args as _unused  # noqa: F401
from exp.shared.eval import evaluate_model
from exp.shared.graph import teacher_graph_paths
from exp.shared.models import create_model
from exp.shared.utils import device, elapsed_minutes, write_json, now


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cifar100", "oxford_pet"])
    p.add_argument("--method", required=True)
    p.add_argument("--graph-type", default="pretrained", choices=["pretrained", "random", "weak"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    graph_path = None
    if args.method in {"relational_mse", "maskcon", "nest"}:
        graph_path = teacher_graph_paths(args.dataset, graph_type=args.graph_type, k=args.k)["graph"]

    eval_rows = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device())
        model = create_model(20 if args.dataset == "cifar100" else 2).to(device())
        model.load_state_dict(ckpt["model"], strict=False)
        start = now()
        metrics = evaluate_model(model, args.dataset, str(graph_path) if graph_path is not None else None)
        eval_rows.append({
            "epoch": int(ckpt_path.stem.split("_")[1]),
            "evaluation_minutes": elapsed_minutes(start),
            "fine_linear_probe_test_accuracy": metrics["fine_linear_probe"]["test_accuracy"],
            "fine_knn20_test_accuracy": metrics["fine_knn20"]["test_accuracy"],
            "coarse_acc": metrics["coarse_acc"],
        })

    metrics_by_epoch_path = run_dir / "metrics_by_epoch.json"
    payload = {}
    if metrics_by_epoch_path.exists():
        payload = json.loads(metrics_by_epoch_path.read_text())
    payload["eval_checkpoints"] = eval_rows
    write_json(metrics_by_epoch_path, payload)


if __name__ == "__main__":
    main()
