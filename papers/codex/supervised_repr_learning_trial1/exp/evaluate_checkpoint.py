import argparse
from pathlib import Path

import numpy as np
import torch

from exp.shared.eval import evaluate_model
from exp.shared.graph import teacher_graph_paths
from exp.shared.models import create_model
from exp.shared.utils import device, elapsed_minutes, now, write_json


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
    ckpt = torch.load(run_dir / "final.ckpt", map_location=device())
    model = create_model(20 if args.dataset == "cifar100" else 2).to(device())
    model.load_state_dict(ckpt["model"], strict=False)
    graph_path = None
    if args.method in {"relational_mse", "maskcon", "nest"}:
        graph_path = teacher_graph_paths(args.dataset, graph_type=args.graph_type, k=args.k)["graph"]
    start = now()
    metrics = evaluate_model(model, args.dataset, str(graph_path) if graph_path is not None else None)
    metrics["evaluation_minutes"] = elapsed_minutes(start)
    prev = {}
    metrics_path = run_dir / "metrics_final.json"
    if metrics_path.exists():
        import json
        prev = json.loads(metrics_path.read_text())
    prev.update(metrics)
    write_json(metrics_path, prev)


if __name__ == "__main__":
    main()
