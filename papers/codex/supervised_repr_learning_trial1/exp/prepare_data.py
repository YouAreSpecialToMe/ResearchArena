import argparse
import subprocess
from pathlib import Path

from exp.shared.data import get_dataset_bundle
from exp.shared.graph import build_graph_from_features, extract_features, graph_metrics, teacher_graph_paths
from exp.shared.utils import analysis_root, ensure_dir, repo_root, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cifar100", "oxford_pet"])
    p.add_argument("--include-weak", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    bundle = get_dataset_bundle(args.dataset)
    paths = teacher_graph_paths(args.dataset, graph_type="pretrained", k=10)
    if not paths["features"].exists():
        extract_features(args.dataset, paths["features"])
    if not paths["graph"].exists():
        build_graph_from_features(args.dataset, paths["features"], paths["graph"], graph_type="pretrained", k=10)
    graph_metrics(paths["graph"], paths["metrics"])

    if args.dataset == "cifar100":
        rnd = teacher_graph_paths(args.dataset, graph_type="random", k=10)
        if not rnd["graph"].exists():
            build_graph_from_features(args.dataset, paths["features"], rnd["graph"], graph_type="random", k=10)
        graph_metrics(rnd["graph"], rnd["metrics"])

        k5 = teacher_graph_paths(args.dataset, graph_type="pretrained", k=5)
        if not k5["graph"].exists():
            build_graph_from_features(args.dataset, paths["features"], k5["graph"], graph_type="pretrained", k=5)
        graph_metrics(k5["graph"], k5["metrics"])

        if args.include_weak:
            weak_out = repo_root() / "exp" / "cifar100" / "weak_teacher" / "seed_7"
            ensure_dir(weak_out)
            if not (weak_out / "final.ckpt").exists():
                subprocess.run(
                    [
                        "python", "-m", "exp.train",
                        "--dataset", "cifar100",
                        "--method", "ce",
                        "--seed", "7",
                        "--epochs", "15",
                        "--skip-eval",
                        "--output-dir", str(weak_out),
                    ],
                    check=True,
                )
            weak = teacher_graph_paths(args.dataset, graph_type="weak", k=10)
            if not weak["features"].exists():
                extract_features(args.dataset, weak["features"], checkpoint_path=weak_out / "final.ckpt")
            if not weak["graph"].exists():
                build_graph_from_features(args.dataset, weak["features"], weak["graph"], graph_type="weak", k=10)
            graph_metrics(weak["graph"], weak["metrics"])


if __name__ == "__main__":
    main()
