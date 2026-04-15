import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.runner import aggregate
from exp.shared.utils import ensure_dir, json_dump

SEEDS = [11, 22, 33]


def base_cifar(method, setting="coarse", rank=2, label_type="coarse", epochs=30):
    return {
        "dataset": "cifar100",
        "data_root": str(ROOT / "data" / "cifar100"),
        "results_root": str(ROOT / "results"),
        "batch_size": 256,
        "num_workers": 4,
        "embedding_dim": 128,
        "method": method,
        "setting": setting,
        "label_type": label_type,
        "epochs": epochs,
        "lr": 0.5,
        "tau_c": 0.2,
        "tau_a": 0.1,
        "supcon_temp": 0.1,
        "lambda_supcon": 1.0,
        "lambda_div": 0.05,
        "lambda_cov": 0.02,
        "rank": rank,
        "status": "completed",
    }


def base_synthetic(method, regime, rank=2, epochs=15):
    return {
        "dataset": "synthetic",
        "synthetic_root": str(ROOT / "data" / "synthetic_modes"),
        "batch_size": 512,
        "method": method,
        "setting": regime,
        "regime": regime,
        "epochs": epochs,
        "lr": 1e-3,
        "tau_c": 0.2,
        "tau_a": 0.1,
        "supcon_temp": 0.1,
        "lambda_supcon": 1.0,
        "lambda_div": 0.05,
        "lambda_cov": 0.02,
        "rank": rank,
        "status": "completed",
    }


def write_config(config, path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_one(config, output_dir):
    ensure_dir(output_dir)
    result_path = output_dir / "results.json"
    if result_path.exists():
        print(f"SKIP {output_dir}", flush=True)
        return
    config_path = output_dir / "config.yaml"
    write_config(config, config_path)
    logs_dir = output_dir / "logs"
    ensure_dir(logs_dir)
    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "exp" / "shared" / "runner.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    print("RUN", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with open(stdout_path, "w", encoding="utf-8") as stdout_f, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_f:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=stdout_f, stderr=stderr_f, env=env)


def smoke_jobs():
    jobs = []
    for method in ["cross_entropy", "supcon", "psc", "mpsc", "clop_style", "span"]:
        config = base_cifar(method, setting="smoke", epochs=3)
        config["seed"] = 11
        config["status"] = "smoke_only"
        jobs.append((config, ROOT / "exp" / method / "smoke" / "seed11"))
    for regime in ["directional_low_rank", "mean_or_isotropic"]:
        for method in ["cross_entropy", "supcon", "psc", "mpsc", "clop_style", "span"]:
            config = base_synthetic(method, regime, epochs=3)
            config["seed"] = 11
            config["status"] = "smoke_only"
            jobs.append((config, ROOT / "exp" / "synthetic" / regime / method / "smoke_seed11"))
    return jobs


def full_jobs():
    jobs = []
    for method in ["cross_entropy", "supcon", "psc", "mpsc", "clop_style", "span"]:
        for seed in SEEDS:
            config = base_cifar(method, setting="coarse")
            config["seed"] = seed
            jobs.append((config, ROOT / "exp" / method / "coarse" / f"seed{seed}"))
    for regime in ["directional_low_rank", "mean_or_isotropic"]:
        for method in ["cross_entropy", "supcon", "psc", "mpsc", "clop_style", "span"]:
            for seed in SEEDS:
                config = base_synthetic(method, regime)
                config["seed"] = seed
                jobs.append((config, ROOT / "exp" / "synthetic" / regime / method / f"seed{seed}"))
    ablations = [
        ("span_rank1", 1, "rank1"),
        ("span_no_div", 2, "no_div"),
        ("span_no_cov", 2, "no_cov"),
    ]
    for method, rank, setting in ablations:
        for seed in SEEDS:
            config = base_cifar(method, setting=setting, rank=rank)
            config["seed"] = seed
            jobs.append((config, ROOT / "exp" / method / setting / f"seed{seed}"))
    for method in ["supcon", "mpsc", "span"]:
        config = base_cifar(method, setting="fine", label_type="fine", epochs=30)
        config["seed"] = 11
        jobs.append((config, ROOT / "exp" / "fine_sanity" / method / "seed11"))
    return jobs


def freeze_base_configs():
    configs = {
        "cross_entropy.yaml": base_cifar("cross_entropy"),
        "supcon.yaml": base_cifar("supcon"),
        "psc.yaml": base_cifar("psc"),
        "mpsc.yaml": base_cifar("mpsc"),
        "clop_style.yaml": base_cifar("clop_style"),
        "span.yaml": base_cifar("span"),
        "span_rank1.yaml": base_cifar("span_rank1", setting="rank1", rank=1),
        "span_no_div.yaml": base_cifar("span_no_div", setting="no_div"),
        "span_no_cov.yaml": base_cifar("span_no_cov", setting="no_cov"),
        "synthetic.yaml": base_synthetic("span", "directional_low_rank"),
    }
    for name, cfg in configs.items():
        write_config(cfg, ROOT / "configs" / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["freeze", "smoke", "full"], required=True)
    args = parser.parse_args()
    freeze_base_configs()
    jobs = []
    if args.mode == "freeze":
        return
    if args.mode == "smoke":
        jobs = smoke_jobs()
    elif args.mode == "full":
        jobs = full_jobs()
    manifest = []
    for config, out_dir in jobs:
        manifest.append({"output_dir": str(out_dir), "config": copy.deepcopy(config)})
        run_one(config, out_dir)
    json_dump(manifest, ROOT / "results" / f"{args.mode}_manifest.json")
    aggregate(ROOT / "exp", ROOT / "results" / "results.json")


if __name__ == "__main__":
    main()
