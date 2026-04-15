import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    seeds = [7, 13, 21]
    methods = ["ce", "supcon", "feature_l2", "relational_mse", "maskcon", "nest"]
    for seed in seeds:
        for method in methods:
            out_dir = ROOT / "exp" / "cifar100" / method / f"seed_{seed}"
            if (out_dir / "final.ckpt").exists():
                continue
            cmd = [
                "python", "-m", "exp.train",
                "--dataset", "cifar100",
                "--method", method,
                "--seed", str(seed),
                "--epochs", "60",
                "--output-dir", str(out_dir),
            ]
            if seed == 7 and method in {"supcon", "relational_mse", "maskcon", "nest"}:
                cmd.append("--save-epoch-metrics")
            run(cmd)

    ablations = [
        ("nest", 7, "random", 10, 0.5, ROOT / "exp" / "cifar100" / "nest_random_graph" / "seed_7"),
        ("nest", 7, "weak", 10, 0.5, ROOT / "exp" / "cifar100" / "nest_weak_graph" / "seed_7"),
        ("nest", 7, "pretrained", 5, 0.5, ROOT / "exp" / "cifar100" / "nest_k5" / "seed_7"),
        ("nest", 7, "pretrained", 10, 0.1, ROOT / "exp" / "cifar100" / "nest_lambda_0p1" / "seed_7"),
    ]
    for method, seed, graph_type, k, lam, out_dir in ablations:
        if (out_dir / "final.ckpt").exists():
            continue
        cmd = [
            "python", "-m", "exp.train",
            "--dataset", "cifar100",
            "--method", method,
            "--seed", str(seed),
            "--epochs", "60",
            "--graph-type", graph_type,
            "--k", str(k),
            "--lambda-nest", str(lam),
            "--output-dir", str(out_dir),
        ]
        if seed == 7:
            cmd.append("--save-epoch-metrics")
        run(cmd)


if __name__ == "__main__":
    main()
