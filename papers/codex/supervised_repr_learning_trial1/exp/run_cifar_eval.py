import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def maybe_eval(method: str, seed: int, graph_type: str = "pretrained", k: int = 10, run_name: str | None = None):
    run_dir = ROOT / "exp" / "cifar100" / (run_name or method) / f"seed_{seed}"
    metrics = run_dir / "metrics_final.json"
    if not (run_dir / "final.ckpt").exists():
        return
    if metrics.exists():
        import json
        data = json.loads(metrics.read_text())
        if "fine_linear_probe" in data:
            return
    cmd = [
        "python", "-m", "exp.evaluate_checkpoint",
        "--dataset", "cifar100",
        "--method", method,
        "--graph-type", graph_type,
        "--k", str(k),
        "--run-dir", str(run_dir),
    ]
    run(cmd)


def main():
    for seed in [7, 13, 21]:
        for method in ["ce", "supcon", "feature_l2", "relational_mse", "maskcon", "nest"]:
            maybe_eval(method, seed)
    maybe_eval("nest", 7, graph_type="random", run_name="nest_random_graph")
    maybe_eval("nest", 7, graph_type="weak", run_name="nest_weak_graph")
    maybe_eval("nest", 7, graph_type="pretrained", k=5, run_name="nest_k5")
    maybe_eval("nest", 7, graph_type="pretrained", k=10, run_name="nest_lambda_0p1")


if __name__ == "__main__":
    main()
