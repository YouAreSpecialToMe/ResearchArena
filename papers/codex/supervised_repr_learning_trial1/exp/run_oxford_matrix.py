import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    seeds = [7, 13]
    methods = ["supcon", "maskcon", "nest"]
    for seed in seeds:
        for method in methods:
            out_dir = ROOT / "exp" / "oxford_pet" / method / f"seed_{seed}"
            if (out_dir / "final.ckpt").exists():
                continue
            cmd = [
                "python", "-m", "exp.train",
                "--dataset", "oxford_pet",
                "--method", method,
                "--seed", str(seed),
                "--epochs", "20",
                "--output-dir", str(out_dir),
            ]
            run(cmd)


if __name__ == "__main__":
    main()
