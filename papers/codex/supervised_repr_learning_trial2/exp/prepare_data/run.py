import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.data import cache_features_for_dataset
from exp.shared.utils import ROOT, ensure_dir


def main():
    ensure_dir(ROOT / "data")
    ensure_dir(ROOT / "cache")
    ensure_dir(ROOT / "results")
    ensure_dir(ROOT / "figures")
    ensure_dir(ROOT / "exp")

    with (ROOT / "results" / "study_scope.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "This study is a careful incremental empirical comparison of adaptive per-class prototype count selection\n"
            "in a frozen CLIP head-adaptation regime. It does not claim a fundamentally new representation-learning method.\n\n"
            "Pre-registered comparison set: linear probe, cross-entropy adapter, class-level contrastive adapter,\n"
            "fixed-K=2 contrastive prototype adapter, fixed-K=2 non-contrastive prototype head,\n"
            "PB-inspired frozen-feature spread-preserving comparator, adaptive penalized-vMF split-merge prototype selection.\n"
        )

    with (ROOT / "results" / "hardware.txt").open("w", encoding="utf-8") as handle:
        for cmd in [["nvidia-smi"], ["free", "-h"], ["nproc"]]:
            handle.write(f"$ {' '.join(cmd)}\n")
            handle.write(subprocess.check_output(cmd, text=True))
            handle.write("\n")

    with (ROOT / "results" / "env.txt").open("w", encoding="utf-8") as handle:
        handle.write(subprocess.check_output(["pip", "freeze"], text=True))

    cache_features_for_dataset("waterbirds", ROOT / "data", ROOT / "cache")
    cache_features_for_dataset("cub", ROOT / "data", ROOT / "cache")


if __name__ == "__main__":
    main()
