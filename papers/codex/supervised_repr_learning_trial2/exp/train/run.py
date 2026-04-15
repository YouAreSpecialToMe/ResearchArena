import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.data import SEEDS, load_cached_split
from exp.shared.train import train_method
from exp.shared.utils import ROOT, ensure_dir


METHODS = {
    "linear_probe",
    "ce_adapter",
    "contrastive_adapter",
    "fixed_k_contrastive",
    "fixed_k_noncontrastive",
    "pb_spread",
    "adaptive_vmf",
    "ablation_no_adapt",
    "ablation_no_vmf",
    "ablation_no_occ",
    "adaptive_margin_0025",
    "adaptive_margin_0050",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["waterbirds", "cub"])
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    args = parser.parse_args()

    cache_payloads = {name: load_cached_split(args.dataset, name) for name in ["train_base", "train_aug1", "train_aug2", "val", "test"]}
    out_method = args.method
    margin = 0.0
    base_method = args.method
    if args.method == "adaptive_margin_0025":
        base_method = "adaptive_vmf"
        margin = 0.0025
    elif args.method == "adaptive_margin_0050":
        base_method = "adaptive_vmf"
        margin = 0.0050
    output_dir = ensure_dir(ROOT / "exp" / args.dataset / out_method / f"seed_{args.seed}")
    train_method(args.dataset, base_method, args.seed, cache_payloads, output_dir, sensitivity_margin=margin)


if __name__ == "__main__":
    main()
