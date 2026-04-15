import argparse

from exp.shared.run_core import run_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    run_method(args.split, "daam_no_counterfactual", k=args.k, assignment_source="detector_daam", use_counterfactual=False)


if __name__ == "__main__":
    main()
