import argparse

from exp.shared.run_core import run_single_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    run_single_sample(args.split)


if __name__ == "__main__":
    main()
