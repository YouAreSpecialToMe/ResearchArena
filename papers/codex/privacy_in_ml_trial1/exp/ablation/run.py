import sys

from exp.shared.pipeline import cli


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--stage", "ablation"])
    cli()
