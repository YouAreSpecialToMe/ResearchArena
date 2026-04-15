from __future__ import annotations

from .pipeline import run_all
from .utils import ROOT, json_dump
from .visualize import make_figures


def main() -> None:
    results = run_all()
    json_dump(results, ROOT / "results.json")
    make_figures()


if __name__ == "__main__":
    main()
