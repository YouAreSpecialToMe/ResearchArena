from __future__ import annotations

from exp.shared.analysis import aggregate_results, make_figures
from exp.shared.common import ensure_layout


def main() -> None:
    ensure_layout()
    aggregate_results()
    make_figures()


if __name__ == "__main__":
    main()
