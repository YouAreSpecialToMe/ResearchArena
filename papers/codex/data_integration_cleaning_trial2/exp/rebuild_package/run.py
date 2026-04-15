import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.aggregate.run import main as aggregate_main
from exp.download_data.run import main as download_main
from exp.prepare_data.run import main as prepare_main
from exp.run_suite.run import run_all
from exp.shared.pipeline import (
    FIGURES_DIR,
    LOGS_DIR,
    ROOT,
    RUNS_DIR,
    TABLES_DIR,
    archive_previous_outputs,
    ensure_dir,
    set_env_threads,
)


def main() -> None:
    set_env_threads()
    archive_dir = archive_previous_outputs()
    for path in [RUNS_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR]:
        ensure_dir(path)
    if archive_dir is not None:
        print(f"archived previous outputs to {archive_dir.relative_to(ROOT)}")
    download_main()
    prepare_main()
    run_all()
    aggregate_main()


if __name__ == "__main__":
    main()
