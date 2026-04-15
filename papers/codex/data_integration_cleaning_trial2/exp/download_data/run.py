import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.pipeline import download_raw_data, ensure_dir, FIGURES_DIR, TABLES_DIR, ARTIFACTS_DIR, LOGS_DIR, RUNS_DIR, set_env_threads, write_json, machine_metadata


def main() -> None:
    set_env_threads()
    for path in [FIGURES_DIR, TABLES_DIR, ARTIFACTS_DIR, LOGS_DIR, RUNS_DIR]:
        ensure_dir(path)
    download_raw_data()
    write_json(ARTIFACTS_DIR / "machine_metadata.json", machine_metadata())


if __name__ == "__main__":
    main()
