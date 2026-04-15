import os
from pathlib import Path

from exp.shared.core import RESULTS, ensure_dirs, package_versions, system_info, write_json


def main() -> None:
    ensure_dirs()
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "1"
    out_dir = RESULTS / "system_info"
    out_dir.mkdir(parents=True, exist_ok=True)
    info = system_info()
    info["execution_policy"] = {
        "cpu_only": True,
        "max_concurrent_jobs": 2,
        "gpu_parallelism_applicable": False,
    }
    write_json(out_dir / "system_info.json", info)
    write_json(out_dir / "package_versions.json", package_versions())
    write_json(
        Path(__file__).resolve().parent / "results.json",
        {
            "experiment": "environment_setup",
            "status": "completed",
            "system_info_path": str(out_dir / "system_info.json"),
            "package_versions_path": str(out_dir / "package_versions.json"),
        },
    )


if __name__ == "__main__":
    main()
