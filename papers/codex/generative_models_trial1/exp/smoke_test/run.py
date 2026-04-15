import json
from pathlib import Path

import torch

from exp.shared.run_core import run_generation
from exp.shared.utils import project_path, write_json


def main() -> None:
    rows = run_generation("dev", k=1, limit=10)
    total_runtime = sum(row["runtime_sec"] for row in rows)
    heatmap_sizes = []
    for row in rows:
        heatmap_sizes.append(Path(row["heatmap_path"]).stat().st_size / (1024 * 1024))
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    result = {
        "num_images": len(rows),
        "per_image_runtime_sec": total_runtime / max(1, len(rows)),
        "peak_vram_gb": peak_vram,
        "mean_daam_trace_mb": sum(heatmap_sizes) / max(1, len(heatmap_sizes)),
        "status": "pass" if peak_vram < 44.0 and (total_runtime / max(1, len(rows))) < 12.0 else "warn",
    }
    write_json(project_path("exp", "smoke_test", "results.json"), result)


if __name__ == "__main__":
    main()
