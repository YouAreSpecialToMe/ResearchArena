from pathlib import Path

from exp.shared.data import write_splits
from exp.shared.utils import project_path, write_json


def main() -> None:
    repo_root = project_path()
    write_splits(repo_root)
    manifest = {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "detector": "IDEA-Research/grounding-dino-tiny",
        "verifier": "google/siglip-base-patch16-224",
        "resolution": [512, 512],
        "sampler": "Euler ancestral",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "seeds": [11, 22, 33],
        "candidate_counts": {"default": 4, "candidate_budget_subset": 8},
        "resources": {"gpu": "1x NVIDIA RTX A6000 48GB", "ram": "60GB requested / 503GiB visible", "cpu_cores": 4},
        "notes": [
            "The legacy daam package conflicts with diffusers==0.30.*, so cross-attention tracing is implemented locally in exp/shared/attention.py.",
            "Manual human annotation steps are not automatically executable; the codebase will emit manifests and SKIPPED notes if no human labels are provided.",
        ],
    }
    write_json(project_path("configs", "experiment_manifest.json"), manifest)


if __name__ == "__main__":
    main()
