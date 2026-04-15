import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.common import ARTIFACTS_DIR, DATA_DIR, EXP_DIR, environment_manifest, write_json


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    manifest = environment_manifest()
    write_json(EXP_DIR / "environment_setup" / "run_manifest.json", manifest)
    write_json(
        Path("generation_defaults.json"),
        {
            **manifest["generation_defaults"],
            "alpha_schedule_candidates": {
                "flat_0.4": [0.4] * 8,
                "early_heavy": [0.6, 0.6, 0.5, 0.5, 0.4, 0.3, 0.2, 0.2],
                "middle_heavy": [0.2, 0.3, 0.5, 0.6, 0.6, 0.5, 0.3, 0.2],
            },
            "wall_clock_budget_hours": 8,
        },
    )
    write_json(
        Path("requirements_lock.txt"),
        {
            "python": manifest["python_version"],
            "note": "Executed with the preinstalled Python 3.12 runtime because a complete Python 3.10 diffusion stack was not available locally.",
            "runtime_mismatch": {
                "planned_python": "3.10.x",
                "actual_python": manifest["python_version"],
                "planned_torch": "2.2.x",
                "actual_torch": manifest["torch_version"],
            },
            "core_packages": {
                "torch": manifest["torch_version"],
                "diffusers": _pkg_version("diffusers"),
                "transformers": _pkg_version("transformers"),
                "accelerate": _pkg_version("accelerate"),
                "torchvision": _pkg_version("torchvision"),
                "lpips": _pkg_version("lpips"),
                "matplotlib": _pkg_version("matplotlib"),
                "pandas": _pkg_version("pandas"),
                "scipy": _pkg_version("scipy"),
                "scheduler": "DDIM",
            },
        },
    )
    write_json(
        Path("bibliography_audit.json"),
        {
            "papers": [
                {"paper_id": "ldm", "title": "High-Resolution Image Synthesis with Latent Diffusion Models", "parsed_copy_present": True, "used_in_plan": True},
                {"paper_id": "compbenchpp", "title": "T2I-CompBench++", "parsed_copy_present": True, "used_in_plan": True},
                {"paper_id": "metalogic", "title": "MetaLogic", "parsed_copy_present": True, "used_in_plan": True},
                {"paper_id": "ssa", "title": "Uncovering Limitations in Text-to-Image Generation: A Contrastive Approach with Structured Semantic Alignment", "parsed_copy_present": True, "used_in_plan": True},
                {"paper_id": "aapb", "title": "Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation", "parsed_copy_present": True, "used_in_plan": True},
                {"paper_id": "maskdiffusion", "title": "MaskDiffusion", "parsed_copy_present": True, "used_in_plan": False},
            ],
            "novelty_position": "ParaDG is a narrow robustness-oriented extension of AAPB-style multi-prompt blending that restricts auxiliary prompts to audited paraphrases and activates stronger blending only under paraphrase disagreement; it is not a broadly new guidance family.",
        },
    )


if __name__ == "__main__":
    main()
