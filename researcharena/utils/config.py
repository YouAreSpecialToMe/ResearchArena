"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base config."""
    merged = base.copy()
    for k, v in overrides.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged
