"""Checkpoint support for pipeline state.

Saves and loads PipelineState to/from a JSON file in the workspace,
enabling resume after preemption or crash.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

CHECKPOINT_FILENAME = "checkpoint.json"


def save_checkpoint(state, base_dir: Path, tracker=None) -> Path:
    """Save pipeline state to checkpoint file.

    Args:
        state: PipelineState dataclass instance.
        base_dir: Workspace root directory.
        tracker: Optional RunTracker — its action log is saved alongside state.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_path = base_dir / CHECKPOINT_FILENAME

    data = {}
    for f in fields(state):
        val = getattr(state, f.name)
        # Stage enum → string
        if f.name == "stage":
            data[f.name] = val.value
        # BestPaper dataclass → dict with Path converted
        elif f.name == "best":
            best_dict = {}
            for bf in fields(val):
                bv = getattr(val, bf.name)
                if isinstance(bv, Path):
                    best_dict[bf.name] = str(bv)
                elif bf.name == "review_result":
                    # ReviewResult is not trivially serializable — skip it,
                    # the pipeline doesn't need it to resume
                    best_dict[bf.name] = None
                elif bf.name == "idea":
                    best_dict[bf.name] = bv
                else:
                    best_dict[bf.name] = bv
            data[f.name] = best_dict
        elif isinstance(val, Path):
            data[f.name] = str(val)
        # ReviewResult — skip (not needed for resume)
        elif f.name == "review_result":
            data[f.name] = None
        else:
            data[f.name] = val

    # Also save tracker actions so we don't lose tracking data
    if tracker is not None:
        data["_tracker_actions"] = [
            a.to_dict() if hasattr(a, "to_dict") else a
            for a in (tracker.actions if hasattr(tracker, "actions") else [])
        ]

    # Atomic write: write to temp file then rename, so a kill mid-write
    # won't corrupt the checkpoint (rename is atomic on POSIX)
    tmp_path = checkpoint_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, default=str))
    tmp_path.rename(checkpoint_path)
    return checkpoint_path


def load_checkpoint(base_dir: Path) -> dict | None:
    """Load checkpoint data from workspace.

    Returns:
        Dict of checkpoint data, or None if no checkpoint exists.
    """
    checkpoint_path = base_dir / CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        return None

    return json.loads(checkpoint_path.read_text())


def restore_state(state, checkpoint: dict):
    """Restore PipelineState fields from a checkpoint dict.

    Args:
        state: PipelineState instance to restore into.
        checkpoint: Dict from load_checkpoint().
    """
    from researcharena.pipeline import Stage, BestPaper

    for f in fields(state):
        if f.name not in checkpoint:
            continue
        val = checkpoint[f.name]

        if f.name == "stage":
            state.stage = Stage(val)
        elif f.name == "workspace" and val is not None:
            state.workspace = Path(val)
        elif f.name == "best":
            if val is not None:
                best = BestPaper()
                best.score = val.get("score", 0.0)
                best.idea = val.get("idea")
                pdf = val.get("paper_pdf_path")
                best.paper_pdf_path = Path(pdf) if pdf else None
                ws = val.get("workspace")
                best.workspace = Path(ws) if ws else None
                state.best = best
        elif f.name == "review_result":
            # Can't restore ReviewResult — leave as None
            pass
        else:
            setattr(state, f.name, val)
