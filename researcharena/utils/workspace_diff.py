"""Capture filesystem changes in the workspace before/after agent invocations.

Takes a snapshot of all files (path, size, mtime) before the agent runs,
then diffs against the state after. Reports files created, modified, and deleted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileInfo:
    path: str          # relative to workspace
    size: int          # bytes
    md5: str           # content hash (first 8KB for large files)

    def to_dict(self) -> dict:
        return {"path": self.path, "size": self.size}


@dataclass
class WorkspaceDiff:
    """Diff between two workspace snapshots."""
    created: list[FileInfo] = field(default_factory=list)
    modified: list[FileInfo] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {}
        if self.created:
            d["created"] = [f.to_dict() for f in self.created]
        if self.modified:
            d["modified"] = [f.to_dict() for f in self.modified]
        if self.deleted:
            d["deleted"] = self.deleted
        return d

    @property
    def total_changes(self) -> int:
        return len(self.created) + len(self.modified) + len(self.deleted)

    def summary(self) -> str:
        parts = []
        if self.created:
            parts.append(f"+{len(self.created)} created")
        if self.modified:
            parts.append(f"~{len(self.modified)} modified")
        if self.deleted:
            parts.append(f"-{len(self.deleted)} deleted")
        return ", ".join(parts) if parts else "no changes"


# Directories to skip — logs and review_logs are written by the harness, not the agent
_SKIP_DIRS = {"logs", "review_logs", ".git", "__pycache__", ".cache", "node_modules"}


def snapshot(workspace: Path) -> dict[str, FileInfo]:
    """Take a snapshot of all files in the workspace.

    Returns:
        Dict mapping relative path → FileInfo
    """
    files: dict[str, FileInfo] = {}

    if not workspace.exists():
        return files

    for path in workspace.rglob("*"):
        if not path.is_file():
            continue

        # Skip harness-generated directories
        rel = path.relative_to(workspace)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue

        rel_str = str(rel)
        try:
            size = path.stat().st_size
            md5 = _quick_hash(path)
            files[rel_str] = FileInfo(path=rel_str, size=size, md5=md5)
        except OSError:
            pass

    return files


def diff(before: dict[str, FileInfo], after: dict[str, FileInfo]) -> WorkspaceDiff:
    """Compute the diff between two snapshots."""
    result = WorkspaceDiff()

    all_paths = set(before.keys()) | set(after.keys())

    for path in sorted(all_paths):
        in_before = path in before
        in_after = path in after

        if in_after and not in_before:
            result.created.append(after[path])
        elif in_before and not in_after:
            result.deleted.append(path)
        elif in_before and in_after:
            if before[path].md5 != after[path].md5:
                result.modified.append(after[path])

    return result


def _quick_hash(path: Path) -> str:
    """Hash file content. Reads in chunks to handle large files."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()
