from __future__ import annotations

from pathlib import Path

from exp.shared.common import ensure_layout, write_json


def main() -> None:
    ensure_layout()
    skipped = {
        "status": "skipped_replay_only_scope",
        "reason": "The planned live proxy requires writable cgroup v2 memory.high controls plus reliable cache-reset and residency sampling hooks. Those controls are not available in this unprivileged environment, so executing the live step faithfully is infeasible here.",
        "scope": "Claims are scoped to replay-only evidence. exp/external_validation remains an auxiliary replay perturbation check and is not presented as a substitute for the planned live sanity check.",
        "rows": [],
    }
    write_json(Path("live_validation/results.json"), skipped)
    Path("exp/live_validation/SKIPPED.md").write_text(
        "Live validation was not executed.\n\n"
        "The plan requires writable cgroup v2 memory.high control, cache resets, and mincore-based residency sampling. Those controls are not available in this environment without elevated privileges, so a faithful live proxy would be misleading.\n\n"
        "This artifact is therefore scoped to replay-only evidence. exp/external_validation is retained only as an auxiliary replay perturbation check on captured SQLite streams; it does not satisfy the planned live sanity-check step.\n"
    )


if __name__ == "__main__":
    main()
