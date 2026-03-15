"""Parse CLI agent stdout into structured sub-actions.

Each agent CLI outputs tool calls / actions in its own format. This module
extracts them into a uniform list of SubAction records so the tracker can
log what the agent actually did during a pipeline stage.

Supported agents:
  - Claude Code (stream-json): structured JSON lines with full detail
  - Codex: best-effort parsing of stdout patterns
  - Aider: best-effort parsing of stdout patterns
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class SubAction:
    """One tool call or discrete action taken by the agent."""
    tool: str                          # e.g. "Read", "Write", "Bash", "WebSearch"
    input_summary: str = ""            # what was passed to the tool
    output_summary: str = ""           # what the tool returned (truncated)
    reasoning: str = ""                # LLM's thinking text before this tool call
    duration_seconds: float | None = None  # wall time for this tool call
    tokens: dict | None = None         # {"input": N, "output": N} for the turn
    error: str | None = None           # error message if the tool call failed
    files_affected: list[str] | None = None  # files this tool call touched (from workspace diff)

    def to_dict(self) -> dict:
        d: dict = {"tool": self.tool}
        if self.input_summary:
            d["input"] = self.input_summary
        if self.output_summary:
            d["output"] = self.output_summary
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.duration_seconds is not None:
            d["duration_seconds"] = round(self.duration_seconds, 2)
        if self.tokens:
            d["tokens"] = self.tokens
        if self.error:
            d["error"] = self.error
        if self.files_affected:
            d["files_affected"] = self.files_affected
        return d


def parse_agent_stdout(agent_type: str, stdout: str, events_path: str | None = None) -> list[SubAction]:
    """Parse agent stdout into sub-actions based on agent type.

    Args:
        agent_type: "claude", "codex", "aider", or "custom"
        stdout: Raw stdout from the agent (used for codex/aider/fallback)
        events_path: Path to timestamped events JSONL file (Claude Code only).
                     If provided, used instead of stdout for richer parsing.
    """
    if not stdout and not events_path:
        return []

    if agent_type == "claude":
        return _parse_claude_events(events_path, stdout)
    elif agent_type == "codex":
        return _parse_codex_stdout(stdout)
    elif agent_type == "aider":
        return _parse_aider_stdout(stdout)
    else:
        return _parse_generic_stdout(stdout)


# ── Claude Code (stream-json with timestamps) ────────────────────────


def _parse_claude_events(events_path: str | None, stdout: str) -> list[SubAction]:
    """Parse Claude Code stream-json output into rich sub-actions.

    If events_path is provided, reads timestamped JSONL where each line is:
        {"ts": <unix_timestamp>, "event": <stream_event>}

    Otherwise falls back to parsing raw stdout lines (no timestamps).

    Claude Code stream-json emits these event types:
        stream_event.event.type = "message_start"       → new turn begins
        stream_event.event.type = "content_block_start"  → text or tool_use block
        stream_event.event.type = "content_block_delta"  → incremental content
        stream_event.event.type = "content_block_stop"   → block done
        stream_event.event.type = "message_delta"        → usage + stop_reason
        stream_event.event.type = "message_stop"         → turn ends

    We group events into turns. Each turn may contain:
        - Text blocks (LLM reasoning)
        - Tool use blocks (tool calls)
        - Token usage (from message_delta)
    """
    # Load events — either timestamped file or raw stdout lines
    events = _load_events(events_path, stdout)
    if not events:
        return []

    # Group into turns and extract sub-actions
    return _events_to_sub_actions(events)


def _load_events(events_path: str | None, stdout: str) -> list[dict]:
    """Load events from timestamped JSONL file or raw stdout."""
    events = []

    if events_path:
        try:
            with open(events_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except FileNotFoundError:
            pass

    if events:
        return events

    # Fallback: parse raw stdout, no timestamps
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # Wrap in consistent format (no timestamp)
            if obj.get("type") == "stream_event":
                events.append({"ts": None, "event": obj})
            elif "type" in obj:
                # Might be a direct event or different format
                events.append({"ts": None, "event": obj})
        except json.JSONDecodeError:
            pass

    return events


def _events_to_sub_actions(events: list[dict]) -> list[SubAction]:
    """Convert a stream of timestamped events into SubAction records.

    Strategy: walk through events, accumulating state per turn:
    - Collect text deltas as "reasoning"
    - When a tool_use block starts, create a SubAction
    - Attach the accumulated reasoning to it
    - When message_delta arrives, capture per-turn tokens
    - Use timestamps to compute durations
    """
    actions: list[SubAction] = []

    # Current turn state
    current_reasoning_chunks: list[str] = []
    current_tool_name: str | None = None
    current_tool_input_chunks: list[str] = []
    current_tool_id: str | None = None
    current_block_type: str | None = None  # "text" or "tool_use"
    turn_tokens: dict | None = None
    tool_start_ts: float | None = None

    # Pending sub-actions in current turn (waiting for token info)
    turn_actions: list[SubAction] = []

    # Map tool_use_id → SubAction for matching results
    pending_by_id: dict[str, SubAction] = {}

    for entry in events:
        ts = entry.get("ts")
        raw_event = entry.get("event", {})

        # Handle both wrapped (stream_event) and direct event formats
        if raw_event.get("type") == "stream_event":
            inner = raw_event.get("event", {})
        else:
            inner = raw_event

        event_type = inner.get("type", "")

        # ── message_start: new turn ──
        if event_type == "message_start":
            current_reasoning_chunks = []
            turn_actions = []
            turn_tokens = None

        # ── content_block_start ──
        elif event_type == "content_block_start":
            block = inner.get("content_block", {})
            current_block_type = block.get("type")

            if current_block_type == "tool_use":
                current_tool_name = block.get("name", "unknown")
                current_tool_id = block.get("id", "")
                current_tool_input_chunks = []
                tool_start_ts = ts
            elif current_block_type == "text":
                pass  # will accumulate via deltas

        # ── content_block_delta ──
        elif event_type == "content_block_delta":
            delta = inner.get("delta", {})
            delta_type = delta.get("type", "")

            if delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    current_reasoning_chunks.append(text)

            elif delta_type == "input_json_delta":
                chunk = delta.get("partial_json", "")
                if chunk:
                    current_tool_input_chunks.append(chunk)

        # ── content_block_stop ──
        elif event_type == "content_block_stop":
            if current_block_type == "tool_use" and current_tool_name:
                # Parse accumulated tool input
                raw_input = "".join(current_tool_input_chunks)
                try:
                    tool_input = json.loads(raw_input)
                except json.JSONDecodeError:
                    tool_input = {}

                # Build the sub-action
                reasoning_text = "".join(current_reasoning_chunks).strip()

                sub = SubAction(
                    tool=current_tool_name,
                    input_summary=_summarize_tool_input(current_tool_name, tool_input),
                    reasoning=_truncate(reasoning_text, 500) if reasoning_text else "",
                )

                if current_tool_id:
                    pending_by_id[current_tool_id] = sub

                turn_actions.append(sub)
                actions.append(sub)

                # Reset reasoning for next tool call in same turn
                current_reasoning_chunks = []
                current_tool_name = None
                current_tool_id = None

            current_block_type = None

        # ── message_delta: per-turn tokens ──
        elif event_type == "message_delta":
            usage = inner.get("usage", {})
            if usage:
                turn_tokens = {
                    "input": usage.get("input_tokens", 0),
                    "output": usage.get("output_tokens", 0),
                }

        # ── message_stop: finalize turn ──
        elif event_type == "message_stop":
            # Distribute turn tokens across tool calls in this turn
            if turn_tokens and turn_actions:
                n = len(turn_actions)
                for sub in turn_actions:
                    sub.tokens = {
                        "input": turn_tokens["input"] // n,
                        "output": turn_tokens["output"] // n,
                    }
                # Give remainder to last action
                remainder_in = turn_tokens["input"] - (turn_tokens["input"] // n) * n
                remainder_out = turn_tokens["output"] - (turn_tokens["output"] // n) * n
                if remainder_in or remainder_out:
                    turn_actions[-1].tokens["input"] += remainder_in
                    turn_actions[-1].tokens["output"] += remainder_out
            elif turn_tokens and not turn_actions:
                # Turn with only text (no tool calls) — reasoning-only turn
                reasoning_text = "".join(current_reasoning_chunks).strip()
                if reasoning_text:
                    actions.append(SubAction(
                        tool="(thinking)",
                        reasoning=_truncate(reasoning_text, 500),
                        tokens=turn_tokens,
                    ))

            turn_actions = []
            turn_tokens = None
            current_reasoning_chunks = []

        # ── tool result (from tool execution) ──
        # These appear as separate messages with role="tool"
        # In stream-json they may appear as content events in a tool message
        elif event_type == "tool":
            tool_id = inner.get("tool_use_id", "")
            sub = pending_by_id.pop(tool_id, None)
            if sub:
                content = inner.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        b.get("text", "") for b in content
                        if b.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)

                is_error = inner.get("is_error", False)
                if is_error:
                    sub.error = _truncate(str(content), 300)
                else:
                    sub.output_summary = _truncate(str(content), 300)

                # Compute duration if timestamps available
                if ts is not None and tool_start_ts is not None:
                    sub.duration_seconds = ts - tool_start_ts
                    tool_start_ts = None

    return actions


# ── Codex ─────────────────────────────────────────────────────────────


def _parse_codex_stdout(stdout: str) -> list[SubAction]:
    """Parse Codex CLI stdout for tool call patterns.

    Codex outputs tool calls in a structured format. We look for patterns like:
      [tool_call] shell: <command>
      [tool_call] file_write: <path>
      [tool_call] file_read: <path>
    """
    actions = []

    tool_call_pattern = re.compile(
        r'\[tool_call\]\s+(\w+):\s*(.*)', re.IGNORECASE
    )
    running_pattern = re.compile(r'Running:\s+(.*)')
    reading_pattern = re.compile(r'Reading\s+(.*)')
    writing_pattern = re.compile(r'Writing\s+(.*)')

    for line in stdout.splitlines():
        line = line.strip()

        m = tool_call_pattern.match(line)
        if m:
            actions.append(SubAction(
                tool=_normalize_tool_name(m.group(1)),
                input_summary=_truncate(m.group(2).strip(), 200),
            ))
            continue

        m = running_pattern.match(line)
        if m:
            actions.append(SubAction(
                tool="Bash",
                input_summary=_truncate(m.group(1).strip(), 200),
            ))
            continue

        m = reading_pattern.match(line)
        if m:
            actions.append(SubAction(
                tool="Read",
                input_summary=_truncate(m.group(1).strip(), 200),
            ))
            continue

        m = writing_pattern.match(line)
        if m:
            actions.append(SubAction(
                tool="Write",
                input_summary=_truncate(m.group(1).strip(), 200),
            ))
            continue

    return actions


# ── Aider ─────────────────────────────────────────────────────────────


def _parse_aider_stdout(stdout: str) -> list[SubAction]:
    """Parse Aider CLI stdout for action patterns."""
    actions = []

    applied_pattern = re.compile(r'Applied edit to\s+(.*)')
    running_pattern = re.compile(r'>\s*Running\s+(.*)')
    added_pattern = re.compile(r'Added\s+(.*?)\s+to the chat')

    for line in stdout.splitlines():
        line = line.strip()

        m = running_pattern.match(line)
        if m:
            actions.append(SubAction(
                tool="Bash",
                input_summary=_truncate(m.group(1).strip(), 200),
            ))
            continue

        m = applied_pattern.match(line)
        if m:
            actions.append(SubAction(
                tool="Edit",
                input_summary=_truncate(m.group(1).strip(), 200),
            ))
            continue

        m = added_pattern.match(line)
        if m:
            actions.append(SubAction(
                tool="Read",
                input_summary=_truncate(m.group(1).strip(), 200),
            ))
            continue

    return actions


# ── Generic fallback ──────────────────────────────────────────────────


def _parse_generic_stdout(stdout: str) -> list[SubAction]:
    """Best-effort parsing for unknown agent types."""
    actions = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("$ ") or line.startswith("+ "):
            actions.append(SubAction(
                tool="Bash",
                input_summary=_truncate(line[2:], 200),
            ))
    return actions


# ── Helpers ───────────────────────────────────────────────────────────


def annotate_with_workspace_diff(sub_actions: list[SubAction], workspace_diff: dict | None):
    """Cross-reference sub-actions with workspace diff to annotate file changes.

    For Write/Edit/Bash tool calls, matches the tool's target file against the
    workspace diff to confirm the file was actually created/modified. For Bash
    commands, any files created by the command (e.g. running a training script
    that produces checkpoints) are attributed to that Bash sub-action.

    Files in the diff that no sub-action claims are added as a synthetic
    "(side_effect)" sub-action at the end.
    """
    if not workspace_diff or not sub_actions:
        return

    # Build set of all changed files
    created = {f["path"] for f in workspace_diff.get("created", [])}
    modified = {f["path"] for f in workspace_diff.get("modified", [])}
    deleted = set(workspace_diff.get("deleted", []))
    all_changed = created | modified | deleted

    claimed: set[str] = set()

    for sub in sub_actions:
        tool = sub.tool.lower()
        inp = sub.input_summary

        if tool in ("write", "edit", "read"):
            # Extract file path from input_summary
            # Write: "experiment.py (42 lines)" → "experiment.py"
            # Edit: "train.py: old_text → ..." → "train.py"
            # Read: "research_guidelines.md" → "research_guidelines.md"
            file_path = inp.split(" (")[0].split(":")[0].strip()
            if file_path:
                # Match against changed files (handle both exact and basename)
                matches = _match_file(file_path, all_changed)
                if matches:
                    sub.files_affected = sorted(matches)
                    claimed.update(matches)

        elif tool == "bash":
            # Bash commands can create files as side effects.
            # We can't know exactly which files — we'll attribute unclaimed
            # files to Bash commands in a second pass below.
            pass

    # Second pass: attribute unclaimed files to Bash sub-actions
    # Heuristic: files created between consecutive Bash calls are attributed
    # to the preceding Bash call. If only one Bash call, it gets everything.
    unclaimed = all_changed - claimed
    if unclaimed:
        bash_actions = [s for s in sub_actions if s.tool.lower() == "bash"]
        if bash_actions:
            # Simple attribution: last Bash command gets all unclaimed files
            # (most common case: `python train.py` produces results + checkpoints)
            last_bash = bash_actions[-1]
            if last_bash.files_affected:
                last_bash.files_affected.extend(sorted(unclaimed))
            else:
                last_bash.files_affected = sorted(unclaimed)
            claimed.update(unclaimed)
            unclaimed = set()

    # Any files still unclaimed get a synthetic sub-action
    if unclaimed:
        sub_actions.append(SubAction(
            tool="(side_effect)",
            input_summary=f"{len(unclaimed)} file(s) changed outside tracked tool calls",
            files_affected=sorted(unclaimed),
        ))


def _match_file(target: str, changed_files: set[str]) -> list[str]:
    """Match a file path from a tool call against the set of changed files.

    Handles cases like:
      - Exact match: "experiment.py" in changed_files
      - Basename match: "/workspace/experiment.py" → "experiment.py"
      - Prefix match: "figures/" → all files under figures/
    """
    matches = []

    # Normalize: strip leading /workspace/ or ./
    target = target.lstrip("./")
    if target.startswith("workspace/"):
        target = target[len("workspace/"):]

    for f in changed_files:
        if f == target:
            matches.append(f)
        elif f.endswith("/" + target) or f.endswith(target):
            matches.append(f)
        elif target.endswith("/") and f.startswith(target):
            matches.append(f)

    return matches



    """Create a brief human-readable summary of a tool's input."""
    if not tool_input:
        return ""

    name = tool_name.lower()

    if name == "read":
        return tool_input.get("file_path", "")

    if name == "write":
        path = tool_input.get("file_path", "")
        content = tool_input.get("content", "")
        lines = content.count("\n") + 1 if content else 0
        return f"{path} ({lines} lines)"

    if name == "edit":
        path = tool_input.get("file_path", "")
        old = _truncate(tool_input.get("old_string", ""), 50)
        return f"{path}: {old} → ..."

    if name == "bash":
        return _truncate(tool_input.get("command", ""), 200)

    if name == "glob":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        return f"{pattern}" + (f" in {path}" if path else "")

    if name == "grep":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        return f"/{pattern}/" + (f" in {path}" if path else "")

    if name in ("websearch", "web_search"):
        return tool_input.get("query", "")

    if name in ("webfetch", "web_fetch"):
        return tool_input.get("url", "")

    # Fallback: show first key=value pairs
    parts = []
    for k, v in list(tool_input.items())[:3]:
        v_str = _truncate(str(v), 60)
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)


def _normalize_tool_name(name: str) -> str:
    mapping = {
        "shell": "Bash",
        "file_read": "Read",
        "file_write": "Write",
        "file_edit": "Edit",
        "search": "WebSearch",
    }
    return mapping.get(name.lower(), name)


def _truncate(s: str, max_len: int) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s
