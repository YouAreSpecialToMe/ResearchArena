"""Invoke a CLI agent inside a Docker container.

Each agent invocation runs in an isolated Docker container with:
  - Its own filesystem (workspace mounted as a volume)
  - GPU access via NVIDIA Container Toolkit
  - Network access for downloading datasets/models
  - Resource limits (memory, CPU, timeout)
  - Pre-installed base packages
  - API keys passed through as environment variables

The workspace directory is bind-mounted so artifacts (idea.json, results.json,
paper.tex) persist after the container exits.

Supported agents:
  - claude: Claude Code CLI
  - codex: OpenAI Codex CLI
  - aider: Aider CLI
  - custom: Any command
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()

# Default Docker image — user can override in config
DEFAULT_IMAGE = "autoresearch/agent:latest"

# Path to research guidelines template (relative to this file)
_GUIDELINES_PATH = Path(__file__).parent.parent / "templates" / "research_guidelines.md"

# Pre-authorization files written into the workspace before agent starts
CLAUDE_MD_CONTENT = """\
# AutoResearch Agent Workspace

You are running autonomously as part of a research benchmark.
You have full permission to:
- Read and write any files in this directory
- Install Python packages with pip
- Run Python scripts and experiments
- Use GPUs (CUDA is available)
- Download datasets and models from the internet
- Run shell commands

Do NOT ask for confirmation. Execute everything directly.
Save your outputs to the files specified in the task.

IMPORTANT: Read research_guidelines.md before starting. It contains critical rules
that will determine whether your paper is accepted or rejected. In particular:
- EVERY reference must be a real, verifiable publication (fake refs = auto reject)
- Include ablation studies and error bars
- Compare against real baselines
"""

CODEX_INSTRUCTIONS = """\
You are running autonomously. Execute all commands without asking for approval.
You have full filesystem, network, and GPU access in this workspace.

IMPORTANT: Read research_guidelines.md before starting. It contains critical rules
for producing publishable research. Fake references cause automatic rejection.
"""


@dataclass
class AgentResult:
    """Result of a CLI agent invocation."""
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    workspace: Path
    container_id: str | None = None
    log_files: dict[str, str] | None = None  # {"stdout": path, "stderr": path, "command": path}
    failure_category: str | None = None      # classified failure reason


# ── Failure classification ───────────────────────────────────────────────

# Patterns checked against stderr and stdout (case-insensitive) to classify
# why an agent invocation failed. Order matters — first match wins.
_FAILURE_PATTERNS: list[tuple[str, list[str]]] = [
    ("oom", [
        "out of memory", "oom", "cuda out of memory",
        "cannot allocate memory", "memory allocation failed",
        "torch.cuda.OutOfMemoryError",
    ]),
    ("timeout", [
        "timed out", "timeout", "deadline exceeded",
    ]),
    ("rate_limit", [
        "rate limit", "rate_limit", "429", "too many requests",
        "quota exceeded", "overloaded",
    ]),
    ("auth_error", [
        "authentication", "unauthorized", "401", "403",
        "invalid api key", "permission denied",
    ]),
    ("gpu_error", [
        "cuda error", "cudnn error", "nccl error",
        "no cuda gpus", "gpu not available",
    ]),
    ("import_error", [
        "modulenotfounderror", "importerror", "no module named",
    ]),
    ("crash", [
        "segmentation fault", "core dumped", "killed",
        "fatal error", "panic:",
    ]),
    ("syntax_error", [
        "syntaxerror", "indentationerror",
    ]),
    ("runtime_error", [
        "runtimeerror", "typeerror", "valueerror",
        "keyerror", "indexerror", "attributeerror",
        "filenotfounderror", "zerodivisionerror",
    ]),
    ("docker_error", [
        "docker daemon", "container failed", "image not found",
        "no such image", "pull access denied",
    ]),
    ("network_error", [
        "connectionerror", "connectionrefused", "dns resolution",
        "network unreachable", "ssl", "certificate",
    ]),
]


def classify_failure(exit_code: int, stdout: str, stderr: str) -> str | None:
    """Classify the failure reason from exit code and output.

    Returns:
        Category string, or None if the invocation succeeded (exit_code == 0).
    """
    if exit_code == 0:
        return None

    # Search stderr first (more likely to have error info), then stdout
    combined = (stderr + "\n" + stdout[-5000:]).lower()

    for category, patterns in _FAILURE_PATTERNS:
        for pattern in patterns:
            if pattern in combined:
                return category

    # Fallback based on exit code
    if exit_code == -1:
        return "timeout"
    if exit_code == 137:
        return "oom"  # killed by OOM killer
    if exit_code == 139:
        return "crash"  # segfault

    return "unknown"


def invoke_agent(
    agent_type: str,
    task: str,
    workspace: Path,
    timeout: int = 14400,
    agent_config: dict | None = None,
    readonly: bool = False,
) -> AgentResult:
    """Invoke a CLI agent inside a Docker container.

    Args:
        agent_type: "claude", "codex", "aider", or "custom"
        task: The task description / prompt for the agent
        workspace: Host directory to mount as /workspace in the container
        timeout: Max seconds before killing the container
        agent_config: Additional config (model, image, resources, etc.)
        readonly: If True, mount workspace as read-only (for reviewer agents)

    Returns:
        AgentResult with exit code, logs, and elapsed time
    """
    workspace.mkdir(parents=True, exist_ok=True)
    agent_config = agent_config or {}

    # Write permission files into workspace before container starts
    # (only for read-write invocations — reviewers don't need them)
    if not readonly:
        _setup_workspace(agent_type, workspace)

    # Build docker run command
    docker_cmd = _build_docker_command(
        agent_type, task, workspace, agent_config, readonly=readonly,
    )

    mode = "read-only" if readonly else "read-write"
    console.print(f"  Agent: {agent_type} ({mode})")
    console.print(f"  Workspace: {workspace}")
    console.print(f"  Image: {agent_config.get('docker_image', DEFAULT_IMAGE)}")
    console.print(f"  Timeout: {timeout}s")

    # Set up log directory
    # For read-only mounts, logs go to a sibling directory on the host
    # (can't write inside the read-only workspace)
    if readonly:
        log_dir = workspace.parent / f"{workspace.name}_review_logs"
    else:
        log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_prefix = f"{agent_type}_{int(time.time())}"

    # Save docker command before execution
    command_path = log_dir / f"{log_prefix}_command.txt"
    command_path.write_text(" ".join(docker_cmd))

    start = time.time()

    # Stream stdout line-by-line for all known agents to timestamp events
    # for per-tool-call duration tracking. Only use subprocess.run for custom.
    role = "reviewer" if readonly else "researcher"
    if agent_type in ("claude", "codex", "aider", "kimi", "minimax"):
        agent_result = _run_with_streaming(
            docker_cmd, workspace, log_dir, log_prefix, timeout, start, role,
        )
    else:
        agent_result = _run_simple(
            docker_cmd, workspace, log_dir, log_prefix, timeout, start, role,
        )

    return agent_result


# ── Execution strategies ─────────────────────────────────────────────────


def _run_with_streaming(
    docker_cmd: list[str],
    workspace: Path,
    log_dir: Path,
    log_prefix: str,
    timeout: int,
    start: float,
    role: str = "researcher",
) -> AgentResult:
    """Run agent with Popen, streaming stdout to timestamp each line.

    Used for Claude Code (stream-json output). Produces an events.jsonl
    file where each line is {"ts": <float>, "event": <json>} so the
    action parser can compute per-tool-call durations.
    """
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    events_path = log_dir / f"{log_prefix}_events.jsonl"

    try:
        proc = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        with open(events_path, "w") as events_file:
            while True:
                # Check timeout
                elapsed = time.time() - start
                if elapsed > timeout:
                    proc.kill()
                    proc.wait()
                    console.print(f"  [red]Agent timed out after {elapsed:.0f}s. Killing container...[/]")
                    _kill_container(workspace, role)
                    return _save_and_return(
                        exit_code=-1,
                        stdout="".join(stdout_lines),
                        stderr=f"Agent timed out after {timeout}s",
                        elapsed=elapsed,
                        workspace=workspace,
                        log_dir=log_dir,
                        log_prefix=log_prefix,
                        events_path=str(events_path),
                    )

                # Use select to read from both stdout and stderr without blocking
                readable = []
                if proc.stdout:
                    readable.append(proc.stdout)
                if proc.stderr:
                    readable.append(proc.stderr)

                if not readable:
                    break

                try:
                    ready, _, _ = select.select(readable, [], [], 1.0)
                except (ValueError, OSError):
                    break

                for stream in ready:
                    line = stream.readline()
                    if not line:
                        continue

                    if stream is proc.stdout:
                        stdout_lines.append(line)
                        ts = time.time()
                        stripped = line.strip()
                        if stripped:
                            try:
                                event = json.loads(stripped)
                                # Structured JSON event (Claude Code, Codex)
                                events_file.write(
                                    json.dumps({"ts": ts, "event": event}) + "\n"
                                )
                            except json.JSONDecodeError:
                                # Plain text line (Aider, Kimi, MiniMax)
                                events_file.write(
                                    json.dumps({"ts": ts, "line": stripped}) + "\n"
                                )
                            events_file.flush()
                    else:
                        stderr_lines.append(line)

                # Check if process has finished
                if proc.poll() is not None:
                    # Drain remaining output
                    if proc.stdout:
                        for line in proc.stdout:
                            stdout_lines.append(line)
                            stripped = line.strip()
                            if stripped:
                                ts = time.time()
                                try:
                                    event = json.loads(stripped)
                                    events_file.write(
                                        json.dumps({"ts": ts, "event": event}) + "\n"
                                    )
                                except json.JSONDecodeError:
                                    events_file.write(
                                        json.dumps({"ts": ts, "line": stripped}) + "\n"
                                    )
                    if proc.stderr:
                        for line in proc.stderr:
                            stderr_lines.append(line)
                    break

        elapsed = time.time() - start
        console.print(f"  Finished in {elapsed:.0f}s, exit code {proc.returncode}")

        return _save_and_return(
            exit_code=proc.returncode,
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
            elapsed=elapsed,
            workspace=workspace,
            log_dir=log_dir,
            log_prefix=log_prefix,
            events_path=str(events_path),
        )

    except Exception as e:
        elapsed = time.time() - start
        console.print(f"  [red]Error running agent: {e}[/]")
        return AgentResult(
            exit_code=-1,
            stdout="".join(stdout_lines),
            stderr=str(e),
            elapsed_seconds=elapsed,
            workspace=workspace,
        )


def _run_simple(
    docker_cmd: list[str],
    workspace: Path,
    log_dir: Path,
    log_prefix: str,
    timeout: int,
    start: float,
    role: str = "researcher",
) -> AgentResult:
    """Run agent with subprocess.run (simple, no streaming timestamps)."""
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        console.print(f"  Finished in {elapsed:.0f}s, exit code {result.returncode}")

        return _save_and_return(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            elapsed=elapsed,
            workspace=workspace,
            log_dir=log_dir,
            log_prefix=log_prefix,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        console.print(f"  [red]Agent timed out after {elapsed:.0f}s. Killing container...[/]")
        _kill_container(workspace, role)

        return AgentResult(
            exit_code=-1,
            stdout="",
            stderr=f"Agent timed out after {timeout}s",
            elapsed_seconds=elapsed,
            workspace=workspace,
        )


def _save_and_return(
    exit_code: int,
    stdout: str,
    stderr: str,
    elapsed: float,
    workspace: Path,
    log_dir: Path,
    log_prefix: str,
    events_path: str | None = None,
) -> AgentResult:
    """Save log files and return AgentResult."""
    stdout_path = log_dir / f"{log_prefix}_stdout.txt"
    stderr_path = log_dir / f"{log_prefix}_stderr.txt"
    command_path = log_dir / f"{log_prefix}_command.txt"
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    # command.txt is written by caller before execution — skip if exists
    if not command_path.exists():
        command_path.write_text("")

    log_files = {
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": str(command_path),
    }
    if events_path:
        log_files["events"] = events_path

    return AgentResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=elapsed,
        workspace=workspace,
        log_files=log_files,
        failure_category=classify_failure(exit_code, stdout, stderr),
    )


# ── Docker command building ──────────────────────────────────────────────


def _build_docker_command(
    agent_type: str,
    task: str,
    workspace: Path,
    config: dict,
    readonly: bool = False,
) -> list[str]:
    """Build the full `docker run` command."""

    image = config.get("docker_image", DEFAULT_IMAGE)
    role = "reviewer" if readonly else "researcher"
    container_name = f"autoresearch-{role}-{workspace.name}-{int(time.time())}"

    # Mount workspace read-only for reviewers, read-write for researchers
    mount_spec = f"{workspace.resolve()}:/workspace"
    if readonly:
        mount_spec += ":ro"

    cmd = [
        "docker", "run",
        "--rm",
        "--name", container_name,

        # ── Mount workspace ──
        "-v", mount_spec,
        "-w", "/workspace",

        # ── Resource limits ──
        "--memory", config.get("memory_limit", "32g"),
        "--cpus", str(config.get("cpus", 8)),
        "--shm-size", config.get("shm_size", "8g"),  # needed for PyTorch DataLoader
    ]

    # ── GPU access ──
    gpus = config.get("gpus", 1)
    if gpus:
        if isinstance(gpus, int):
            cmd.extend(["--gpus", str(gpus)])
        elif gpus == "all":
            cmd.extend(["--gpus", "all"])

    # ── Network ──
    # Allow network access for downloading datasets, models, API calls
    # (default Docker behavior, but being explicit)
    cmd.extend(["--network", "host"])

    # ── Environment variables ──
    # Pass through API keys so the agent CLI can authenticate
    api_key_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "WANDB_API_KEY",
        "MOONSHOT_API_KEY",      # Kimi (Moonshot AI)
        "MINIMAX_API_KEY",       # MiniMax
        "MINIMAX_GROUP_ID",      # MiniMax group ID
    ]
    # Also pass any extra env vars from config
    extra_env = config.get("env", {})

    for var in api_key_vars:
        val = os.environ.get(var)
        if val:
            cmd.extend(["-e", f"{var}={val}"])

    # For Kimi/MiniMax: map their API key to OPENAI_API_KEY inside the container
    # so Aider's --openai-api-base picks it up without leaking via CLI args
    if agent_type == "kimi":
        moonshot_key = os.environ.get("MOONSHOT_API_KEY", "")
        if moonshot_key:
            cmd.extend(["-e", f"OPENAI_API_KEY={moonshot_key}"])
    elif agent_type == "minimax":
        minimax_key = os.environ.get("MINIMAX_API_KEY", "")
        if minimax_key:
            cmd.extend(["-e", f"OPENAI_API_KEY={minimax_key}"])

    for key, val in extra_env.items():
        cmd.extend(["-e", f"{key}={val}"])

    # Non-interactive signals
    cmd.extend([
        "-e", "NONINTERACTIVE=1",
        "-e", "CI=1",
    ])

    # ── Image + agent command ──
    cmd.append(image)
    cmd.extend(_build_agent_command(agent_type, task, config))

    return cmd


def _build_agent_command(agent_type: str, task: str, config: dict) -> list[str]:
    """Build the command that runs inside the container."""

    if agent_type == "claude":
        cmd = [
            "claude",
            "--print",
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--verbose",
        ]
        if config.get("model"):
            cmd.extend(["--model", config["model"]])
        cmd.extend(["--max-turns", str(config.get("max_turns", 200))])
        cmd.extend(["--prompt", task])
        return cmd

    elif agent_type == "codex":
        cmd = [
            "codex",
            "--full-auto",
            "--quiet",
            "--json",
        ]
        if config.get("model"):
            cmd.extend(["--model", config["model"]])
        cmd.extend([task])
        return cmd

    elif agent_type == "aider":
        cmd = [
            "aider",
            "--yes-always",
            "--no-git",
            "--no-auto-commits",
            "--verbose",
            "--llm-history-file", "/workspace/logs/aider_llm_history.jsonl",
        ]
        if config.get("model"):
            cmd.extend(["--model", config["model"]])
        cmd.extend(["--message", task])
        return cmd

    elif agent_type == "kimi":
        # Kimi (Moonshot AI) — runs through Aider with OpenAI-compatible API
        # API key is passed via OPENAI_API_KEY env var in the container
        # (set by _build_docker_command via MOONSHOT_API_KEY → OPENAI_API_KEY mapping)
        model = config.get("model", "moonshot-v1-auto")
        api_base = config.get("api_base", "https://api.moonshot.cn/v1")
        cmd = [
            "aider",
            "--yes-always",
            "--no-git",
            "--no-auto-commits",
            "--verbose",
            "--llm-history-file", "/workspace/logs/aider_llm_history.jsonl",
            "--model", f"openai/{model}",
            "--openai-api-base", api_base,
        ]
        cmd.extend(["--message", task])
        return cmd

    elif agent_type == "minimax":
        # MiniMax — runs through Aider with OpenAI-compatible API
        # API key is passed via OPENAI_API_KEY env var in the container
        model = config.get("model", "MiniMax-Text-01")
        api_base = config.get("api_base", "https://api.minimax.chat/v1")
        cmd = [
            "aider",
            "--yes-always",
            "--no-git",
            "--no-auto-commits",
            "--verbose",
            "--llm-history-file", "/workspace/logs/aider_llm_history.jsonl",
            "--model", f"openai/{model}",
            "--openai-api-base", api_base,
        ]
        cmd.extend(["--message", task])
        return cmd

    elif agent_type == "custom":
        import shlex
        template = config.get("command", 'echo "{task}"')
        cmd_str = (
            template
            .replace("{task}", shlex.quote(task))
            .replace("{workspace}", shlex.quote("/workspace"))
            .replace("{model}", shlex.quote(config.get("model", "")))
        )
        return ["bash", "-c", cmd_str]

    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Supported: claude, codex, aider, kimi, minimax, custom"
        )


# ── Workspace setup ──────────────────────────────────────────────────────


def _setup_workspace(agent_type: str, workspace: Path):
    """Write permission/config files and research guidelines into workspace."""

    # Copy research guidelines into every workspace
    guidelines_dest = workspace / "research_guidelines.md"
    if not guidelines_dest.exists() and _GUIDELINES_PATH.exists():
        shutil.copy2(_GUIDELINES_PATH, guidelines_dest)

    if agent_type == "claude":
        claude_md = workspace / "CLAUDE.md"
        if not claude_md.exists():
            claude_md.write_text(CLAUDE_MD_CONTENT)

    elif agent_type == "codex":
        codex_dir = workspace / ".codex"
        codex_dir.mkdir(exist_ok=True)
        instructions = codex_dir / "instructions.md"
        if not instructions.exists():
            instructions.write_text(CODEX_INSTRUCTIONS)


# ── Container management ────────────────────────────────────────────────


def _kill_container(workspace: Path, role: str = "researcher"):
    """Find and kill running containers for this workspace and role."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name=autoresearch-{role}-{workspace.name}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for container_id in result.stdout.strip().split("\n"):
            if container_id:
                subprocess.run(
                    ["docker", "kill", container_id],
                    capture_output=True,
                    timeout=10,
                )
                console.print(f"  Killed container {container_id[:12]}")
    except Exception as e:
        console.print(f"  [yellow]Failed to kill container: {e}[/]")
