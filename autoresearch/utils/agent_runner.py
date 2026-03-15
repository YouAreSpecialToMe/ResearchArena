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

import os
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


def invoke_agent(
    agent_type: str,
    task: str,
    workspace: Path,
    timeout: int = 14400,
    agent_config: dict | None = None,
) -> AgentResult:
    """Invoke a CLI agent inside a Docker container.

    Args:
        agent_type: "claude", "codex", "aider", or "custom"
        task: The task description / prompt for the agent
        workspace: Host directory to mount as /workspace in the container
        timeout: Max seconds before killing the container
        agent_config: Additional config (model, image, resources, etc.)

    Returns:
        AgentResult with exit code, logs, and elapsed time
    """
    workspace.mkdir(parents=True, exist_ok=True)
    agent_config = agent_config or {}

    # Write permission files into workspace before container starts
    _setup_workspace(agent_type, workspace)

    # Build docker run command
    docker_cmd = _build_docker_command(agent_type, task, workspace, agent_config)

    console.print(f"  Agent: {agent_type}")
    console.print(f"  Workspace: {workspace}")
    console.print(f"  Image: {agent_config.get('docker_image', DEFAULT_IMAGE)}")
    console.print(f"  Timeout: {timeout}s")

    start = time.time()

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start

        # Save logs to workspace
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        (log_dir / "agent_stdout.txt").write_text(result.stdout)
        (log_dir / "agent_stderr.txt").write_text(result.stderr)
        (log_dir / "docker_command.txt").write_text(" ".join(docker_cmd))

        console.print(f"  Finished in {elapsed:.0f}s, exit code {result.returncode}")

        return AgentResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            elapsed_seconds=elapsed,
            workspace=workspace,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        console.print(f"  [red]Agent timed out after {elapsed:.0f}s. Killing container...[/]")

        # Find and kill the container
        _kill_container(workspace)

        return AgentResult(
            exit_code=-1,
            stdout="",
            stderr=f"Agent timed out after {timeout}s",
            elapsed_seconds=elapsed,
            workspace=workspace,
        )


# ── Docker command building ──────────────────────────────────────────────


def _build_docker_command(
    agent_type: str,
    task: str,
    workspace: Path,
    config: dict,
) -> list[str]:
    """Build the full `docker run` command."""

    image = config.get("docker_image", DEFAULT_IMAGE)
    container_name = f"autoresearch-{workspace.name}-{int(time.time())}"

    cmd = [
        "docker", "run",
        "--rm",                             # auto-remove container on exit
        "--name", container_name,

        # ── Mount workspace ──
        "-v", f"{workspace.resolve()}:/workspace",
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
    ]
    # Also pass any extra env vars from config
    extra_env = config.get("env", {})

    for var in api_key_vars:
        val = os.environ.get(var)
        if val:
            cmd.extend(["-e", f"{var}={val}"])

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
        ]
        if config.get("model"):
            cmd.extend(["--model", config["model"]])
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
            f"Supported: claude, codex, aider, custom"
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


def _kill_container(workspace: Path):
    """Find and kill any running container for this workspace."""
    try:
        # List containers with our naming pattern
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name=autoresearch-{workspace.name}"],
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
