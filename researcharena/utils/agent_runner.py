"""Invoke a CLI agent, either in a container or locally on the host.

Supports two runtime modes (set via config agent.runtime):
  - "docker" (default): runs in a Docker/Podman container
  - "local": runs directly on the host with a per-workspace virtualenv

Local mode creates an isolated virtualenv for each workspace so agents
can pip install packages without conflicting with each other or the host.

Supported agents:
  - claude: Claude Code CLI
  - codex: OpenAI Codex CLI
  - kimi: Kimi Code CLI (Moonshot AI)
  - minimax: Mini-Agent CLI (MiniMax)
  - custom: Any command
"""

from __future__ import annotations

import json
import os
import select
import shlex
import shutil
import subprocess
import time
import venv
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()

# Default Docker image — user can override in config
DEFAULT_IMAGE = "researcharena/agent:latest"

# Paths to guideline templates (relative to this file)
# Domain-specific templates live in templates/{domain}/.
_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# Valid domains that have their own template subdirectory
_DOMAINS_WITH_TEMPLATES = {"ml", "systems", "databases", "pl", "theory", "security"}


def _get_template_path(filename: str, domain: str = "ml") -> Path:
    """Return the path to a guideline template, domain-specific if available."""
    if domain in _DOMAINS_WITH_TEMPLATES:
        domain_path = _TEMPLATES_DIR / domain / filename
        if domain_path.exists():
            return domain_path
    # Fallback to ml templates
    return _TEMPLATES_DIR / "ml" / filename

# Pre-authorization files written into the workspace before agent starts
def _build_agent_instructions(agent_type: str, platform: str = "gpu") -> str:
    """Generate agent instruction text based on platform (GPU/CPU).

    All agents get the same content — only the filename differs
    (CLAUDE.md, .codex/instructions.md, AGENTS.md, AGENT_INSTRUCTIONS.md).
    """
    is_gpu = (platform == "gpu")
    gpu_line = "- Use GPUs (CUDA is available)\n" if is_gpu else ""
    cpu_note = (
        "\nNOTE: No GPU is available. All computation runs on CPU only.\n"
        "Design your experiments accordingly — prefer analytical, algorithmic,\n"
        "or systems-level experiments that don't require GPU compute.\n"
    ) if not is_gpu else ""

    return (
        "# ResearchArena Agent Workspace\n\n"
        "You are a researcher conducting end-to-end research autonomously.\n"
        "Your goal is to advance scientific understanding — find a meaningful problem,\n"
        "investigate it rigorously, and communicate your findings in a research paper.\n\n"
        "The research is conducted in stages. Each stage has a dedicated guideline\n"
        "file in this workspace — read it before starting each stage:\n\n"
        "  Stage 1 — IDEATION:     Read idea_guidelines.md\n"
        "  Stage 2 — EXPERIMENTS:  Read experiment_guidelines.md\n"
        "  Stage 3 — PAPER:        Read paper_writing_guidelines.md\n\n"
        "You will receive a task prompt telling you which stage you are in and\n"
        "what output is expected. Follow the corresponding guideline closely.\n\n"
        "You have full permission to:\n"
        "- Read and write any files in this directory\n"
        "- Install Python packages with pip (use the workspace .venv if present)\n"
        "- Run Python scripts and experiments\n"
        f"{gpu_line}"
        "- Download datasets and models from the internet\n"
        "- Search the web (arXiv, Semantic Scholar, Google Scholar)\n"
        "- Run shell commands\n"
        f"{cpu_note}\n"
        "Do NOT ask for confirmation. Execute everything directly.\n\n"
        "IMPORTANT — scientific integrity:\n"
        "- EVERY reference must be a real, verifiable publication\n"
        "- ALL experimental results must come from actually running code\n"
        "- Include ablation studies and error bars\n"
        "- Compare against real baselines\n"
    )


@dataclass
class AgentResult:
    """Result of a CLI agent invocation."""
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    workspace: Path
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
    """Invoke a CLI agent, either in a container or locally.

    When agent_config["runtime"] is "local", the agent runs directly on
    the host with a per-workspace virtualenv. Otherwise it runs in Docker.

    Args:
        agent_type: "claude", "codex", "kimi", "minimax", or "custom"
        task: The task description / prompt for the agent
        workspace: Directory for agent artifacts
        timeout: Max seconds before killing the agent
        agent_config: Additional config (model, image, resources, runtime, etc.)
        readonly: If True, workspace is read-only (for reviewer agents)

    Returns:
        AgentResult with exit code, logs, and elapsed time
    """
    workspace.mkdir(parents=True, exist_ok=True)
    agent_config = agent_config or {}

    if not readonly:
        platform = "cpu" if agent_config.get("gpus", 1) == 0 else "gpu"
        domain = agent_config.get("domain", "ml")
        _setup_workspace(agent_type, workspace, platform=platform, domain=domain)

    runtime = agent_config.get("runtime", "docker")

    if runtime == "local":
        return _invoke_local(agent_type, task, workspace, timeout, agent_config, readonly)
    else:
        return _invoke_docker(agent_type, task, workspace, timeout, agent_config, readonly)


# ── Local runtime ────────────────────────────────────────────────────────


def _invoke_local(
    agent_type: str,
    task: str,
    workspace: Path,
    timeout: int,
    agent_config: dict,
    readonly: bool,
) -> AgentResult:
    """Run the agent CLI directly on the host with a per-workspace virtualenv."""

    # Create a virtualenv for this workspace (inherits system packages)
    venv_dir = workspace / ".venv"
    if not venv_dir.exists():
        console.print(f"  Creating virtualenv at {venv_dir}...")
        venv.create(str(venv_dir), with_pip=True, system_site_packages=True)

    # Build the agent command
    cmd = _build_agent_command(agent_type, task, agent_config, workspace_path=str(workspace.resolve()))

    # Set up environment: activate venv, set CUDA devices
    env = os.environ.copy()
    venv_bin = venv_dir / "bin"
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    env["NONINTERACTIVE"] = "1"
    env["CI"] = "1"

    # GPU assignment (or explicit block for CPU platform)
    cuda_devices = agent_config.get("cuda_devices")
    if cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    elif agent_config.get("gpus", 1) == 0:
        # CPU platform: explicitly block GPU access on the host
        env["CUDA_VISIBLE_DEVICES"] = ""

    mode = "read-only" if readonly else "read-write"
    console.print(f"  Agent: {agent_type} ({mode})")
    console.print(f"  Workspace: {workspace}")
    console.print(f"  Runtime: local (virtualenv)")
    console.print(f"  Timeout: {timeout}s")

    # Set up logging
    if readonly:
        log_dir = workspace.parent / f"{workspace.name}_review_logs"
    else:
        log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_prefix = f"{agent_type}_{int(time.time())}"

    command_path = log_dir / f"{log_prefix}_command.txt"
    command_path.write_text(" ".join(cmd))

    start = time.time()
    cwd = workspace.resolve()

    if agent_type in ("claude", "codex", "kimi", "minimax"):
        return _run_with_streaming(
            cmd, workspace, log_dir, log_prefix, timeout, start,
            role="reviewer" if readonly else "researcher",
            cwd=cwd, env=env,
        )
    else:
        return _run_simple(
            cmd, workspace, log_dir, log_prefix, timeout, start,
            role="reviewer" if readonly else "researcher",
            cwd=cwd, env=env,
        )


# ── Docker runtime ───────────────────────────────────────────────────────


def _invoke_docker(
    agent_type: str,
    task: str,
    workspace: Path,
    timeout: int,
    agent_config: dict,
    readonly: bool,
) -> AgentResult:
    """Run the agent inside a Docker/Podman container."""

    docker_cmd = _build_docker_command(
        agent_type, task, workspace, agent_config, readonly=readonly,
    )

    mode = "read-only" if readonly else "read-write"
    console.print(f"  Agent: {agent_type} ({mode})")
    console.print(f"  Workspace: {workspace}")
    console.print(f"  Image: {agent_config.get('docker_image', DEFAULT_IMAGE)}")
    console.print(f"  Timeout: {timeout}s")

    if readonly:
        log_dir = workspace.parent / f"{workspace.name}_review_logs"
    else:
        log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_prefix = f"{agent_type}_{int(time.time())}"

    command_path = log_dir / f"{log_prefix}_command.txt"
    command_path.write_text(" ".join(docker_cmd))

    start = time.time()
    role = "reviewer" if readonly else "researcher"

    if agent_type in ("claude", "codex", "kimi", "minimax"):
        return _run_with_streaming(
            docker_cmd, workspace, log_dir, log_prefix, timeout, start, role,
        )
    else:
        return _run_simple(
            docker_cmd, workspace, log_dir, log_prefix, timeout, start, role,
        )


# ── Execution strategies ─────────────────────────────────────────────────


def _run_with_streaming(
    cmd: list[str],
    workspace: Path,
    log_dir: Path,
    log_prefix: str,
    timeout: int,
    start: float,
    role: str = "researcher",
    cwd: Path | None = None,
    env: dict | None = None,
) -> AgentResult:
    """Run agent with Popen, streaming stdout to timestamp each line.

    Produces an events.jsonl file where each line is {"ts": <float>, "event": <json>}
    so the action parser can compute per-tool-call durations.
    """
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    events_path = log_dir / f"{log_prefix}_events.jsonl"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
        )

        with open(events_path, "w") as events_file:
            while True:
                # Check timeout
                elapsed = time.time() - start
                if elapsed > timeout:
                    proc.kill()
                    proc.wait()
                    console.print(f"  [red]Agent timed out after {elapsed:.0f}s.[/]")
                    if not cwd:  # docker mode
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
                                events_file.write(
                                    json.dumps({"ts": ts, "event": event}) + "\n"
                                )
                            except json.JSONDecodeError:
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
    cmd: list[str],
    workspace: Path,
    log_dir: Path,
    log_prefix: str,
    timeout: int,
    start: float,
    role: str = "researcher",
    cwd: Path | None = None,
    env: dict | None = None,
) -> AgentResult:
    """Run agent with subprocess.run (simple, no streaming timestamps)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
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
        console.print(f"  [red]Agent timed out after {elapsed:.0f}s.[/]")
        if not cwd:  # docker mode
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
    container_name = f"researcharena-{role}-{workspace.name}-{int(time.time())}"

    mount_spec = f"{workspace.resolve()}:/workspace"
    if readonly:
        mount_spec += ":ro"

    runtime = _container_runtime()
    cmd = [
        runtime, "run",
        "--rm",
        "--name", container_name,
        "-v", mount_spec,
        "-w", "/workspace",
        "--memory", config.get("memory_limit", "32g"),
        "--cpus", str(config.get("cpus", 8)),
        "--shm-size", config.get("shm_size", "8g"),
    ]

    # Podman rootless needs --userns=host when no subuid mappings exist
    if _is_podman():
        cmd.extend(["--userns=host"])

    # GPU access (skip entirely for CPU platform)
    is_podman = _is_podman()
    cuda_devices = config.get("cuda_devices")
    gpus = config.get("gpus", 1)
    if gpus and gpus != 0:
        if is_podman:
            cmd.extend(["--device", "nvidia.com/gpu=all"])
            if cuda_devices:
                cmd.extend(["-e", f"NVIDIA_VISIBLE_DEVICES={cuda_devices}"])
        elif cuda_devices:
            cmd.extend(["--gpus", f'"device={cuda_devices}"'])
        elif isinstance(gpus, int):
            cmd.extend(["--gpus", str(gpus)])
        elif gpus == "all":
            cmd.extend(["--gpus", "all"])

    cmd.extend(["--network", "host"])

    # Mount CLI agent binaries from host (if not in image)
    home = Path.home()
    # Map agent_type to (binary_name_to_find, binary_name_in_container)
    cli_binaries = {
        "claude": ("claude", "claude"),
        "codex": ("codex", "codex"),
        "kimi": ("kimi", "kimi"),
        "minimax": ("mini-agent", "mini-agent"),
    }
    bin_search, bin_name = cli_binaries.get(agent_type, (agent_type, agent_type))
    host_bin = shutil.which(bin_search)
    if host_bin:
        host_bin = str(Path(host_bin).resolve())
        cmd.extend(["-v", f"{host_bin}:/usr/local/bin/{bin_name}:ro"])

    # Mount CLI agent auth & memory
    auth_mounts = {
        "claude": home / ".claude",
        "codex": home / ".codex",
        "kimi": home / ".kimi",
        "minimax": home / ".mini-agent",
    }
    auth_dir = auth_mounts.get(agent_type)
    if auth_dir and auth_dir.exists():
        mount_mode = "ro" if readonly else "rw"
        cmd.extend(["-v", f"{auth_dir}:/root/{auth_dir.name}:{mount_mode}"])

    config_files = {
        "claude": home / ".claude.json",
        "codex": home / ".codex.json",
    }
    config_file = config_files.get(agent_type)
    if config_file and config_file.exists():
        mount_mode = "ro" if readonly else "rw"
        cmd.extend(["-v", f"{config_file}:/root/{config_file.name}:{mount_mode}"])

    # Environment variables
    api_key_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "WANDB_API_KEY",
        "MOONSHOT_API_KEY",
        "MINIMAX_API_KEY",
        "MINIMAX_GROUP_ID",
    ]
    extra_env = config.get("env", {})

    for var in api_key_vars:
        val = os.environ.get(var)
        if val:
            cmd.extend(["-e", f"{var}={val}"])

    for key, val in extra_env.items():
        cmd.extend(["-e", f"{key}={val}"])

    cmd.extend([
        "-e", "NONINTERACTIVE=1",
        "-e", "CI=1",
    ])

    cmd.append(image)
    cmd.extend(_build_agent_command(agent_type, task, config))

    return cmd


def _build_agent_command(agent_type: str, task: str, config: dict, workspace_path: str = "/workspace") -> list[str]:
    """Build the command that runs inside the container (or locally)."""

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
        cmd.extend(["-p", task])
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

    elif agent_type == "kimi":
        cmd = [
            "kimi",
            "--print",
            "--output-format", "stream-json",
            "--verbose",
        ]
        if config.get("model"):
            cmd.extend(["--model", config["model"]])
        cmd.extend(["--max-steps-per-turn", str(config.get("max_turns", 200))])
        cmd.extend(["--prompt", task])
        return cmd

    elif agent_type == "minimax":
        cmd = [
            "mini-agent",
            "--task", task,
            "--workspace", workspace_path,
        ]
        return cmd

    elif agent_type == "custom":
        template = config.get("command", 'echo "{task}"')
        cmd_str = (
            template
            .replace("{task}", shlex.quote(task))
            .replace("{workspace}", shlex.quote(workspace_path))
            .replace("{model}", shlex.quote(config.get("model", "")))
        )
        return ["bash", "-c", cmd_str]

    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Supported: claude, codex, kimi, minimax, custom"
        )


# ── Workspace setup ──────────────────────────────────────────────────────


def _setup_workspace(agent_type: str, workspace: Path, platform: str = "gpu", domain: str = "ml"):
    """Write permission/config files and research guidelines into workspace."""

    # Copy domain-specific guideline templates
    for filename in [
        "idea_guidelines.md",
        "experiment_guidelines.md",
        "paper_writing_guidelines.md",
    ]:
        dest = workspace / filename
        if not dest.exists():
            src = _get_template_path(filename, domain)
            if src.exists():
                shutil.copy2(src, dest)

    # Write agent-specific instruction file
    # Each agent reads a different filename for its system prompt
    instruction_files = {
        "claude": "CLAUDE.md",
        "codex": ".codex/instructions.md",
        "kimi": "AGENTS.md",
        "minimax": "AGENT_INSTRUCTIONS.md",
    }
    instruction_file = instruction_files.get(agent_type)
    if instruction_file:
        dest = workspace / instruction_file
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(_build_agent_instructions(agent_type, platform))


# ── Runtime detection ──────────────────────────────────────────────────


def _container_runtime() -> str:
    """Return the path to the container runtime ('docker' or 'podman')."""
    docker_path = shutil.which("docker")
    if docker_path:
        return docker_path
    podman_path = shutil.which("podman")
    if podman_path:
        return podman_path
    return "docker"


def _is_podman() -> bool:
    """Detect if the container runtime is podman."""
    rt = _container_runtime()
    if "podman" in str(Path(rt).resolve()):
        return True
    if "podman" in Path(rt).name:
        return True
    return False


# ── Container management ────────────────────────────────────────────────


def _kill_container(workspace: Path, role: str = "researcher"):
    """Find and kill running containers for this workspace and role."""
    try:
        result = subprocess.run(
            [_container_runtime(), "ps", "-q", "--filter", f"name=researcharena-{role}-{workspace.name}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for container_id in result.stdout.strip().split("\n"):
            if container_id:
                subprocess.run(
                    [_container_runtime(), "kill", container_id],
                    capture_output=True,
                    timeout=10,
                )
                console.print(f"  Killed container {container_id[:12]}")
    except Exception as e:
        console.print(f"  [yellow]Failed to kill container: {e}[/]")
