# ResearchArena

Benchmark harness for testing whether CLI agents (Claude Code, Codex, Kimi Code, Mini-Agent) can autonomously conduct end-to-end research — from idea to accepted paper.

## What this does

Given a seed field (e.g., "computer vision", "query optimization", "program analysis"), the pipeline:

1. **Launches a CLI agent** with appropriate compute access (GPU or CPU), network, and packages
2. **The agent does everything** — explores the field, comes up with an idea, designs and runs experiments, writes a LaTeX paper
3. **Other CLI agents review it** — with read-only workspace access, searching online to verify novelty
4. **paperreview.ai** provides an additional external review
5. **References are verified** against Semantic Scholar and CrossRef
6. If rejected, the pipeline iterates — revise the paper, retry experiments, or try a new idea entirely

Each stage has a dedicated guideline file that the agent reads before starting. The agent receives computational resource constraints upfront so it scopes ideas to fit.

## Conferences & Areas

Seeds span multiple CS conferences and two platforms:

| Platform | Areas | Conferences |
|---|---|---|
| **GPU** | NLP, CV, graphics, generative models, RL, robotics, AI4Science | ICLR, NeurIPS, ICML, CVPR, ACL, EMNLP, SIGGRAPH, CoRL |
| **CPU** | Systems, databases, PL, theory, architecture, security | OSDI, SOSP, SIGMOD, VLDB, PLDI, POPL, STOC, FOCS, ISCA, CCS |

GPU seeds get CUDA-enabled containers with GPU access. CPU seeds get lightweight containers optimized for analytical, algorithmic, and systems-level experiments.

## Architecture

```
                    ┌──────────────────────────────────────────────────────────────┐
                    │               researcharena (harness)                        │
                    │                                                              │
Seed field ───────→ │  IDEATION ──→ EXPERIMENTS ──→ PAPER ──→ REVIEW              │
  + platform        │     ↑              ↑                       │                │
  (gpu/cpu)         │     │              │                       ├─ score ≥ 8     │
                    │     │              │                       │   → ACCEPTED   │
                    │     │              │                       │                │
                    │     │              │                       ├─ score = 6     │
                    │     │              │                       │   ↓            │
                    │     │              │                   REFINE IDEA          │
                    │     │              │                   (read feedback,      │
                    │     │              │                    update idea.json)   │
                    │     │              │                       │                │
                    │     │              └───────────────────────┘                │
                    │     │                  re-run experiments                    │
                    │     │                  + rewrite paper                       │
                    │     │                  + re-review                           │
                    │     │                                                        │
                    │     │              ┌─ score ≤ 4 (rejected)                   │
                    │     │              ├─ experiment retries exhausted           │
                    │     │              └─ agent writes abandon.json             │
                    │     └──────────────── try new idea ─────────────────────────│
                    │                                                              │
                    │     score 6 + revisions exhausted → reject, try new idea      │
                    │                                                              │
                    │  Each stage = one CLI agent invocation                       │
                    │  Agent sees resource constraints + retry budget at each stage│
                    └──────────────────────────────────────────────────────────────┘
```

### Stages & guidelines

| Stage | Agent reads | Agent produces | Guideline |
|---|---|---|---|
| 1. IDEATION | seed field + resource constraints + retry budget | `idea.json` | `idea_guidelines.md` |
| 2. EXPERIMENTS | `idea.json` + resource constraints + retry budget | `results.json` + `figures/` (or `abandon.json`) | `experiment_guidelines.md` |
| 3. PAPER | `idea.json` + `results.json` + revision budget | `paper.tex` | `paper_writing_guidelines.md` |
| 4. REVIEW | paper + workspace (read-only) | review scores | `reviewer_guidelines.md` |
| 4b. REFINE IDEA | original idea + reviewer feedback + previous results | updated `idea.json` | `idea_guidelines.md` |

### Runtime modes

The harness supports two runtime modes:

**Docker/Podman mode** (`runtime: "docker"`, default) — each agent runs in an isolated container:

```
GPU platform: researcharena/agent:latest (pytorch/pytorch base, CUDA)
CPU platform: researcharena/agent-cpu:latest (python:3.11-slim base)

├── CLI agent binaries mounted from host
│
├── Researcher (e.g., Claude Code)
│   └── docker run -v workspace:/workspace        (read-write)
│   └── ~/.claude/ mounted rw (persistent memory across stages)
│
├── Reviewer 1 (e.g., Codex)
│   └── docker run -v workspace:/workspace:ro     (read-only)
│
└── Reviewer 2 (e.g., Kimi Code)
    └── docker run -v workspace:/workspace:ro     (read-only)
```

**Local mode** (`runtime: "local"`) — agents run directly on the host:

```
workspace/idea_01/
├── .venv/                    # per-workspace virtualenv (system-site-packages)
├── idea.json, results.json   # agent artifacts
├── logs/                     # stdout, stderr, events.jsonl
└── CLAUDE.md                 # agent context file
```

Each workspace gets its own virtualenv so agents can `pip install` packages without conflicting with each other or the host. Local mode is useful when Docker/Podman is unavailable or for development.

### Review sources

Each paper is evaluated by three independent sources:

1. **Reference checker** — verifies citations against Semantic Scholar + CrossRef APIs. Fake references trigger automatic score 0.
2. **paperreview.ai** — external Stanford agentic reviewer. Submits PDF, polls email for review token, fetches structured review. Requires Gmail with app password.
3. **CLI agent reviewers** — other agents review with read-only workspace access, checking code, logs, and searching online to verify novelty claims.

### Auto-reviewer selection

The agent under test is excluded from the reviewer pool (unless `allow_self_review: true`):

| Researcher | Reviewers |
|---|---|
| Claude Code | Codex, Kimi Code, Mini-Agent |
| Codex | Claude Code, Kimi Code, Mini-Agent |
| Kimi Code | Claude Code, Codex, Mini-Agent |
| Mini-Agent | Claude Code, Codex, Kimi Code |

For smoke tests with only one agent, set `allow_self_review: true` to let the agent review its own work.

## Setup

### 1. Install the harness

```bash
pip install -e .
```

### 2. (Docker mode) Build or pull images

**GPU image** (for NLP, CV, graphics, generative models):

```bash
# Option A: Pull pre-built PyTorch image and tag it
docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
docker tag pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel researcharena/agent:latest

# Option B: Build with extra ML packages
docker build -t researcharena/agent:latest .

# Podman (rootless, no subuid):
podman pull docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
podman tag docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel localhost/researcharena/agent:latest
```

**CPU image** (for systems, databases, PL, theory):

```bash
docker build -f Dockerfile.cpu -t researcharena/agent-cpu:latest .

# Podman:
podman build --userns=host -f Dockerfile.cpu -t localhost/researcharena/agent-cpu:latest .
```

CLI agent binaries (claude, codex, etc.) are automatically mounted from the host into the container at runtime — they don't need to be installed in the image.

### 3. (Local mode) Just install agent CLIs

```bash
# Claude Code
claude login

# Codex (optional)
npm install -g @openai/codex && codex login

# Kimi Code (optional)
pip install kimi-cli

# Mini-Agent (optional)
pip install mini-agent
```

No container needed. Set `runtime: "local"` in your config.

### 4. Set API keys

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export HF_TOKEN=hf_...
export MOONSHOT_API_KEY=sk-...      # for Kimi Code
export MINIMAX_API_KEY=...          # for Mini-Agent
```

For subscription agents (Claude Code, Codex), just run `claude login` / `codex login` — credentials are auto-mounted into containers or used directly in local mode.

### 5. (Optional) Configure paperreview.ai

Requires Gmail with an [app password](https://myaccount.google.com/apppasswords) (Outlook/Microsoft accounts are not supported — they require OAuth2).

```bash
export PAPERREVIEW_EMAIL="you@gmail.com"
export PAPERREVIEW_PASSWORD="your-app-password"
```

Then in your config:

```yaml
review:
  paperreview:
    email: "${PAPERREVIEW_EMAIL}"
    email_password: "${PAPERREVIEW_PASSWORD}"
    imap_server: "imap.gmail.com"       # default
```

Config files support `${ENV_VAR}` substitution so credentials stay out of version control.

## Usage

### Run a single seed

```bash
# GPU seed (default)
researcharena run --seed "computer vision" --agent claude

# CPU seed
researcharena run --seed "query optimization" --agent claude --platform cpu
```

### Run with local mode (no container)

```bash
researcharena run -c configs/smoke_test.yaml
```

### Run the full benchmark

```bash
# All seeds (GPU + CPU)
researcharena bench --agent claude

# GPU seeds only (NLP, CV, graphics, etc.)
researcharena bench --agent claude --platform gpu

# CPU seeds only (systems, DB, PL, theory)
researcharena bench --agent claude --platform cpu

# Filter by conference
researcharena bench --agent claude --conference sigmod
researcharena bench --agent claude --conference iclr

# Single field
researcharena bench --agent claude --field "query optimization"

# Compare agents
researcharena bench --agent claude
researcharena bench --agent codex
researcharena bench --agent kimi
```

### Other commands

```bash
researcharena list-seeds                          # show all seeds
researcharena list-seeds --platform gpu           # GPU seeds only
researcharena list-seeds --platform cpu           # CPU seeds only
researcharena list-seeds --conference sigmod      # SIGMOD seeds
researcharena review-only outputs/runs/idea_01/   # review existing paper
researcharena run -c configs/my_config.yaml       # custom config
```

## Configuration

See [`configs/default.yaml`](configs/default.yaml):

```yaml
seed_topic: "computer vision"
seed_platform: "gpu"              # or "cpu"

# Per-platform resource profiles
platforms:
  gpu:
    resources:
      total_gpus: 4
      gpu_ids: "0,1,2,3"
      gpu_type: "NVIDIA A100"
      gpu_memory_gb: 80
      total_cpus: 32
      total_memory_gb: 128
    docker_image: "researcharena/agent:latest"

  cpu:
    resources:
      total_gpus: 0
      total_cpus: 32
      total_memory_gb: 128
    docker_image: "researcharena/agent-cpu:latest"

agent:
  type: "claude"                # claude, codex, kimi, minimax, custom
  model: "claude-sonnet-4-6"
  runtime: "docker"             # "docker" or "local"
  docker_image: "researcharena/agent:latest"

experiment:
  max_gpu_hours: 8
  max_experiment_retries_per_idea: 3

paper:
  template: "neurips"
  max_revisions: 2

review:
  accept_threshold: 8           # ≥8 accept, 6 revision, ≤4 reject
  allow_self_review: false      # true for smoke tests with one agent

pipeline:
  max_ideas_per_seed: 5
  max_global_steps: 30
```

### Seeds format

Each seed in `configs/seeds.yaml` has:

```yaml
seeds:
  - name: "computer vision"
    conferences: [iclr, neurips, cvpr, eccv]
    platform: gpu

  - name: "query optimization"
    conferences: [sigmod, vldb, icde]
    platform: cpu
```

### Smoke test config

A minimal config for quick verification ([`configs/smoke_test.yaml`](configs/smoke_test.yaml)):

```yaml
agent:
  runtime: "local"              # no container needed
  max_turns: 50
  ideation_timeout: 1800        # 30 min

experiment:
  max_gpu_hours: 1
  max_experiment_retries_per_idea: 1

paper:
  max_revisions: 1              # allow 1 revision after review

pipeline:
  max_ideas_per_seed: 1
  max_global_steps: 10
```

## Scoring (ICLR 2026 scale)

Reviews use the ICLR 2026 scoring system, aligned with paperreview.ai:

| Score | Meaning | Decision |
|---|---|---|
| 10 | Seminal paper, top 5% | accept |
| 8 | Clear accept, strong contribution | accept |
| 6 | Marginal | revision (try to improve), reject if revisions exhausted |
| 4 | Below threshold | reject, abandon idea |
| 2 | Strong rejection | reject, abandon idea |
| 0 | Fabricated or trivial | reject, abandon idea |

Acceptance threshold: **8**. Score 6 triggers a revision loop. Scores are averaged across all review sources.

### Scoring dimensions (per-dimension 1-10)

1. **Novelty** — reviewer searches online (arXiv, Semantic Scholar) to verify
2. **Soundness** — methodology and evidence
3. **Significance** — does it matter to the community
4. **Clarity** — writing quality and organization
5. **Reproducibility** — enough detail to reimplement
6. **Experimental rigor** — baselines, ablations, error bars
7. **References** — reviewer verifies citations exist online
8. **Results integrity** — sanity check: code/logs match reported numbers

### Automatic rejection grounds

- Fake or non-existent references
- Experiment code cannot run or doesn't produce claimed results
- Logs show different numbers than the paper reports
- Numbers in paper don't match results.json

## Iteration & backtracking

| Outcome | Action |
|---|---|
| Experiments fail (code crashes) | Retry with error context (up to 3 attempts) |
| Experiment retries exhausted | Abandon idea, try new one |
| Agent writes `abandon.json` | Abandon idea early (agent decides idea isn't viable) |
| Paper writing fails (no paper.tex) | Retry paper writing |
| Paper score = 6 (marginal) | Full revision loop: refine idea → re-run experiments → rewrite paper (up to 2 revisions) |
| Paper score ≤ 4 (rejected) | Abandon idea, try new one |
| Revisions exhausted, still rejected | Abandon idea, try new one |
| Fake references detected | Score 0, abandon idea |

The agent sees its retry budget at every stage so it can adjust strategy — e.g., be more conservative on the last attempt, or abandon early if retries are available.

Best paper across all attempts is tracked and returned.

## Output

```
outputs/runs/
├── summary.json                         # final result
├── tracker.json                         # time, tokens, cost per action
│
├── idea_01/
│   ├── .venv/                           # per-workspace virtualenv (local mode)
│   ├── CLAUDE.md                        # agent context (stage overview)
│   ├── idea_guidelines.md               # how to find a novel idea
│   ├── experiment_guidelines.md         # how to design & run experiments
│   ├── paper_writing_guidelines.md      # how to write the paper
│   ├── idea.json                        # research idea
│   ├── results.json                     # experiment results
│   ├── paper.tex / paper.pdf            # the paper
│   ├── figures/                         # generated figures
│   ├── logs/                            # researcher agent output
│   ├── reviews.json                     # aggregated reviews
│   └── reference_check.json             # citation verification
│
├── idea_01_review_logs/                 # reviewer agent outputs
│
└── idea_02/                             # if idea_01 was rejected
```

## Project structure

```
researcharena/
├── cli.py                       # CLI entry point (run, bench, review-only, list-seeds)
├── pipeline.py                  # State machine orchestrator (platform-aware)
├── stages/
│   ├── ideation.py              # Stage 1: agent explores field, produces idea.json
│   ├── experiment_design.py     # Stage 2: agent implements & runs experiments
│   ├── paper_writing.py         # Stage 3: agent writes LaTeX paper
│   └── review.py                # Stage 4: reference check + paperreview.ai + CLI agent reviewers
├── utils/
│   ├── agent_runner.py          # Local + Docker/Podman execution, GPU/CPU platform support
│   ├── tracker.py               # Time, tokens, cost tracking
│   ├── config.py                # YAML config loading
│   ├── paperreview.py           # paperreview.ai automation (Playwright)
│   └── reference_checker.py     # Citation verification (Semantic Scholar + CrossRef)
├── templates/                   # Domain-specific guideline templates
│   ├── ml/                      # ML/AI (ICLR, NeurIPS, ICML, CVPR, ACL)
│   ├── systems/                 # Systems (OSDI, SOSP, EuroSys)
│   ├── databases/               # Databases (SIGMOD, VLDB, ICDE)
│   ├── pl/                      # Programming Languages (PLDI, POPL, OOPSLA)
│   ├── theory/                  # Theory (STOC, FOCS, SODA)
│   └── security/                # Security (CCS, S&P, USENIX Security)
├── configs/
│   ├── default.yaml             # Default configuration (GPU + CPU platforms)
│   ├── smoke_test.yaml          # Quick single-idea smoke test
│   └── seeds.yaml               # Seed fields across multiple conferences
├── Dockerfile                   # GPU image: pytorch base (CUDA-enabled)
├── Dockerfile.cpu               # CPU image: python:3.11-slim base
└── pyproject.toml
```
