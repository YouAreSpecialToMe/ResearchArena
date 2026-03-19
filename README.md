# ResearchArena

Benchmark harness for testing whether CLI agents (Claude Code, Codex, Kimi Code, Mini-Agent) can autonomously conduct end-to-end research — from idea to accepted paper.

## What this does

Given a seed field (e.g., "computer vision", "query optimization", "program analysis"), the pipeline:

1. **Launches a CLI agent** with appropriate compute access (GPU or CPU), network, and packages
2. **The agent produces a structured research proposal** — `proposal.md`, `plan.json`, and `idea.json`
3. **Self-review gates** check quality at each stage before proceeding
4. **The agent follows its experiment plan** step-by-step, producing `results.json`
5. **The agent writes a LaTeX paper** from the proposal, plan, and results
6. **Other CLI agents review it** — with read-only workspace access, searching online to verify novelty
7. If rejected, the pipeline iterates — refine the idea, retry experiments, or try a new idea entirely

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
                    ┌──────────────────────────────────────────────────────────────────┐
                    │                  researcharena (harness)                          │
                    │                                                                    │
Seed field ───────→ │  IDEATION ──→ SELF-REVIEW ──→ EXPERIMENTS ──→ SELF-REVIEW        │
  + platform        │  (proposal.md    (idea)        (follow           (experiment)      │
  (gpu/cpu)         │   plan.json                     plan.json)                         │
                    │   idea.json)                        │                              │
                    │       ↑                             ↓                              │
                    │       │ score<6              PAPER ──→ SELF-REVIEW ──→ PEER REVIEW │
                    │       │ (revise)             (paper.tex)  (paper)        │         │
                    │       │                                      ↑           │         │
                    │       │                         score<6 ─────┘           │         │
                    │       │                                                  │         │
                    │       │                                        ├─ score ≥ 8        │
                    │       │                                        │   → ACCEPTED      │
                    │       │                                        │                   │
                    │       │                                        ├─ score 5-7        │
                    │       │                                        │   → REFINE IDEA   │
                    │       │                                        │   → re-experiment  │
                    │       │                                        │   → rewrite paper  │
                    │       │                                        │                   │
                    │       │                                        └─ score < 5        │
                    │       │                                            → reject         │
                    │       └────────────────────────── try new idea ────────────────────│
                    │                                                                    │
                    │  Self-review: score >= 6 pass, < 6 revise (up to 2 retries)       │
                    │  Peer review: 3 independent reviewers in parallel                  │
                    │  Each stage = one CLI agent invocation                              │
                    └──────────────────────────────────────────────────────────────────┘
```

### Stages & guidelines

| Stage | Agent reads | Agent produces | Guideline |
|---|---|---|---|
| 1. IDEATION | seed field + resources + retry budget | `proposal.md` + `plan.json` + `idea.json` | `idea_guidelines.md` |
| 1b. SELF-REVIEW (idea) | proposal + plan + idea | score + feedback | `self_review_idea.md` |
| 2. EXPERIMENTS | `plan.json` + `idea.json` + resources | `results.json` + `figures/` (or `abandon.json` / `refine_idea.json`) | `experiment_guidelines.md` |
| 2b. SELF-REVIEW (experiment) | results + code + logs vs plan | score + feedback | `self_review_experiment.md` |
| 3. PAPER | proposal + plan + results + revision budget | `paper.tex` | `paper_writing_guidelines.md` |
| 3b. SELF-REVIEW (paper) | paper + results + code | score + feedback | `self_review_paper.md` |
| 4. PEER REVIEW | paper + workspace (read-only) | review scores | `reviewer_guidelines.md` |
| 4b. REFINE IDEA | original idea + reviewer feedback + results | updated `proposal.md` + `plan.json` + `idea.json` | `idea_guidelines.md` |

### Self-review gates

Three quality checkpoints between stages, using the researcher agent as its own reviewer:

| Gate | Checks | Pass threshold |
|---|---|---|
| After ideation | Novelty, soundness, significance, plan quality, references | score >= 6 |
| After experiments | Plan compliance, rigor, integrity, reproducibility | score >= 6 |
| After paper | All 9 dimensions (pre-submission check) | score >= 6 |

Each gate retries up to 2 times before proceeding anyway. Self-review can be disabled per-gate or entirely via config.

### Mid-experiment options

During experiments, the agent can:

| Signal | File | What happens |
|---|---|---|
| **Continue** | `results.json` | Normal flow → self-review → paper |
| **Refine** | `refine_idea.json` | Goes to REFINE_IDEA stage for proper re-ideation (up to 3x per idea) |
| **Abandon** | `abandon.json` | Drops idea entirely, new ideation |

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
├── proposal.md               # research proposal
├── plan.json                 # experiment plan
├── idea.json                 # idea summary
├── results.json              # experiment results
├── paper.tex / paper.pdf     # the paper
├── figures/                  # generated figures
├── references/               # parsed reference papers
├── logs/                     # stdout, stderr, events.jsonl
└── CLAUDE.md                 # agent context file
```

### Review sources

Each paper is evaluated by independent peer reviewers running in parallel:

| Reviewer | Model |
|---|---|
| Claude Code | claude-opus-4-6 |
| Codex | gpt-5.4 |
| Kimi Code | kimi-k2.5 |

Reviewers get read-only workspace access, check code/logs, and search online to verify novelty claims.

### Auto-reviewer selection

The agent under test is excluded from the reviewer pool (unless `allow_self_review: true`):

| Researcher | Reviewers |
|---|---|
| Claude Code | Codex, Kimi Code |
| Codex | Claude Code, Kimi Code |
| Kimi Code | Claude Code, Codex |

For smoke tests with only one agent, set `allow_self_review: true` to let the agent review its own work.

## Setup

### 1. Install the harness

```bash
pip install -e .
```

### 2. (Docker mode) Build or pull images

```bash
# GPU image
docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
docker tag pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel researcharena/agent:latest

# CPU image
docker build -f Dockerfile.cpu -t researcharena/agent-cpu:latest .
```

CLI agent binaries (claude, codex, etc.) are automatically mounted from the host into the container at runtime.

### 3. (Local mode) Just install agent CLIs

```bash
claude login        # Claude Code
codex login         # Codex (optional)
pip install kimi-cli  # Kimi Code (optional)
```

No container needed. Set `runtime: "local"` in your config.

### 4. Set API keys

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export HF_TOKEN=hf_...
export MOONSHOT_API_KEY=sk-...      # for Kimi Code
```

For subscription agents (Claude Code, Codex), just run `claude login` / `codex login`.

## Usage

### Run a single seed

```bash
# GPU seed (default)
researcharena run --seed "computer vision" --agent claude

# CPU seed
researcharena run --seed "query optimization" --agent claude --platform cpu

# Custom workspace
researcharena run --seed "computer vision" --workspace outputs/claude/cv_run1
```

### Run 8 parallel experiments on SLURM

```bash
# Submit 8 Claude researchers, one per A6000 GPU
sbatch scripts/slurm_claude_gpu.sh

# Submit 8 Codex researchers
sbatch scripts/slurm_codex_gpu.sh
```

### Run 8 parallel experiments locally

```bash
# Claude researchers
bash scripts/launch_parallel.sh

# Codex researchers
bash scripts/launch_parallel_codex.sh
```

### Run the full benchmark

```bash
researcharena bench --agent claude                      # all seeds
researcharena bench --agent claude --platform gpu       # GPU seeds only
researcharena bench --agent claude --conference sigmod  # SIGMOD seeds
researcharena bench --agent claude --field "query optimization"
```

### Other commands

```bash
researcharena list-seeds                          # show all seeds
researcharena list-seeds --platform gpu           # GPU seeds only
researcharena review-only outputs/runs/idea_01/   # review existing paper
```

## Configuration

See [`configs/8xa6000.yaml`](configs/8xa6000.yaml) for a full example:

```yaml
seed_topic: "computer vision"
seed_platform: "gpu"
seed_domain: "ml"

platforms:
  gpu:
    resources:
      total_gpus: 1
      gpu_type: "NVIDIA RTX A6000"
      gpu_memory_gb: 48
      total_cpus: 4
      total_memory_gb: 60

agent:
  type: "claude"
  model: "claude-opus-4-6"
  runtime: "local"
  max_turns: 200
  ideation_timeout: 3600      # 1 hour
  paper_timeout: 3600         # 1 hour

experiment:
  max_gpu_hours: 8
  max_experiment_retries_per_idea: 3
  max_refine_per_idea: 3      # mid-experiment idea refinements

paper:
  template: "neurips"
  max_revisions: 2

self_review:
  enabled: true
  threshold: 6                # score >= 6 to pass
  max_retries_per_gate: 2
  timeout: 900                # 15 min per self-review
  gates:
    idea: true
    experiment: true
    paper: true

review:
  accept_threshold: 8
  agents:
    - type: "claude"
      name: "Claude Code"
      model: "claude-opus-4-6"
      review_timeout: 3600
    - type: "codex"
      name: "Codex"
      model: "gpt-5.4"
      review_timeout: 3600
    - type: "kimi"
      name: "Kimi Code"
      review_timeout: 3600

pipeline:
  max_ideas_per_seed: 3
  max_global_steps: 50
```

## Scoring

Reviews use a 0-10 scale (even numbers only):

| Score | Meaning | Decision |
|---|---|---|
| 10 | Seminal paper, top 5% | accept |
| 8 | Clear accept, strong contribution | accept |
| 6 | Marginal | revision |
| 4 | Below threshold | reject |
| 2 | Strong rejection | reject |
| 0 | Fabricated or trivial | reject |

Acceptance threshold: **8**. Score 5-7 triggers a revision loop. Score < 5 is rejected.

### Scoring dimensions (per-dimension 1-10)

1. **Novelty** — reviewer searches online (arXiv, Semantic Scholar) to verify
2. **Soundness** — methodology and evidence
3. **Significance** — does it matter to the community
4. **Clarity** — writing quality and organization
5. **Reproducibility** — enough detail to reimplement
6. **Experimental rigor** — baselines, ablations, error bars
7. **References** — key related works cited and discussed
8. **Reference integrity** — reviewer verifies citations exist online
9. **Results integrity** — sanity check: code/logs match reported numbers

## Iteration & backtracking

| Outcome | Action |
|---|---|
| Self-review fails (score < 6) | Revise and retry (up to 2 retries per gate) |
| Self-review budget exhausted | Proceed to next stage anyway |
| Experiments fail (code crashes) | Retry with error context (up to 3 attempts) |
| Agent writes `refine_idea.json` | Full re-ideation with experiment context (up to 3x) |
| Agent writes `abandon.json` | Abandon idea, try new one |
| Paper score 5-7 (marginal) | Refine idea → re-experiment → rewrite → re-review (up to 2 revisions) |
| Paper score < 5 (rejected) | Abandon idea, try new one |
| Revisions exhausted | Abandon idea, try new one |

The agent sees its retry budget at every stage. Best paper across all attempts is tracked.

## Output

```
outputs/claude/computer_vision_20260319_143052/
├── summary.json                         # final result
├── tracker.json                         # time, tokens, cost per action
│
├── idea_01/
│   ├── .venv/                           # per-workspace virtualenv (local mode)
│   ├── CLAUDE.md                        # agent context
│   ├── idea_guidelines.md               # stage guidelines
│   ├── experiment_guidelines.md
│   ├── paper_writing_guidelines.md
│   ├── proposal.md                      # research proposal
│   ├── plan.json                        # experiment plan
│   ├── idea.json                        # idea summary
│   ├── results.json                     # experiment results
│   ├── paper.tex / paper.pdf            # the paper
│   ├── figures/                         # generated figures
│   ├── references/                      # parsed reference papers
│   ├── logs/                            # agent stdout/stderr/events
│   └── reviews.json                     # aggregated reviews
│
└── idea_02/                             # if idea_01 was rejected
```

## Project structure

```
researcharena/
├── cli.py                       # CLI entry point (run, bench, review-only, list-seeds)
├── pipeline.py                  # State machine orchestrator
├── stages/
│   ├── ideation.py              # Stage 1: structured proposal + plan + idea
│   ├── self_review.py           # Self-review gates (idea, experiment, paper)
│   ├── experiment_design.py     # Stage 2: plan-driven experiments
│   ├── paper_writing.py         # Stage 3: LaTeX paper writing
│   └── review.py                # Stage 4: parallel CLI agent peer reviewers
├── utils/
│   ├── agent_runner.py          # Local + Docker/Podman execution
│   ├── tracker.py               # Time, tokens, cost tracking
│   ├── config.py                # YAML config loading
│   ├── paperreview.py           # paperreview.ai automation
│   └── reference_checker.py     # Citation verification (disabled, high false-positive rate)
├── templates/                   # Domain-specific guideline templates
│   ├── ml/                      # ML/AI (ICLR, NeurIPS, ICML, CVPR, ACL)
│   │   ├── idea_guidelines.md
│   │   ├── experiment_guidelines.md
│   │   ├── paper_writing_guidelines.md
│   │   ├── reviewer_guidelines.md
│   │   ├── self_review_idea.md
│   │   ├── self_review_experiment.md
│   │   └── self_review_paper.md
│   ├── systems/                 # Systems (OSDI, SOSP, EuroSys)
│   ├── databases/               # Databases (SIGMOD, VLDB, ICDE)
│   ├── pl/                      # Programming Languages (PLDI, POPL, OOPSLA)
│   ├── theory/                  # Theory (STOC, FOCS, SODA)
│   └── security/                # Security (CCS, S&P, USENIX Security)
├── configs/
│   ├── 8xa6000.yaml             # 8x A6000 GPU parallel config (Claude)
│   ├── 8xa6000_codex.yaml       # 8x A6000 GPU parallel config (Codex)
│   ├── smoke_test.yaml          # Quick single-idea smoke test
│   ├── seed_gpu_exp.yaml        # GPU experiment seeds (8 topics)
│   └── seeds.yaml               # Full seed list
├── scripts/
│   ├── slurm_claude_gpu.sh      # SLURM array job: 8 Claude researchers
│   ├── slurm_codex_gpu.sh       # SLURM array job: 8 Codex researchers
│   ├── launch_parallel.sh       # Local parallel launch (Claude)
│   └── launch_parallel_codex.sh # Local parallel launch (Codex)
├── Dockerfile                   # GPU image
├── Dockerfile.cpu               # CPU image
└── pyproject.toml
```
