# AutoResearch

Benchmark harness for testing whether CLI agents (Claude Code, Codex, Aider, Kimi, MiniMax) can autonomously conduct end-to-end ML research — from idea to accepted paper.

## What this does

Given a seed topic, the pipeline:

1. **Launches a CLI agent in Docker** with GPU access, network, and ML packages
2. **The agent does everything** — comes up with an idea, writes experiment code, runs it, analyzes results, writes a LaTeX paper
3. **Other CLI agents review it** — running in the same Docker image with read-only access to the full workspace (code, logs, results)
4. **paperreview.ai** provides an additional external review
5. **References are verified** against Semantic Scholar and CrossRef (any fake citation = auto reject)
6. If rejected, the pipeline iterates — revise the paper, retry experiments, or try a new idea entirely

The pipeline is a state machine with backtracking. It tracks the best paper across all attempts.

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │           autoresearch (harness)                    │
                    │                                                     │
Seed topic ───────→ │  IDEATION ──→ EXPERIMENTS ──→ PAPER ──→ REVIEW     │
                    │     ↑              ↑                      │        │
                    │     │              │         reject ───────┘        │
                    │     │              └── retry experiments            │
                    │     └── try new idea                               │
                    │                                                     │
                    │  Each stage = one CLI agent invocation in Docker    │
                    └─────────────────────────────────────────────────────┘
```

### Docker sharing

All agents (researcher + reviewers) run in the **same Docker image**:

```
autoresearch/agent:latest
├── Python, CUDA, PyTorch, Transformers, etc.
├── Claude Code, Codex, Aider CLIs
│
├── Researcher (e.g., Claude Code)
│   └── docker run -v workspace:/workspace        (read-write)
│
├── Reviewer 1 (e.g., Codex)
│   └── docker run -v workspace:/workspace:ro     (read-only)
│
└── Reviewer 2 (e.g., Aider)
    └── docker run -v workspace:/workspace:ro     (read-only)
```

Reviewers get full access to code, logs, and results — they can even re-run experiments to verify. They just can't modify anything.

### Auto-reviewer selection

The agent under test is excluded from the reviewer pool:

| Researcher | Reviewers |
|---|---|
| Claude Code | Codex, Aider, Kimi, MiniMax |
| Codex | Claude Code, Aider, Kimi, MiniMax |
| Aider | Claude Code, Codex, Kimi, MiniMax |
| Kimi | Claude Code, Codex, Aider, MiniMax |
| MiniMax | Claude Code, Codex, Aider, Kimi |

## Setup

### 1. Build the Docker image

```bash
docker build -t autoresearch/agent:latest .
```

### 2. Install the harness

```bash
pip install -e .
```

### 3. Set API keys

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export HF_TOKEN=hf_...
export MOONSHOT_API_KEY=sk-...      # for Kimi
export MINIMAX_API_KEY=...          # for MiniMax
export MINIMAX_GROUP_ID=...         # for MiniMax
```

### 4. (Optional) Configure paperreview.ai

Edit `configs/default.yaml`:

```yaml
review:
  paperreview:
    email: "you@gmail.com"
    email_password: "app-specific-password"
```

## Usage

### Run the full pipeline

```bash
# Test Claude Code
autoresearch run --seed "efficient fine-tuning for LLMs" --agent claude

# Test Codex
autoresearch run --seed "efficient fine-tuning for LLMs" --agent codex

# Test Kimi
autoresearch run --seed "efficient fine-tuning for LLMs" --agent kimi

# Test MiniMax
autoresearch run --seed "efficient fine-tuning for LLMs" --agent minimax

# Override model
autoresearch run --agent claude --model claude-opus-4-6

# Limit ideas
autoresearch run --seed "your topic" --max-ideas 3
```

### Review an existing workspace

```bash
autoresearch review-only outputs/runs/idea_01/
```

### Custom config

```bash
autoresearch run -c configs/my_config.yaml
```

## Configuration

See [`configs/default.yaml`](configs/default.yaml) for all options:

```yaml
seed_topic: "your research topic"

agent:
  type: "claude"              # claude, codex, aider, kimi, minimax, custom
  model: "claude-sonnet-4-6"
  docker_image: "autoresearch/agent:latest"
  gpus: 1
  memory_limit: "32g"

experiment:
  max_gpu_hours: 4
  max_experiment_retries_per_idea: 3

paper:
  template: "neurips"
  max_revisions: 2

review:
  accept_threshold: 6         # out of 10
  agents:                     # all agents; researcher auto-excluded
    - type: "claude"
      name: "Claude Code"
    - type: "codex"
      name: "Codex"
    - type: "aider"
      name: "Aider"
    - type: "kimi"
      name: "Kimi"
    - type: "minimax"
      name: "MiniMax"

pipeline:
  max_ideas_per_seed: 5
  max_global_steps: 30
```

## Iteration & backtracking

The pipeline doesn't just retry linearly. It backtracks to the right stage based on what failed:

| Failure | Action |
|---|---|
| Experiments fail (code crashes) | Retry experiments with error context |
| Experiment budget exhausted | Abandon idea, try a new one |
| Paper rejected (score far below threshold) | Abandon idea, try new one |
| Paper rejected (close to threshold) | Revise paper with reviewer feedback |
| Paper revisions exhausted | Abandon idea, try new one |
| Fake references detected | Auto reject, revise with specific feedback |

When all ideas are exhausted, the pipeline returns the **best-scoring paper** across all attempts (or `None` if no paper was ever produced).

## Review process

Each paper goes through three review gates:

### 1. Reference verification

Every citation is checked against Semantic Scholar and CrossRef APIs. **Any fake reference = automatic rejection (score 0).** The feedback lists exactly which references need replacing.

### 2. paperreview.ai

The compiled PDF is submitted to [paperreview.ai](https://paperreview.ai/) (Stanford Agentic Reviewer) for an external review.

### 3. CLI agent reviewers

Other CLI agents run in the **same Docker image** with the workspace mounted read-only. They verify the full chain:

- **Code → Logs**: Does the code run? Do logs show real training?
- **Logs → Results**: Do final metrics in logs match results.json?
- **Results → Paper**: Do numbers in the paper match results.json?
- **Code → Paper**: Does the code implement what the paper describes?

If any link in this chain is broken, the reviewer scores `results_integrity: 0` and the paper is rejected.

## Output

Each run produces a workspace directory:

```
outputs/runs/
├── summary.json                         # final run summary
├── tracker.json                         # full tracking data (see below)
│
├── idea_01/
│   ├── CLAUDE.md                        # agent permissions
│   ├── research_guidelines.md           # research best practices
│   ├── reviewer_guidelines.md           # reviewer instructions
│   ├── idea.json                        # research idea
│   ├── experiment.py                    # experiment code (agent-written)
│   ├── results.json                     # experiment results
│   ├── paper.tex                        # LaTeX paper
│   ├── paper.pdf                        # compiled PDF
│   ├── figures/                         # generated figures
│   ├── logs/
│   │   ├── claude_1710523200_stdout.txt # researcher agent output
│   │   ├── claude_1710523200_stderr.txt
│   │   ├── claude_1710523200_command.txt
│   │   └── claude_1710523200_events.jsonl  # timestamped stream events (Claude)
│   ├── review_logs/                     # reviewer agent outputs (same format)
│   ├── reviews.json                     # aggregated reviews
│   └── reference_check.json            # citation verification results
│
└── idea_02/
    └── ...
```

## Tracking

Every run produces a `tracker.json` with structured data at three levels of granularity:

### Run level

Total time, tokens, cost, and per-stage aggregation:

```json
{
  "total_elapsed_seconds": 18247.3,
  "total_tokens": {"input_tokens": 1842000, "output_tokens": 563000, "total_tokens": 2405000},
  "total_cost_usd": 13.97,
  "stages": {
    "ideation":    {"elapsed_seconds": 2105, "tokens": {...}, "cost_usd": 1.21, "actions": 2, "successes": 2, "failures": 0},
    "experiments": {"elapsed_seconds": 9842, "tokens": {...}, "cost_usd": 4.78, "actions": 3, "successes": 2, "failures": 1,
                    "failure_categories": {"oom": 1}},
    "paper":       {"elapsed_seconds": 3210, "tokens": {...}, "cost_usd": 3.96, "actions": 3, "successes": 3, "failures": 0},
    "review":      {"elapsed_seconds": 3089, "tokens": {...}, "cost_usd": 4.01, "actions": 8, "successes": 7, "failures": 1}
  }
}
```

### Action level

Each pipeline stage invocation (ideation attempt, experiment run, paper draft, review round) is a separate action with its own timing, tokens, cost, failure classification, workspace diff, and log file links:

```json
{
  "stage": "experiments",
  "action": "run_experiments",
  "agent_type": "claude",
  "model": "claude-sonnet-4-6",
  "attempt": 1,
  "elapsed_seconds": 5124.7,
  "tokens": {"input_tokens": 320000, "output_tokens": 98000, "total_tokens": 418000},
  "outcome": "failure",
  "failure_category": "oom",
  "details": "No results.json produced",
  "cost_usd": 2.43,
  "log_files": {
    "stdout": "outputs/runs/idea_01/logs/claude_1710523200_stdout.txt",
    "stderr": "outputs/runs/idea_01/logs/claude_1710523200_stderr.txt",
    "command": "outputs/runs/idea_01/logs/claude_1710523200_command.txt",
    "events": "outputs/runs/idea_01/logs/claude_1710523200_events.jsonl"
  },
  "workspace_diff": {
    "created": [{"path": "experiment.py", "size": 4821}, {"path": "train.py", "size": 12043}],
    "modified": [],
    "deleted": []
  },
  "sub_actions": [...]
}
```

Failure categories: `oom`, `timeout`, `rate_limit`, `auth_error`, `gpu_error`, `import_error`, `crash`, `syntax_error`, `runtime_error`, `docker_error`, `network_error`, `unknown`

### Sub-action level

Each tool call the agent made within a stage. For Claude Code (via `--output-format stream-json`), this includes the LLM's reasoning, per-turn token usage, wall time per tool call, tool output, and which files each tool call actually changed on disk:

```json
{
  "tool": "Bash",
  "input": "python experiment.py --epochs 50 --seed 42",
  "output": "Epoch 50/50 - loss: 0.234 - acc: 0.891",
  "reasoning": "Let me run the training with the first seed.",
  "tokens": {"input": 9400, "output": 450},
  "duration_seconds": 342.8,
  "files_affected": ["results.json", "figures/accuracy_curve.png", "checkpoints/best_model.pt"]
}
```

### Tracking coverage by agent

All agents get full tracking at every level. The data source differs but the output is the same:

| Capability | Claude Code | Codex | Aider / Kimi / MiniMax |
|---|---|---|---|
| **Output format** | `--output-format stream-json` | `--json` (JSONL) | `--verbose` (plaintext) |
| Tool calls | structured events | structured events | regex parsing |
| Tool input | rich per-tool summaries | from event payload | from verbose output |
| Tool output | tool result events | command output | lines after tool call |
| LLM reasoning | text_delta blocks | agent_message events | accumulated text between actions |
| Per-turn tokens | message_delta usage | turn.completed usage | `Tokens: N sent, N received` |
| Per-tool duration | timestamped events.jsonl | timestamped events.jsonl | timestamped events.jsonl |
| Error details | is_error in result | exit code | error pattern detection |
| File changes | workspace diff | workspace diff | workspace diff |
| Failure category | stderr patterns | stderr patterns | stderr patterns |

All agents stream through `Popen` with line-by-line timestamping, so per-tool-call duration is available for every agent. Kimi and MiniMax run through Aider with their OpenAI-compatible APIs.

Sub-action fields:

| Field | Description | Source |
|---|---|---|
| `tool` | Tool name (Read, Write, Edit, Bash, WebSearch, etc.) | Parsed from agent stdout |
| `input` | What was passed to the tool | Summarized from tool input |
| `output` | What the tool returned (truncated) | Tool result event |
| `reasoning` | LLM's thinking text before this tool call | Text blocks between tool calls |
| `tokens` | `{input, output}` token counts for this turn | `message_delta` usage event |
| `duration_seconds` | Wall time for the tool execution | Timestamped events file |
| `error` | Error message if the tool call failed | Tool result with `is_error` |
| `files_affected` | Files this tool call created/modified | Cross-referenced with workspace diff |

Files changed on disk but not claimed by any tool call appear as `(side_effect)` entries — these are typically artifacts produced by running scripts (checkpoints, figures, caches).

## Project structure

```
autoresearch/
├── cli.py                       # CLI entry point
├── pipeline.py                  # State machine orchestrator
├── stages/
│   ├── ideation.py              # Agent generates research idea
│   ├── experiment_design.py     # Agent implements & runs experiments
│   ├── paper_writing.py         # Agent writes LaTeX paper
│   └── review.py                # Multi-source review (refs + paperreview.ai + CLI agents)
├── utils/
│   ├── agent_runner.py          # Docker container management + streaming + failure classification
│   ├── tracker.py               # Run/action/sub-action tracking with time, tokens, cost
│   ├── action_parser.py         # Parse agent stdout into structured sub-actions
│   ├── workspace_diff.py        # Before/after filesystem snapshots
│   ├── config.py                # YAML config loading
│   ├── llm.py                   # LLM client (Anthropic/OpenAI)
│   ├── paperreview.py           # paperreview.ai automation
│   └── reference_checker.py     # Citation verification (Semantic Scholar + CrossRef)
├── templates/
│   ├── research_guidelines.md   # Guidelines for the researcher agent
│   └── reviewer_guidelines.md   # Guidelines for reviewer agents
├── configs/
│   └── default.yaml             # Default configuration
└── Dockerfile                   # GPU-enabled container with all agent CLIs
```
