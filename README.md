# AutoResearch

Benchmark harness for testing whether CLI agents (Claude Code, Codex, Aider) can autonomously conduct end-to-end ML research — from idea to accepted paper.

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
| Claude Code | Codex, Aider |
| Codex | Claude Code, Aider |
| Aider | Claude Code, Codex |

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
  type: "claude"              # claude, codex, aider, custom
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
outputs/runs/idea_01/
├── CLAUDE.md                    # agent permissions
├── research_guidelines.md       # research best practices
├── reviewer_guidelines.md       # reviewer instructions
├── idea.json                    # research idea
├── experiment.py                # experiment code (agent-written)
├── results.json                 # experiment results
├── paper.tex                    # LaTeX paper
├── paper.pdf                    # compiled PDF
├── figures/                     # generated figures
├── logs/
│   ├── agent_stdout.txt         # researcher agent output
│   └── agent_stderr.txt
├── review_logs/                 # reviewer agent outputs
├── reviews.json                 # aggregated reviews
└── reference_check.json         # citation verification results
```

Final summary is saved to `outputs/runs/summary.json`:

```json
{
  "agent": "claude",
  "status": "accepted",
  "ideas_tried": 2,
  "best_paper": {
    "title": "...",
    "score": 6.8,
    "workspace": "outputs/runs/idea_02"
  },
  "idea_history": [
    {"title": "...", "failure_stage": "review", "best_score": 4.2},
    {"title": "...", "failure_stage": null, "best_score": 6.8}
  ]
}
```

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
│   ├── agent_runner.py          # Docker container management
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
