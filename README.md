# How Far Are We From True Auto Research?

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Website](https://img.shields.io/badge/Website-ResearchArena-5eead4.svg)](https://youarespecialtome.github.io/ResearchArena/)
[![GitHub stars](https://img.shields.io/github/stars/YouAreSpecialToMe/ResearchArena?style=social)](https://github.com/YouAreSpecialToMe/ResearchArena/stargazers)

An in-depth analysis of frontier CLI agents — **Claude Code (Opus 4.6)**, **Codex (GPT-5.4)**, **Kimi Code (K2.5)** — conducting end-to-end research across diverse fields and compute resources.

- **117 agent-generated papers** — 39 per agent (Claude Code, Codex, Kimi Code), 3 trials × 13 seeds, spanning both GPU and CPU domains
- **351 code-aware peer reviews** (3 CLI-agent reviewers per paper) + **117 Stanford Agentic Reviewer scores**
- **Human analysis of every paper**, its artifacts, and agentic reviews

➡️ Read the [full write-up](https://youarespecialtome.github.io/ResearchArena/) for scores, per-domain breakdowns, case studies, and the human-inspection findings.

## What this does

Given a seed field (e.g., "computer vision", "compiler optimization"), each CLI agent follows a standardized pipeline:

1. **Ideation** — Generate a research idea and experiment plan; self-review for up to 3 iterations.
2. **Experiments** — Write and execute code, collect results; self-review for up to 3 iterations.
3. **Paper Writing** — Produce a paper; self-review for up to 3 iterations.
4. **Review** — Evaluate via Stanford Agentic Reviewer and triple peer review (all three agents review each paper alongside its code).

## Conferences & areas

Seeds span multiple CS conferences and two compute platforms. Hardware: **1× RTX A6000 (48GB), 4 CPUs, 60GB RAM** (main experiments); **H100 (80GB)** re-run for GPU seeds.

| Platform | Seeds | Target conferences |
|---|---|---|
| **GPU (8)** | AI for Biology, Computer Vision, Datasets & Benchmarks, Generative Models, Interpretability, NLP, Privacy in ML, Supervised Representation Learning | ICLR, NeurIPS, ICML, CVPR, ACL, EMNLP |
| **CPU (5)** | Causal Learning, Compiler Optimization, Data Integration & Cleaning, Operating System Design, Probabilistic Methods | OSDI, SOSP, SIGMOD, VLDB, PLDI, POPL |

## Repo structure

```
ResearchArena/
├── papers/                     # 117 agent-generated papers
│   └── {claude,codex,kimi}/{seed}_trial{N}/
│       ├── paper.pdf, paper.tex, references.bib
│       ├── idea.json, plan.json, proposal.md
│       ├── reviews.json              # 3 peer reviews
│       ├── stanford_review.json      # SAR review
│       └── exp/                      # experiment code (.py/.sh)
├── researcharena/              # the benchmark harness
│   ├── stages/                 # ideation / experiment / paper / review
│   ├── templates/              # domain guidelines (ml/systems/databases/pl/theory/security)
│   └── utils/                  # agent_runner, tracker, checkpoint, …
└── Dockerfile[.cpu]            # agent containers
```

## Setup

```bash
pip install -e .

# Containers (Docker or Podman)
# GPU image: PyTorch 2.6 + CUDA 12.4 + transformers/datasets/accelerate/…
docker build -t researcharena/agent:latest .
# CPU image: Python 3.11 + scipy/sklearn/networkx/sympy/z3-solver/…
docker build -f Dockerfile.cpu -t researcharena/agent-cpu:latest .

# For rootless podman, add --userns=host:
#   podman build --userns=host -t researcharena/agent:latest .
#   podman build --userns=host -f Dockerfile.cpu -t researcharena/agent-cpu:latest .

# Install the CLI agents (claude, codex, kimi) on the host — they are NOT
# baked into the image. agent_runner.py mounts each binary + its auth
# (~/.claude, ~/.codex, ~/.kimi) into the container at runtime, so log in
# once on the host with `claude login` / `codex login` / `kimi login` and
# you're done.

# Optional: API keys / tokens (forwarded into the container if set)
export ANTHROPIC_API_KEY=sk-ant-...      # if not using `claude login`
export OPENAI_API_KEY=sk-...             # if not using `codex login`
export MOONSHOT_API_KEY=sk-...           # if not using `kimi login`
export HF_TOKEN=hf_...                   # needed for gated HuggingFace models
export WANDB_API_KEY=...                 # optional, for experiment logging
```

## Usage

```bash
researcharena run --seed "computer vision" --agent claude --platform gpu
```

That's it — the pipeline handles ideation, experiments, paper writing, and review end-to-end. Swap `--agent` for `codex` or `kimi`, and `--platform` for `cpu` to pick a different configuration.

Everything is configurable — swap agents, change self-review intensity, give the agent more ideas to try, or raise the acceptance bar. The main knobs in `configs/*.yaml`:

| Knob | What it does |
|---|---|
| `agent.type` / `agent.model` | Which CLI agent runs the research (claude / codex / kimi / minimax) and which model it uses. |
| `agent.max_turns`, `ideation_timeout`, `paper_timeout` | Per-stage turn and wall-clock budgets for the researcher. |
| `self_review.max_retries_per_gate` | How many times each gate (idea / experiment / paper) can send itself back for revision. |
| `self_review.thresholds.{idea,experiment,paper}` | The score each self-review must clear to pass (default: idea 8, experiment 6, paper 8). |
| `experiment.max_experiment_retries_per_idea` | How many times the agent can retry failed experiments before abandoning an idea. |
| `pipeline.max_ideas_per_seed` | How many fresh research ideas to try on one seed before giving up. |
| `review.agents` | Which CLI agents act as peer reviewers, with optional per-reviewer model/timeout. |
| `review.accept_threshold` | Cutoff for accept vs. revise vs. reject after peer review. |

See [`configs/8xa6000.yaml`](configs/8xa6000.yaml) for a full annotated example.

## Review pipeline

Each paper is evaluated by two complementary reviewers:

**1. Peer Review (PR) — code-aware.** All three CLI agents (Claude Code, Codex, Kimi Code) review every paper with read-only access to the entire workspace (code, logs, `results.json`). This lets them verify whether reported numbers match the actual artifacts, flag incomplete experiments, and catch fabricated claims. Reviewers score 9 dimensions (novelty, soundness, significance, clarity, reproducibility, experimental rigor, references, reference integrity, results integrity), decide accept/revision/reject, and run online verification of citations against arXiv, Semantic Scholar, and CrossRef.

**2. Stanford Agentic Reviewer (SAR) — external, PDF-only.** We additionally submit each paper to [paperreview.ai](https://paperreview.ai) (Stanford ML Group) for an independent, ICLR-calibrated score. SAR sees only the PDF, not the code — so comparing SAR against our code-aware PR isolates how much reviewers miss when they can't inspect artifacts.

Scores use a 0–10 scale. Acceptance threshold: 8. Score 5–7 triggers revision. Score < 5 is rejected.

## Citation

If you use Research Arena or its data, please cite:

```
@misc{researcharena2026,
  title   = {How Far Are We From True Auto Research?},
  author  = {Zhang, Zhengxin and Wang, Ning and Galhotra, Sainyam and Cardie, Claire},
  year    = {2026},
  note    = {Cornell University. \url{https://youarespecialtome.github.io/ResearchArena/}}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
