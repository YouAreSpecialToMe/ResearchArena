# How Far Are We From True Auto Research?

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Website](https://img.shields.io/badge/Website-ResearchArena-5eead4.svg)](https://youarespecialtome.github.io/ResearchArena/)
[![GitHub stars](https://img.shields.io/github/stars/YouAreSpecialToMe/ResearchArena?style=social)](https://github.com/YouAreSpecialToMe/ResearchArena/stargazers)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ed.svg)](https://www.docker.com/)

An in-depth analysis of frontier CLI agents — **Claude Code (Opus 4.6)**, **Codex (GPT-5.4)**, **Kimi Code (K2.5)** — conducting end-to-end research across diverse fields and compute resources.

**Full write-up:** [blog post](https://youarespecialtome.github.io/ResearchArena/)

**Authors:** Zhengxin Zhang\*, Ning Wang\*, Sainyam Galhotra, Claire Cardie — Cornell University. (\*Order determined by coin flip.)

---

## At a glance

- **117 agent-generated papers** (39 per agent, 3 trials × 13 seeds) spanning both GPU and CPU domains
- **351 code-aware peer reviews** (3 CLI-agent reviewers per paper) + **117 Stanford Agentic Reviewer (SAR) scores**
- **Manual annotation of every paper** for results mismatches, implementation mismatches, and fabricated references

## Key findings

**Stanford Agentic Review (0–10, ICLR scale):**

Claude Code **5.45** > FARS **5.06** > Codex **4.93** > Kimi Code **4.24**. Claude Code beats Analemma's $186K FARS system with $200 in API spend, but still trails the average ICLR-accepted paper (5.59).

**Peer Review (0–10, ICLR scale, code-aware):**

Claude Code **4.59** > Codex **4.51** > Kimi Code **3.38**. PR is stricter than SAR — all agents drop because reviewers have read-only access to the code, logs, and `results.json`.

### Human-annotated SAR accept decisions

| | Accept | Reject | Accept % |
|---|---|---|---|
| ICLR Accepted (baseline) | 76 | 24 | 76.0% |
| ICLR Weighted (32% acc / 68% rej) | 59.7 | 40.3 | 59.7% |
| ICLR Rejected (baseline) | 52 | 48 | 52.0% |
| **Claude Code** | 16 | 23 | **41.0%** |
| FARS (Analemma) | 22 | 80 | 21.6% |
| **Codex** | 5 | 34 | **12.8%** |
| **Kimi Code** | 2 | 37 | **5.1%** |

### Paper–artifact integrity (manual, n=39 per agent)

| | Claude | Codex | Kimi |
|---|---|---|---|
| Results mismatch only | 6 (15%) | 2 (5%) | 4 (10%) |
| Setting mismatch only | 10 (26%) | 1 (3%) | 5 (13%) |
| Both (results + setting) | 12 (31%) | 2 (5%) | 30 (77%) |
| Fake reference | 14 (36%) | 3 (8%) | 28 (72%) |

**Takeaways:**
- Current CLI agents trail human-authored papers on SAR acceptance by a wide margin.
- **Claude Code** is the strongest agent overall (41% SAR accept, highest PR score, full-stack profile).
- **Codex** is the most *trustworthy* — lowest integrity issues, 87% empirical studies, strongest on 7/9 reliability dimensions.
- **Kimi Code** shows the most paper–artifact divergence — 77% of papers have *both* results and setting mismatches, 72% cite fake references.
- Compute is **not** the bottleneck — upgrading A6000 → H100 yields no consistent improvement (Codex 4.51 → 4.26).
- SAR misses code-level failures that only a code-aware peer review can catch.

See [Case Studies: Paper–Artifact Divergence](https://youarespecialtome.github.io/ResearchArena/#case-studies) for six illustrative failure modes with embedded PDFs.

## Three distinct research personas

| Agent | Persona | Signature pattern |
|---|---|---|
| **Claude Code** | Full-stack researcher | 46% empirical studies + 33% novel methods; longest papers; strongest on creative dimensions (novelty 5.42, significance 5.23). |
| **Codex** | Empirical scientist | 87% empirical studies; lowest integrity issues; explicitly scopes work as *pilot / feasibility / negative* with matched-budget comparisons. |
| **Kimi Code** | System builder | 79% "novel methods" with acronym names ("Name: Subtitle"); confident, marketing-leaning; but low actual novelty — often repackages existing work. |

## What this does

Given a seed field (e.g., "computer vision", "compiler optimization"), the pipeline:

1. **Launches a CLI agent** with appropriate compute access (GPU or CPU), network, and packages
2. **The agent produces a structured research proposal** — `proposal.md`, `plan.json`, and `idea.json`
3. **Self-review gates** check quality at each stage before proceeding
4. **The agent follows its experiment plan** step-by-step, producing `results.json`
5. **The agent writes a LaTeX paper** from the proposal, plan, and results
6. **Other CLI agents review it** — with read-only workspace access, and online search to verify novelty and citations
7. If rejected, the pipeline iterates — refine the idea, retry experiments, or try a new idea entirely

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
│   ├── claude/{seed}_trial{N}/
│   │   ├── paper.pdf, paper.tex, references.bib
│   │   ├── idea.json, plan.json, proposal.md
│   │   ├── reviews.json              # 3 peer reviews
│   │   ├── stanford_review.json      # SAR review
│   │   └── exp/                      # experiment code (.py/.sh)
│   ├── codex/…
│   └── kimi/…
├── blog/                       # website (gh-pages branch mirrors this)
│   ├── index.html, index_zh.html, papers.html
│   └── assets/plots/           # all figures used in the blog
├── analysis/                   # baseline assessment data + annotations
│   ├── iclr2025_baseline/      # 300 ICLR papers, SAR assessments
│   ├── stanford_reviews/       # FARS + ICLR SAR fetches
│   └── sar_annotations_*.json  # manual annotations
├── researcharena/              # the benchmark harness
│   ├── stages/                 # ideation / experiment / paper / review
│   ├── templates/              # domain guidelines (ml/systems/databases/pl/theory/security)
│   └── utils/                  # agent_runner, tracker, checkpoint, …
├── paper_viewer.py             # Streamlit app to browse papers + reviews
├── sar_annotator.py            # Streamlit app for manual SAR annotation
├── configs/                    # YAML configs
├── scripts/                    # SLURM / launcher scripts
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

# CLI agent binaries (claude, codex, kimi) are not installed in the image —
# agent_runner.py mounts them from the host at runtime, so install them
# locally first (e.g. `npm install -g @anthropic-ai/claude-code`).

# API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export MOONSHOT_API_KEY=sk-...
```

## Usage

### Run a single seed

```bash
researcharena run --seed "computer vision" --agent claude --platform gpu
researcharena run --seed "compiler optimization" --agent codex --platform cpu
```

### Run with checkpoint/resume (recommended for SLURM)

```bash
python -m researcharena.run_resumable \
  --config configs/claude_cpu.yaml \
  --seed "compiler optimization" \
  --platform cpu \
  --workspace outputs/claude_t1_compiler_optimization
```

A `checkpoint.json` is saved after every step; interrupted jobs resume from the last completed step.

### Run 8 parallel experiments on SLURM

```bash
sbatch scripts/slurm_claude_gpu.sh
sbatch scripts/slurm_codex_gpu.sh
sbatch scripts/slurm_kimi_gpu.sh
```

### Browse all 117 papers + reviews (Streamlit)

```bash
streamlit run paper_viewer.py
```

Filter by agent, domain, platform, decision; expand any paper to see its 3 peer reviews (with 9-dimension scores), Stanford AI Review, idea description, and experiment code listing. Includes a "My Notes" tab per paper that persists to `papers/human_comments.json`.

### Manually annotate ambiguous SAR reviews

```bash
streamlit run sar_annotator.py
```

Classify each SAR assessment as Accept / Reject / Unclear; progress auto-saves to `analysis/sar_annotations.json`.

## Review pipeline

Each paper is evaluated by 3 independent CLI agents (excluding the researcher agent) with read-only workspace access. Reviewers score 9 dimensions (novelty, soundness, significance, clarity, reproducibility, experimental rigor, references, reference integrity, results integrity), decide accept/revision/reject, and run online verification of citations against arXiv, Semantic Scholar, and CrossRef.

| Researcher | Reviewers |
|---|---|
| Claude Code | Codex, Kimi Code |
| Codex | Claude Code, Kimi Code |
| Kimi Code | Claude Code, Codex |

Scores use a 0–10 scale. Acceptance threshold: 8. Score 5–7 triggers revision. Score < 5 is rejected.

## Configuration

See [`configs/8xa6000.yaml`](configs/8xa6000.yaml) for the full example. Key knobs:

- `platforms.gpu.resources` / `platforms.cpu.resources` — compute budget
- `agent.max_turns`, `ideation_timeout`, `paper_timeout`
- `self_review.threshold` (default 6) and `max_retries_per_gate` (default 2)
- `review.agents` — list of reviewer CLI agents + models
- `pipeline.max_ideas_per_seed` — how many fresh ideas to try before giving up

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
