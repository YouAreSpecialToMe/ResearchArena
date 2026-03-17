# Experiment Guidelines (Programming Languages)

Distilled from the SIGPLAN Empirical Evaluation Checklist, best practices from
PLDI/POPL/OOPSLA/ICFP artifact evaluation, and established norms in static
analysis, compiler, and verification research.

## Key difference from ML experiments

PL experiments are fundamentally different from ML experiments. There is
usually no training, no loss curves, and no random seeds. Instead, PL
experiments involve formal proofs, running tools on codebases, measuring
compilation performance, or synthesizing programs. Choose the right
evaluation methodology for your contribution type.

## Phase 1: Experiment Design (do this BEFORE writing code)

### 1.1 Identify your contribution type and evaluation strategy

| Contribution type | Primary evaluation | What to measure |
|---|---|---|
| Type system / formal semantics | Soundness proof | Proof correctness (pen-and-paper or mechanized) |
| Static analysis | Run on real codebases | Precision, recall, false positive rate, analysis time |
| Compiler optimization | Benchmark suites | Speedup, compile time overhead, code size |
| Program synthesis | Synthesis benchmarks | Synthesis time, success rate, program quality |
| Verification / model checking | Verification benchmarks | Verification time, proof size, expressiveness |
| DSL design | Case studies + performance | Expressiveness, usability, performance vs. hand-written code |
| Runtime system | Standard workloads | Throughput, latency, memory usage, pause times |
| Bug finding tool | Run on known buggy code | Bugs found, false positives, analysis time |

### 1.2 Formulate your claim

Before any implementation, write down:
- **What is your hypothesis?** State it as a testable claim.
  Examples:
  - "Our type system is sound: well-typed programs do not get stuck."
  - "Our analysis finds 30% more null-pointer bugs than tool X with fewer
    false positives on the same benchmarks."
  - "Our optimization produces code that is 15% faster on SPEC CPU 2017 with
    less than 5% compile time overhead."
- **What evidence would convince a skeptical reader?**
- **What would DISPROVE your claim?** Design experiments that could fail.

### 1.3 Select baselines fairly

- Include at least 2 meaningful baselines:
  - One established tool or technique (the current standard)
  - One recent published method (the current state of the art)
- Run all baselines with equivalent effort (same machine, same time limits,
  same benchmarks)
- If a baseline tool is unavailable, cite published numbers and clearly state
  you didn't rerun it
- Never compare against intentionally weak baselines to inflate your results
- If comparing against commercial tools, be transparent about version and
  configuration

## Phase 2: Evaluation by Contribution Type

### 2.1 Type systems and formal semantics

**Soundness proofs** are the primary evidence:
- State your main theorem precisely (e.g., progress + preservation, or
  type safety as a single statement)
- **Mechanized proofs** (in Coq, Lean, Agda, or Isabelle) are increasingly
  expected and score higher with reviewers:
  - Provide the full proof development as an artifact
  - State which version of the proof assistant you used
  - Document any axioms assumed (e.g., functional extensionality, classical logic)
  - Measure: lines of proof code, time to check, ratio of proof to spec
- **Pen-and-paper proofs**: acceptable but scrutinized more heavily:
  - Include full proofs in the appendix
  - State every lemma and its proof sketch
  - Be explicit about which cases are "routine" vs. non-trivial
- **Supplementary evaluation**: implement a type checker and run it on
  example programs to demonstrate the type system is practical

### 2.2 Static analyses

**Run on real-world codebases**, not just microbenchmarks:
- Use widely-used open-source projects (e.g., from GitHub top-starred repos
  in the target language, Apache projects, or the DaCapo suite for Java)
- Report the following metrics:
  - **Precision**: fraction of reported issues that are real (not false positives)
  - **Recall**: fraction of real issues that are found (if ground truth is known)
  - **False positive rate**: number of false alarms per 1000 lines of code
  - **Bugs found**: real, previously unknown bugs found (and reported/fixed upstream)
  - **Analysis time**: wall-clock time and memory usage
  - **Scalability**: how time/memory grows with program size (lines of code)
- Standard benchmark suites by area:
  - **Java analysis**: DaCapo, XCorpus
  - **C analysis**: Juliet Test Suite (NIST), SPEC CPU 2017, GNU coreutils
  - **Security analysis**: Juliet, OWASP Benchmark, CVE databases
  - **Concurrency analysis**: SCTBench, CDSChecker benchmarks
- Compare against prior tools on the SAME benchmarks they used, plus new ones
- If your analysis finds real bugs, file bug reports and note whether they
  were confirmed by developers

### 2.3 Compiler optimizations

**Use standard benchmark suites**:
- **SPEC CPU 2017**: the gold standard for CPU-bound performance evaluation
- **PolyBench/C**: for polyhedral and loop optimizations
- **DaCapo**: for JVM optimizations
- **Embench**: for embedded systems compilers
- **LLVM test-suite**: for LLVM-based optimizations
- **NPB (NAS Parallel Benchmarks)**: for parallel optimizations

**What to measure**:
- **Speedup**: geometric mean speedup over baseline across benchmark suite
- **Compile time overhead**: additional time to compile with the optimization
- **Code size impact**: change in binary size
- **Correctness**: verify output matches baseline for all benchmarks
- Report per-benchmark results, not just aggregate — some optimizations help
  some programs and hurt others

**Methodology**:
- Run each benchmark at least 3 times, report median
- Use performance counters (perf, PAPI) for micro-architectural analysis
- Pin processes to cores, disable turbo boost, minimize background load
- Report hardware: CPU model, cache sizes, memory, OS, compiler version
- Warm up JIT compilers before measuring (for JVM/JS experiments)

### 2.4 Program synthesis

**What to measure**:
- **Success rate**: fraction of benchmarks solved within the time limit
- **Synthesis time**: time to produce a correct solution (median and distribution)
- **Quality of synthesized programs**: size, readability, performance compared
  to human-written solutions
- **Generalization**: does the synthesizer handle unseen problem variants?

**Standard benchmarks**:
- **SyGuS-Comp benchmarks**: for syntax-guided synthesis
- **Karel benchmarks**: for program synthesis from examples
- **Domain-specific benchmarks**: string transformations (FlashFill),
  SQL queries, tensor computations, etc.

**Methodology**:
- Set a reasonable time limit (e.g., 5 minutes per benchmark)
- Report cumulative solved problems over time (cactus plot)
- Compare against the same benchmarks used by competing synthesizers

### 2.5 Verification and model checking

**What to measure**:
- **Verification time**: time to prove/disprove properties
- **Expressiveness**: what properties can be expressed and verified?
- **Proof burden**: how much annotation or specification is needed from the user?
- **Scalability**: how verification time grows with program size
- **False alarms**: for incomplete methods, what is the false positive rate?

**Standard benchmarks**:
- **SV-COMP**: the standard competition for software verification tools
- **VerifyThis**: verification challenges
- **SMTLIB benchmarks**: for SMT-based verification
- **Boogie/Dafny benchmarks**: for deductive verification

### 2.6 DSL design and implementation

**Evaluate along multiple axes**:
- **Expressiveness**: can the DSL express the intended class of programs?
  Demonstrate with non-trivial case studies.
- **Performance**: compare DSL-generated code against hand-written code in a
  general-purpose language
- **Usability**: if applicable, user studies or LOC comparisons
- **Correctness**: prove or test that the DSL compiler/interpreter is correct

## Phase 3: Rigorous Evaluation

### Handling non-determinism
- PL tools are often deterministic. If yours is, state this explicitly (no
  need for multiple random seeds)
- If your tool has non-determinism (randomized search, heuristic ordering,
  parallel execution), run multiple times and report statistics
- For performance measurements, always run multiple times to account for
  system variance — report median and min/max or IQR

### Threats to validity
- **Internal validity**: could something other than your technique explain the
  results? (e.g., differences in implementation quality, language version)
- **External validity**: do your benchmarks represent real-world usage? Are
  they diverse enough?
- **Construct validity**: do your metrics actually measure what you claim?
  (e.g., false positive rate depends on ground truth quality)

### Avoid common pitfalls
- DO NOT cherry-pick benchmarks where your tool wins
- DO NOT compare a polished implementation against a research prototype
- DO NOT set time limits that favor your tool
- DO NOT count "warnings" or "potential issues" as bugs found without validation
- DO NOT claim soundness without a proof
- DO NOT report only aggregate numbers — per-benchmark breakdowns reveal where
  your technique helps and where it doesn't
- DO NOT use microbenchmarks alone — real-world code is essential
- DO NOT ignore compile time or analysis time — a 2x speedup that costs 10x
  compile time may not be worth it

## Phase 4: What to Save

Save everything needed to write the paper:

```
results.json          # structured results (see format below)
figures/              # comparison plots, performance charts, cactus plots
proofs/               # mechanized proof development (if applicable)
benchmarks/           # benchmark programs or pointers to them
```

### results.json format (adapt to your contribution type)

```json
{
  "method": {
    "benchmarks_solved": 47,
    "total_benchmarks": 50,
    "median_time_seconds": 12.3,
    "false_positives": 5,
    "bugs_found": 23
  },
  "baselines": {
    "tool_A": {
      "benchmarks_solved": 38,
      "total_benchmarks": 50,
      "median_time_seconds": 45.7,
      "false_positives": 12,
      "bugs_found": 18
    }
  },
  "per_benchmark": {
    "benchmark_1": {
      "method_time": 2.1,
      "method_result": "safe",
      "tool_A_time": 5.3,
      "tool_A_result": "safe"
    }
  },
  "config": {
    "experiment_type": "static_analysis_evaluation",
    "benchmark_suite": "DaCapo + XCorpus",
    "time_limit_seconds": 300,
    "hardware": "Intel Xeon E5-2680 v4, 128GB RAM",
    "os": "Ubuntu 22.04",
    "tool_version": "1.0.0",
    "total_runtime_minutes": 240
  }
}
```

### For formal contributions, also save:

```json
{
  "proof": {
    "assistant": "Coq 8.18",
    "total_loc": 5200,
    "spec_loc": 800,
    "proof_loc": 4400,
    "check_time_seconds": 120,
    "axioms": ["functional_extensionality"],
    "main_theorems": ["soundness", "completeness"]
  }
}
```

## Artifact Evaluation Checklist

SIGPLAN venues have artifact evaluation. Before finishing, verify:
- [ ] Claim is clearly stated and testable
- [ ] Evaluation type matches the contribution type
- [ ] At least 2 meaningful baselines compared fairly
- [ ] Benchmarks are standard or well-justified
- [ ] Per-benchmark results reported (not just aggregates)
- [ ] Analysis time and memory usage reported
- [ ] Threats to validity discussed
- [ ] All tool versions, hardware, and OS documented in results.json
- [ ] Figures saved for key results
- [ ] Negative results reported honestly (if any)
- [ ] Proof development compiles cleanly (if applicable)
- [ ] Artifact is self-contained and reproducible (for AE submission)
