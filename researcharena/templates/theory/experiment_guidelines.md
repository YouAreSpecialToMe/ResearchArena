# Experiment Guidelines (Theory)

Guidelines for the computational/experimental component of a theory paper.
Theory papers are fundamentally different from ML papers -- the primary
contribution is a theorem and its proof, not an empirical result. However,
experiments can complement theory, and this benchmark requires structured
output.

Distilled from practices in the STOC/FOCS/SODA/COLT communities and
experimental algorithmics (ESA, ALENEX, SEA).

## When theory papers have experiments

Most theory papers at STOC/FOCS have NO experiments. The contribution is
purely mathematical. Experiments appear in theory papers when:

| Situation | What experiments show |
|---|---|
| New practical algorithm | The algorithm runs fast on real inputs, not just in theory |
| Approximation algorithm | The practical approximation ratio is much better than worst-case |
| Theoretical prediction | Empirical confirmation that a predicted phase transition or threshold exists |
| Comparison with heuristics | The theoretically-grounded algorithm competes with or beats heuristics |
| Lower bound tightness | Instances that approach the theoretical lower bound |
| Parameterized algorithm | The algorithm is fast when the parameter is small in practice |

If your contribution is a pure lower bound, hardness result, structural
theorem, or complexity class separation, you likely need NO experiments.

## Phase 1: The Primary Task -- Develop the Proof

Before any computation, your main work is mathematical.

### 1.1 Formalize the theorem statement

- State the theorem precisely: computational model, input assumptions,
  claimed bound (time, space, approximation ratio, competitive ratio, etc.)
- Verify the statement does not contradict known lower bounds
- Check boundary cases and degenerate inputs

### 1.2 Develop the proof

- Start with a proof sketch: identify the key lemmas needed
- Prove each lemma. For each, identify whether it uses:
  - A known technique applied in a new way
  - A new technique (this is where novelty lies)
  - A routine calculation (defer to appendix)
- Verify the proof is complete -- no gaps, no hand-waving at critical steps
- Check the constants: if you claim O(n log n), verify the proof actually
  achieves this and not O(n log^2 n) with a hidden factor

### 1.3 Construct extremal examples (for lower bounds)

- If proving a lower bound, construct an explicit hard instance or family
  of instances
- Verify the lower bound holds on your construction by direct calculation
- Check that the construction is valid in your computational model

## Phase 2: Experimental Validation (when applicable)

### 2.1 Decide if experiments add value

Ask yourself:
- Does my algorithm have practical relevance, or is it primarily of
  theoretical interest? (An O(n^{1.99}) algorithm using fast matrix
  multiplication is theoretically interesting but may not be practical)
- Would experiments help a reader understand the result better?
- Are there natural benchmark instances to test on?

If the answer to all three is no, skip to Phase 3.

### 2.2 Implementation

- Implement the algorithm cleanly and correctly. Correctness matters
  more than micro-optimization.
- Use a standard language: C++, Python, or Java are most common in
  experimental algorithmics
- For comparison, implement or obtain implementations of:
  - The previous best algorithm for the same problem
  - Standard practical heuristics (even if they lack theoretical guarantees)
  - Simple baselines (naive algorithm, greedy, random)

### 2.3 Benchmark instances

Use standard benchmarks appropriate to your area:

| Area | Benchmark sources |
|---|---|
| Graph algorithms | SNAP (snap.stanford.edu), DIMACS graph challenges, Network Repository |
| Optimization | MIPLIB (mixed-integer programming), TSPLIB, OR-Library |
| Satisfiability | SAT Competition benchmarks, SATLIB |
| Computational geometry | standard point sets, CGAL benchmarks |
| String algorithms | Pizza&Chili corpus, genome datasets |
| Sorting/searching | random, nearly-sorted, adversarial distributions |
| Learning theory | UCI datasets, synthetic distributions matching assumptions |

Also test on:
- Random instances (specify the distribution precisely)
- Adversarial / worst-case instances (designed to stress your algorithm)
- Instances at multiple scales (to observe asymptotic behavior)

### 2.4 What to measure

- **Running time**: wall-clock time, preferably also operation counts
  (comparisons, memory accesses, arithmetic operations) to compare with
  theoretical bounds
- **Memory usage**: peak and average
- **Solution quality**: for approximation algorithms, report the actual
  approximation ratio achieved (compare solution value to optimal or best
  known)
- **Scaling behavior**: plot running time vs. input size on a log-log
  scale to verify the theoretical exponent
- Run multiple trials on randomized inputs; report median and
  interquartile range

### 2.5 What to show

The experiments should answer:

1. **Does the algorithm work in practice?** Running time on real inputs
   vs. the theoretical worst-case bound
2. **How does it compare?** Side-by-side with the previous best algorithm
   and practical heuristics
3. **Is the worst case tight?** Do hard instances approach the theoretical
   bound, or is typical performance much better?

## Phase 3: Producing results.json (REQUIRED)

This benchmark requires a results.json file for every paper, including
purely theoretical papers. The format depends on whether your paper
includes experiments.

### For purely theoretical papers (no experiments)

```json
{
  "contribution_type": "theoretical",
  "main_results": [
    {
      "type": "theorem",
      "statement": "For any graph G on n vertices and m edges, the maximum flow can be computed in O(m * n^{2/3}) time.",
      "improves_upon": {
        "previous_bound": "O(m * n)",
        "reference": "[Author, Venue Year]"
      },
      "proof_technique": "Blocking flows with layered graph decomposition",
      "proof_status": "complete"
    },
    {
      "type": "lower_bound",
      "statement": "Any comparison-based algorithm for this problem requires Omega(n log n) time.",
      "proof_technique": "Adversary argument",
      "proof_status": "complete"
    }
  ],
  "key_lemmas": [
    {
      "statement": "The layered graph has at most O(n^{1/3}) layers.",
      "role": "Enables the main runtime analysis"
    }
  ],
  "open_problems": [
    "Close the gap between the O(m * n^{2/3}) upper bound and the Omega(m) lower bound."
  ],
  "config": {
    "contribution_type": "theoretical",
    "computational_model": "word RAM",
    "problem_domain": "graph algorithms"
  }
}
```

### For theory papers with experiments

```json
{
  "contribution_type": "theoretical_with_experiments",
  "main_results": [
    {
      "type": "theorem",
      "statement": "We give a (1 + epsilon)-approximation algorithm for metric TSP running in time O(n^2 * 2^{1/epsilon}).",
      "improves_upon": {
        "previous_bound": "O(n^3 * 2^{1/epsilon})",
        "reference": "[Author, Venue Year]"
      },
      "proof_technique": "Hierarchical decomposition with sparse spanners",
      "proof_status": "complete"
    }
  ],
  "experiments": {
    "our_algorithm": {
      "benchmark": "TSPLIB",
      "instances_solved": 42,
      "median_time_seconds": 1.23,
      "median_approx_ratio": 1.02,
      "max_approx_ratio": 1.08
    },
    "baselines": {
      "previous_algorithm": {
        "median_time_seconds": 15.7,
        "median_approx_ratio": 1.02
      },
      "greedy_heuristic": {
        "median_time_seconds": 0.05,
        "median_approx_ratio": 1.35
      }
    },
    "scaling": {
      "input_sizes": [1000, 5000, 10000, 50000, 100000],
      "our_times": [0.1, 0.8, 2.1, 18.5, 52.3],
      "fitted_exponent": 1.95,
      "theoretical_exponent": 2.0
    }
  },
  "config": {
    "contribution_type": "theoretical_with_experiments",
    "computational_model": "word RAM",
    "problem_domain": "combinatorial optimization",
    "language": "C++",
    "hardware": "Intel i9, 64GB RAM",
    "total_runtime_minutes": 240
  }
}
```

Adapt the structure to your specific contribution. The key requirements:
structured, machine-readable, complete, and mathematically precise.

## Phase 4: Common Pitfalls

### Mathematical pitfalls
- DO NOT claim a bound you have not fully proved -- a proof sketch with
  gaps is not a proof
- DO NOT ignore the computational model -- an O(n) algorithm in the
  real RAM is not the same as O(n) in the word RAM
- DO NOT confuse expected time with worst-case time, or amortized with
  per-operation bounds
- DO NOT hide polynomial factors in O-tilde notation without stating them
- DO NOT forget to handle degenerate cases (empty input, disconnected
  graphs, zero-weight edges, etc.)

### Experimental pitfalls (when experiments are included)
- DO NOT compare wall-clock time of a carefully optimized implementation
  against an unoptimized baseline
- DO NOT test only on random instances -- random graphs are structurally
  different from real-world graphs
- DO NOT test only on small instances where asymptotic behavior is not
  visible
- DO NOT report best-of-N runs -- report median or mean
- DO NOT claim practical superiority from theory alone without running
  the algorithm

## Completeness Checklist

Before finishing, verify:
- [ ] Main theorem is precisely stated with model and bounds
- [ ] Proof is complete with no gaps
- [ ] All lemmas are proved (or cited if previously known)
- [ ] Extremal examples constructed (for lower bounds)
- [ ] Constants and log factors are correct
- [ ] results.json contains all theorem statements and proof summaries
- [ ] If experiments are included: benchmarks are standard, comparisons
      are fair, scaling behavior is shown
- [ ] If experiments are included: implementation is correct (verified
      against small instances with known answers)
