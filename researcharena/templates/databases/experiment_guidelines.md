# Experiment Guidelines (Databases)

Distilled from SIGMOD/VLDB experimental methodology standards, the SIGMOD
reproducibility initiative, TPC benchmark specifications, and database
systems evaluation best practices.

## Phase 1: Experiment Design (do this BEFORE writing code)

### 1.1 Formulate your claim

Before any implementation, write down:
- **What is your hypothesis?** State it as a testable claim.
  Example: "Our join algorithm reduces execution time by 2x over hash join
  on skewed TPC-H workloads because it avoids partition imbalance."
- **What evidence would convince a skeptical database reviewer?**
- **What would DISPROVE your claim?** Design experiments that could fail — if
  your experiment cannot possibly produce a negative result, it's not
  informative.

### 1.2 Choose the right experiment type

Not all database research requires building a full system. Choose what fits
your claim:

| Claim type | Experiment type | What to measure |
|---|---|---|
| "Our method outperforms X" | Empirical comparison | Latency, throughput on benchmarks |
| "Component A is critical" | Ablation / micro-benchmark | Performance with/without A |
| "This scales to large data" | Scalability experiment | Metrics vs. data size / clients |
| "Our theory predicts X" | Theoretical validation | Synthetic setup with known properties |
| "This is faster/cheaper" | Systems experiment | Latency, throughput, memory, I/O |
| "This structure is better" | Data structure evaluation | Build time, lookup, space overhead |
| "This protocol is correct" | Correctness argument | Formal proof + stress tests |
| "This tradeoff exists" | Sensitivity analysis | Metrics vs. tuning parameters |

### 1.3 Select what to measure

Standard database metrics (report all that are relevant to your claim):

**Performance metrics:**
- Query execution time (wall-clock, report median and percentiles: p50, p95, p99)
- Throughput (transactions/sec for OLTP, queries/sec for analytics)
- Latency distribution (not just average — tail latency matters)

**Resource metrics:**
- Memory consumption (peak and steady-state)
- Storage overhead (bytes per record, space amplification)
- I/O volume (bytes read/written, number of I/O operations)
- CPU utilization and cache miss rates (when relevant)

**Construction/setup metrics (for data structures and indexes):**
- Build time / index construction time
- Bulk load performance
- Insert/update/delete throughput

**Scalability metrics:**
- Report metrics as a function of: data size (scale factor), number of
  concurrent clients/threads, selectivity, data skew
- Show behavior at multiple scales — a single data point is not a
  scalability experiment

### 1.4 Choose benchmarks that test your claim

Use standard benchmarks when possible — they enable comparison with
published work and are familiar to reviewers.

**Analytics / OLAP:**
- TPC-H: standard decision-support benchmark (8 tables, 22 queries,
  scale factors 1-1000)
- TPC-DS: more complex decision-support (24 tables, 99 queries, more
  realistic schema and query mix)
- Star Schema Benchmark (SSB): simplified star-schema analytics
- JOB (Join Order Benchmark): real-world join queries on IMDb data,
  specifically designed to stress cardinality estimation

**Transactional / OLTP:**
- TPC-C: standard OLTP benchmark (warehouse model, 5 transaction types)
- YCSB (Yahoo Cloud Serving Benchmark): key-value workloads with
  configurable read/write ratio, request distribution, record size
- TATP (Telecom Application Transaction Processing)
- SmallBank: simple banking transactions for concurrency control evaluation

**Specialized:**
- Microbenchmarks: custom workloads that isolate your specific contribution
  (e.g., point lookups, range scans, skewed inserts). Always pair these
  with a standard benchmark.
- Real-world datasets: use publicly available real data when it strengthens
  your claim (e.g., OpenStreetMap for spatial, Wikipedia edit history for
  temporal, LDBC Social Network for graph workloads)

### 1.5 Select baselines fairly

- Include at least 2 meaningful baselines:
  - One established system or textbook approach (e.g., PostgreSQL's hash
    join, B-tree index, 2PL concurrency control)
  - One recent published method that represents the state of the art
- Use well-known open-source systems as baselines when possible:
  PostgreSQL, MySQL, DuckDB, SQLite, LevelDB/RocksDB, or the specific
  system your technique extends
- Run all baselines on the same hardware with equivalent tuning effort
  - This means: same buffer pool size, same number of threads, same
    storage device, same compiler optimization level
  - Tune baseline systems properly — an untuned PostgreSQL is not a fair
    baseline (configure shared_buffers, work_mem, effective_cache_size, etc.)
- If a baseline is too expensive to run yourself, cite published numbers
  and clearly state the hardware differences
- Never compare against intentionally weak or misconfigured baselines

### 1.6 Plan ablation and sensitivity studies

- For each novel component in your system/algorithm, plan to disable it
  and measure the impact
- Plan parameter sensitivity experiments: vary key parameters one at a time
  and show how they affect performance
  - Data size (scale factor): e.g., 1GB, 10GB, 100GB
  - Selectivity: vary predicate selectivity from low to high
  - Skew: vary Zipfian parameter from uniform (0.0) to highly skewed (1.5)
  - Number of concurrent clients: e.g., 1, 2, 4, 8, 16, 32, 64
  - For data structures: vary key size, value size, fill factor
- Plan which parameters to vary BEFORE running experiments, not after
  seeing results

### 1.7 Think about confounders

- What else could explain your results besides your technique?
- Is the improvement from your algorithm or from a systems-level trick
  (better memory allocation, SIMD, prefetching)?
- Are you comparing with the same buffer pool size, thread count, and
  storage configuration?
- Could the improvement disappear on different hardware (SSD vs. HDD,
  different CPU architecture, different memory size)?
- Are you measuring warm cache or cold cache performance? Be explicit.
- Is the workload hitting the bottleneck your technique addresses, or
  is something else the bottleneck?

## Phase 2: Implementation

### General principles
- Start simple. Get a minimal prototype working end-to-end first.
- Implement inside an existing system when possible (e.g., extend
  PostgreSQL, modify DuckDB, build on RocksDB) rather than from scratch.
  This gives you a realistic systems context and credible baselines.
- Add complexity one piece at a time. Evaluate each change independently.
- Use well-tested libraries for supporting infrastructure (e.g., folly,
  abseil, jemalloc for memory management).
- Fix random seeds for reproducibility in any randomized components.

### Systems implementation practices
- Profile before optimizing — use perf, VTune, or similar tools to find
  the actual bottleneck
- Control memory allocation — use a custom allocator or at least track
  allocations to report accurate memory usage
- Compile with optimizations (-O2 or -O3) for all systems including
  baselines — debug builds are not valid benchmarks
- Pin threads to cores and disable hyperthreading if evaluating
  concurrent performance
- Use direct I/O (O_DIRECT) or explicitly manage the OS page cache when
  I/O performance matters
- Flush caches between cold-start measurements

### For data structure experiments
- Implement the data structure correctly first, verify with unit tests
- Measure construction cost separately from query performance
- Report space overhead (bytes per key, or total size vs. raw data size)
- Test with realistic key distributions (not just sequential integers)
- Evaluate both point queries AND range queries if the structure supports both

### For query processing experiments
- Use EXPLAIN/EXPLAIN ANALYZE to verify query plans are what you expect
- Ensure all systems use the same data (same generator seed, same loading)
- Run queries in the same order across all systems
- Drop OS caches between cold runs: `sync && echo 3 > /proc/sys/vm/drop_caches`

### For concurrency control experiments
- Verify correctness first: run a checker (e.g., Elle, or custom
  serializability verifier) before measuring performance
- Vary contention levels: low contention (uniform random access) to high
  contention (hot keys with Zipfian skew)
- Report both throughput AND abort rate — high throughput with high abort
  rate is misleading
- Show latency distribution, not just average (tail latency reveals
  problems that averages hide)

## Phase 3: Rigorous Evaluation

### Multiple runs (non-negotiable)
- Run every experiment at least 3 times (5 is better for high-variance
  workloads like OLTP under contention)
- Report median and standard deviation across runs
- Use the SAME configuration for your method and all baselines
- Never report best-of-N runs — always report the median or mean
- For throughput experiments, run for a sufficient duration (at least 60
  seconds after warmup) to reach steady state

### Warmup
- Always include a warmup phase before measurement
- For buffer pool / cache experiments: run the workload once to populate
  caches, then measure subsequent runs
- For throughput experiments: discard the first 10-30 seconds of
  measurements to avoid startup transients
- Report whether numbers are warm-cache or cold-cache, and justify the choice

### Scalability experiments (expected for most database papers)
- Show how performance changes as data size grows:
  - At least 3 scale points (e.g., 1GB, 10GB, 100GB or SF1, SF10, SF100)
  - Ideally show the scaling trend (linear, sublinear, superlinear)
- Show how performance changes as concurrency grows:
  - Vary thread/client count: 1, 2, 4, 8, 16, 32, 64 (or more)
  - Report speedup relative to single-threaded baseline
- Show how performance changes with workload characteristics:
  - Selectivity: from highly selective to full scan
  - Skew: from uniform to highly skewed (Zipfian parameter 0.0 to 1.5)
  - Read/write ratio: from read-heavy to write-heavy

### Statistical reporting
- Report median and standard deviation for all measurements
- For latency measurements, report percentiles (p50, p95, p99)
- If claiming superiority, the improvement should be clear from the
  numbers — if confidence intervals overlap significantly, you cannot
  claim your method is better
- Be careful with geometric means for summarizing across queries — they
  can hide regressions on individual queries

### Avoid common evaluation mistakes
- DO NOT tune your system on the test queries and leave baselines untuned
- DO NOT compare your optimized C++ code against an unoptimized baseline
- DO NOT report only the queries/workloads where your method wins
- DO NOT claim scalability from only two data points
- DO NOT ignore regression on some queries while highlighting overall
  improvement — report per-query results
- DO NOT measure on unrealistically small data (a few MB) and claim it
  generalizes — database techniques must work at scale
- DO NOT forget to report the system configuration (buffer pool size,
  number of threads, storage device type, OS version)

## Phase 4: What to Save

Save everything needed to write the paper:

```
results.json          # structured results (see format below)
figures/              # comparison plots, scalability curves, latency CDFs
```

### results.json format

```json
{
  "method": {
    "tpch_sf10": {
      "total_time_sec": {"median": 12.34, "std": 0.45},
      "q1_time_ms": {"median": 340, "std": 12},
      "q3_time_ms": {"median": 890, "std": 25}
    },
    "throughput_txn_per_sec": {"median": 45000, "std": 1200},
    "memory_mb": 512,
    "index_build_time_sec": 8.3,
    "storage_overhead_mb": 128
  },
  "baselines": {
    "postgresql": {
      "tpch_sf10": {
        "total_time_sec": {"median": 28.91, "std": 0.82}
      }
    },
    "duckdb": {
      "tpch_sf10": {
        "total_time_sec": {"median": 15.67, "std": 0.33}
      }
    }
  },
  "ablations": {
    "without_component_A": {
      "tpch_sf10": {
        "total_time_sec": {"median": 19.45, "std": 0.56}
      }
    }
  },
  "scalability": {
    "sf1": {"total_time_sec": {"median": 1.2, "std": 0.05}},
    "sf10": {"total_time_sec": {"median": 12.34, "std": 0.45}},
    "sf100": {"total_time_sec": {"median": 145.6, "std": 3.2}}
  },
  "config": {
    "experiment_type": "systems_evaluation",
    "benchmarks": ["TPC-H", "JOB"],
    "scale_factors": [1, 10, 100],
    "hardware": "64-core Intel Xeon, 256GB RAM, NVMe SSD",
    "os": "Ubuntu 22.04",
    "compiler": "gcc 12.2 -O3",
    "buffer_pool_mb": 4096,
    "threads": 16,
    "runs_per_experiment": 5,
    "warmup_runs": 2,
    "total_runtime_hours": 48
  }
}
```

Adapt the structure to your experiment type. The key requirement:
structured, machine-readable, complete, and honest.

## Reproducibility Checklist

Before finishing, verify:
- [ ] Claim is clearly stated and testable
- [ ] Experiment type matches the claim
- [ ] Standard benchmark(s) used (TPC-H, TPC-C, YCSB, JOB, etc.)
- [ ] At least 2 meaningful baselines compared fairly (same hardware, fair tuning)
- [ ] Results from 3+ runs with median and standard deviation
- [ ] Scalability experiment with at least 3 data sizes
- [ ] Parameter sensitivity study (selectivity, skew, concurrency, etc.)
- [ ] Ablation study for each novel component
- [ ] System configuration fully documented (hardware, OS, buffer sizes, threads)
- [ ] Warm/cold cache behavior specified
- [ ] Figures saved for key results (bar charts, line plots, latency CDFs)
- [ ] Negative results or regressions reported honestly (if any)
