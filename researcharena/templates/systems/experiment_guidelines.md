# Experiment Guidelines (Systems)

Distilled from OSDI/SOSP artifact evaluation criteria, "How to Evaluate Systems
Research" guidelines, and established systems benchmarking practices.

## Phase 1: Experiment Design (do this BEFORE writing code)

### 1.1 Formulate your claim

Before any implementation, write down:
- **What is your hypothesis?** State it as a testable claim.
  Example: "System X achieves 3x higher throughput than baseline Y on workload Z
  because design decision W eliminates the bottleneck at component V."
- **What evidence would convince a skeptical systems reviewer?**
- **What would DISPROVE your claim?** Design experiments that could fail -- if your
  system always looks good regardless of workload, your evaluation is not rigorous.

### 1.2 Choose the right experiment type

Systems research does NOT involve model training. Choose what fits your claim:

| Claim type | Experiment type | What to measure |
|---|---|---|
| "Our system outperforms X" | Benchmark comparison | Throughput, latency, resource usage |
| "Component A is critical" | Ablation / microbenchmark | Performance with/without A |
| "This scales better" | Scalability experiment | Performance vs. cores/nodes/data size |
| "This handles failures well" | Fault injection | Recovery time, availability, correctness |
| "This is more efficient" | Resource profiling | CPU%, memory footprint, network bandwidth |
| "This provides stronger guarantees" | Correctness testing | Model checking, stress testing, chaos testing |
| "This works on real workloads" | End-to-end evaluation | Performance on production traces or representative apps |
| "This bottleneck exists" | Measurement study | Profiling, tracing, characterization |

### 1.3 Select what to measure

#### Performance metrics (report ALL that apply)
- **Throughput**: operations/sec, requests/sec, transactions/sec, MB/sec
- **Latency**: report percentiles, not just mean
  - p50 (median), p95, p99, p99.9 -- tail latency matters in systems
  - Mean latency hides important information; always include percentiles
- **Scalability**: how metrics change with increasing cores, nodes, clients,
  data size, or request rate
- **Resource efficiency**: CPU utilization, memory footprint, network
  bandwidth consumed, disk I/O, energy consumption

#### For architecture papers (ISCA, MICRO, ASPLOS)
- **IPC** (instructions per cycle), **CPI** (cycles per instruction)
- **Cache miss rates** (L1, L2, LLC)
- **Branch misprediction rates**
- **Area and power estimates** (from synthesis or analytical models)
- **Speedup over baseline** architecture
- **Energy-delay product** for efficiency claims

#### For networking papers (NSDI, SIGCOMM)
- **Goodput** (application-level throughput)
- **Flow completion time** (FCT) -- report by flow size (small/medium/large)
- **Packet loss rate**, **retransmission rate**
- **Queue occupancy**, **buffer utilization**
- **Convergence time** for routing/control plane changes

### 1.4 Choose benchmarks and workloads

Use standard benchmarks when they exist -- they enable fair comparison with
published work:

| Domain | Standard benchmarks |
|---|---|
| Key-value stores | YCSB (Yahoo Cloud Serving Benchmark), memtier_benchmark |
| Databases / OLTP | TPC-C, TPC-H, TPC-DS, Sysbench |
| Storage / file systems | FIO, Filebench, YCSB, IOzone |
| Web serving | wrk, ab (ApacheBench), siege, hey |
| CPU / general | SPEC CPU 2017, PARSEC, SPLASH-2 |
| Architecture simulation | SPEC CPU, MiBench, MediaBench, MLPerf |
| Networking | iperf3, netperf, Alizadeh's workload generators |
| Machine learning systems | MLPerf Training/Inference, DeepBench |
| Containers / serverless | DeathStarBench, SocialNetwork, HotelReservation |

If no standard benchmark fits your workload:
- Design a representative workload that exercises your system's key features
- Justify WHY this workload is representative (cite real-world traces or
  characterization studies)
- Open-source your benchmark so others can reproduce results

### 1.5 Select baselines fairly

- Compare against real, established systems -- NOT toy reimplementations
  - Good: comparing your KV store against Redis, Memcached, or RocksDB
  - Bad: comparing against "a simple hash table we wrote"
- Include at least 2 baselines:
  - One widely-used production system (the practical standard)
  - One recent research system (the academic state of the art)
- Run all baselines with their recommended configurations and tuning
- If a baseline has tuning knobs, make a good-faith effort to tune it
  (or use the vendor's recommended settings)
- If a baseline is too complex to set up, cite published numbers and
  clearly state you did not rerun it
- Never compare against intentionally misconfigured baselines

### 1.6 Plan microbenchmarks and ablation studies

- **Microbenchmarks**: isolate each key design component and measure its
  contribution independently
  - Example: if your system uses a new indexing structure and a new
    concurrency control protocol, benchmark each in isolation
- **Ablation studies**: disable each novel component one at a time and
  show the performance impact
- **Sensitivity analysis**: vary key parameters (thread count, data size,
  skew factor, read/write ratio) and show how performance changes
- Plan these BEFORE running experiments, not after seeing results

### 1.7 Think about confounders

- Are you comparing on the same hardware, OS, and kernel version?
- Are background processes, NUMA effects, or thermal throttling affecting results?
- Could the improvement come from a different compiler, library version, or
  optimization flag rather than your design?
- Are you comparing at the same scale and under the same load?
- Does your system use more memory or CPU to achieve higher throughput?
  (report resource usage alongside performance)
- For distributed systems: are network conditions identical across experiments?

## Phase 2: Implementation

### General principles
- **Build a working prototype.** Systems papers require running code.
- Start with a minimal end-to-end prototype. Get something that runs first,
  then optimize.
- Use well-tested building blocks: use existing RPC frameworks (gRPC, eRPC),
  concurrency primitives, storage engines where appropriate.
- Write in a language appropriate for systems work (C, C++, Rust, Go).
  Python is acceptable only for control planes, orchestration, or prototyping.
- Version control everything. Tag the exact commit used for each experiment.

### System benchmarking practices
- **Warmup runs**: always run warmup iterations before measurement to
  eliminate cold-start effects (JIT compilation, page faults, cache warming,
  connection establishment)
- **Multiple iterations**: run each experiment at least 5 times minimum.
  Report median and percentiles.
- **Controlled environment**: disable frequency scaling (set CPU governor to
  "performance"), pin processes to cores, disable hyper-threading if it adds
  noise, close unnecessary services
- **Dedicated hardware**: run on dedicated machines, not shared cloud
  instances (or document cloud variability explicitly)
- **Record environment**: document kernel version, compiler version,
  library versions, BIOS settings, NIC firmware, etc.

### For architecture simulation
- Use established simulators: gem5, GPGPU-Sim, Sniper, ZSim, ChampSim,
  Ramulator, McPAT (for power/area)
- Validate your simulator configuration against real hardware when possible
- Use SimPoints or representative sampling for long-running workloads
- Report simulation parameters: core model (in-order/OoO), cache hierarchy,
  memory model, interconnect model
- For FPGA prototypes: report resource utilization (LUTs, FFs, BRAMs),
  clock frequency, and methodology

### For distributed systems
- Test with realistic failure injection (network partitions, node crashes,
  disk failures, slow nodes)
- Use tools like Jepsen, Chaos Monkey, or tc (traffic control) for fault injection
- Verify correctness under failures, not just performance
- Report results at different cluster sizes (scalability)

## Phase 3: Rigorous Evaluation

### Structure your evaluation as:

1. **End-to-end performance** (the headline result)
   - Full system comparison against baselines on representative workloads
   - This answers: "Is the system actually faster/better overall?"

2. **Microbenchmarks** (understanding the design)
   - Isolate each key component and measure its contribution
   - This answers: "Why is the system faster? Which design decisions matter?"

3. **Scalability** (stress testing)
   - Vary the number of cores, nodes, clients, or data size
   - Show how performance scales (linear? sublinear? where does it plateau?)
   - This answers: "Does the system work at scale?"

4. **Sensitivity analysis** (robustness)
   - Vary workload parameters: read/write ratio, key distribution (uniform
     vs. Zipfian), request size, concurrency level
   - This answers: "Does the system only work for one workload, or is it general?"

5. **Breakdown / profiling** (where time goes)
   - Show CPU breakdown, time spent in each component, bottleneck analysis
   - This answers: "Where are the remaining bottlenecks?"

6. **Fault tolerance** (if applicable)
   - Inject failures and show recovery behavior
   - This answers: "Does the system handle real-world failures correctly?"

### Multiple runs (non-negotiable)
- Run every experiment at least 5 times (systems have more variance than ML)
- Report median and percentiles (p5, p95 at minimum)
- For latency: report p50, p95, p99, and p99.9
- Never report best-of-N runs -- always report the median
- If variance is high, investigate and explain why

### Saturation and overload behavior
- Do NOT only show performance at low load. Sweep the load from low to
  saturation and beyond.
- Show the throughput-latency curve: throughput on X axis, latency on Y axis,
  as you increase offered load
- Show what happens at overload -- does the system degrade gracefully or
  fall off a cliff?

### Statistical rigor
- Report confidence intervals for all measurements
- For latency distributions, show CDFs or box plots, not just percentiles
- If claiming a performance difference, ensure it is well outside the
  noise range of your measurement setup
- Document measurement methodology: how you collected timestamps, what
  clock you used, resolution of your measurements

### Common pitfalls in systems evaluation
- DO NOT only measure at low utilization -- systems often behave differently
  under load
- DO NOT use mean latency as your primary metric -- use percentiles
- DO NOT compare against misconfigured or untuned baselines
- DO NOT test only with uniform random workloads -- real workloads have skew
- DO NOT ignore warmup effects -- the first N seconds of measurement are
  often unrepresentative
- DO NOT test at only one scale -- show scalability
- DO NOT run on shared hardware without accounting for interference
- DO NOT report only the workload where your system wins -- show diverse workloads
- DO NOT use wall-clock time without pinning CPU frequency
- DO NOT ignore memory consumption when claiming throughput improvements

## Phase 4: What to Save

Save everything needed to write the paper:

```
results.json          # structured results (see format below)
figures/              # throughput plots, latency CDFs, scalability curves
```

### results.json format

```json
{
  "system": {
    "throughput_ops_sec": {"median": 1250000, "p5": 1210000, "p95": 1280000},
    "latency_us": {"p50": 12.3, "p95": 28.7, "p99": 45.1, "p999": 112.4},
    "cpu_utilization_pct": 78.2,
    "memory_mb": 1024
  },
  "baselines": {
    "baseline_system_1": {
      "throughput_ops_sec": {"median": 420000, "p5": 405000, "p95": 435000},
      "latency_us": {"p50": 35.1, "p95": 89.2, "p99": 210.5, "p999": 1450.0}
    }
  },
  "scalability": {
    "1_thread": {"throughput_ops_sec": 180000},
    "4_threads": {"throughput_ops_sec": 680000},
    "16_threads": {"throughput_ops_sec": 1250000},
    "64_threads": {"throughput_ops_sec": 1180000}
  },
  "ablations": {
    "without_component_A": {
      "throughput_ops_sec": {"median": 890000}
    }
  },
  "config": {
    "experiment_type": "system_benchmark",
    "workload": "YCSB-A (50% read, 50% update, Zipfian)",
    "hardware": "2x Intel Xeon 8380, 256GB DDR4, 2x 1TB NVMe SSD",
    "os": "Ubuntu 22.04, kernel 5.15.0",
    "compiler": "gcc 12.2, -O3",
    "iterations": 5,
    "warmup_seconds": 30,
    "measurement_seconds": 60
  }
}
```

Adapt the structure to your experiment type. The key requirement:
structured, machine-readable, complete, and honest.

## Reproducibility Checklist

Before finishing, verify:
- [ ] Claim is clearly stated and testable
- [ ] Experiment type matches the claim
- [ ] Standard benchmarks used where they exist
- [ ] At least 2 real baseline systems compared fairly (properly configured)
- [ ] Results from 5+ runs with median and percentiles
- [ ] Microbenchmarks isolate each novel component
- [ ] End-to-end evaluation on representative workloads
- [ ] Scalability tested (varying cores, nodes, or data size)
- [ ] Sensitivity analysis on key workload parameters
- [ ] Latency reported as percentiles (p50, p95, p99), not just mean
- [ ] Resource usage reported (CPU, memory, network)
- [ ] Hardware and software environment fully documented
- [ ] Warmup runs performed before measurement
- [ ] Figures saved for key results
- [ ] Negative results or limitations reported honestly
