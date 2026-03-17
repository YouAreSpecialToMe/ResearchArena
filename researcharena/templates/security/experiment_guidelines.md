# Experiment Guidelines (Computer Security)

Distilled from reviewing practices at CCS, IEEE S&P, USENIX Security, and NDSS,
combined with best practices from systems security, network security, and
applied cryptography research.

## Phase 1: Experiment Design (do this BEFORE writing code)

### 1.1 Formulate your claim

Before any implementation, write down:
- **What is your hypothesis?** State it as a testable claim.
  Examples:
  - Attack: "Our attack can extract secret keys from implementation X with
    Y% success rate using Z resources."
  - Defense: "Our defense detects attack class X with Y% detection rate
    and less than Z% false positive rate, with at most W% performance overhead."
  - Analysis: "Protocol X is vulnerable to attack Y under threat model Z."
  - Measurement: "X% of deployed systems are vulnerable to attack Y."
- **What evidence would convince a skeptical reviewer?**
- **What would DISPROVE your claim?** Design experiments that could fail.

### 1.2 Choose the right experiment type

Security research uses different evaluation approaches than ML. Choose what
fits your claim:

| Claim type | Experiment type | What to measure |
|---|---|---|
| "Our attack works" | Attack demonstration | Success rate, time to exploit, required access/resources |
| "Our defense detects X" | Detection evaluation | True positive rate, false positive rate, detection latency |
| "Our defense prevents X" | Prevention evaluation | Attack success rate with/without defense, evasion resistance |
| "Our defense is practical" | Performance evaluation | Latency overhead, throughput impact, memory cost |
| "Protocol X is secure" | Formal verification | Proof in ProVerif/Tamarin/EasyCrypt, or counterexample |
| "Protocol X is broken" | Protocol attack | Concrete attack trace, exploit demonstration |
| "X is widespread" | Measurement study | Prevalence statistics, temporal trends, demographic breakdown |
| "Our crypto scheme is secure" | Security proof + benchmarks | Reduction-based proof, parameter analysis, implementation benchmarks |
| "Component A is critical" | Ablation/component analysis | Performance with/without A |

### 1.3 Select what to measure

Metrics depend on your contribution type:

**For attacks:**
- Success rate (% of attempts that succeed)
- Time to exploit (seconds/minutes/hours)
- Required resources (compute, network access, physical proximity)
- Required attacker capabilities (what privileges are needed?)
- Stealthiness (can the attack be detected by existing defenses?)
- Impact (what exactly is compromised? confidentiality, integrity, availability?)

**For defenses:**
- Detection rate / true positive rate (% of attacks correctly identified)
- False positive rate (% of benign activity incorrectly flagged) -- this is
  often the most important metric; high false positive rates make defenses unusable
- False negative rate (% of attacks missed)
- Performance overhead:
  - Latency: added delay per operation (ms, us)
  - Throughput: reduction in requests/second or transactions/second
  - Memory: additional RAM/storage required
  - CPU: additional processing load
- Evasion resistance: does the defense hold against adaptive adversaries?
- Deployment cost: what changes are required to adopt this defense?

**For protocol/formal analysis:**
- Security properties verified (authentication, confidentiality, integrity,
  forward secrecy, etc.)
- Assumptions required for the proof
- Verification time and model size (for automated provers)
- Any attacks found, with concrete traces

**For measurement studies:**
- Sample size and representativeness
- Confidence intervals for prevalence estimates
- Temporal trends (is the problem growing or shrinking?)
- Breakdown by category (OS, version, geography, sector)
- Limitations of the measurement methodology

**For cryptographic constructions:**
- Security parameter analysis (what parameters achieve 128-bit security?)
- Key sizes, ciphertext sizes, signature sizes
- Computation time (encryption, decryption, signing, verification)
- Comparison with existing schemes at equivalent security levels

### 1.4 Choose evaluation targets and datasets

- **For attacks**: test against real, widely-deployed systems or faithful
  reproductions of them. Attacks on toy systems are not convincing.
- **For defenses**: evaluate on realistic workloads and real attack datasets.
  Use established benchmarks when they exist:
  - Network intrusion: CICIDS, CTU-13, or capture your own traffic
  - Malware: VirusTotal, MalwareBazaar, or curated collections
  - Web security: OWASP benchmarks, real-world vulnerability databases
  - Binary analysis: LAVA-M, Magma, or real CVEs
  - Fuzzing: Google FuzzBench, OSS-Fuzz targets
- **For measurement studies**: use representative datasets. Clearly document
  any sampling bias and its implications.
- Document: data source, collection methodology, size, time period,
  any filtering or preprocessing applied, ethical considerations for
  data collection

### 1.5 Select baselines fairly

- Include at least 2 meaningful baselines:
  - One simple baseline (signature matching, rule-based, standard config)
  - One strong baseline (recent published defense/attack from a top venue)
- Run all baselines with equivalent effort on the same evaluation setup
- If a baseline tool is publicly available, run it yourself rather than
  citing numbers from a different evaluation setup
- If a baseline is too complex to reproduce, cite published numbers and
  clearly state the comparison is indirect
- Never compare against intentionally weak baselines

### 1.6 Plan for adaptive adversaries

This is unique to security and CRITICAL:
- If you propose a defense, you MUST evaluate against adversaries who
  know about your defense and actively try to evade it
- Design at least one adaptive attack: an attacker who knows your
  detection algorithm and modifies their behavior accordingly
- If your defense relies on a secret (key, model, threshold), evaluate
  what happens if the secret is partially or fully compromised
- A defense that only works against non-adaptive adversaries will be
  rejected at top venues

### 1.7 Think about confounders

- Could your attack succeed due to a misconfiguration rather than a
  fundamental vulnerability?
- Could your defense's performance come from the dataset being easy
  rather than your technique being good?
- Are you comparing defenses at the same false positive rate?
- Is the performance overhead measured under realistic load conditions?
- Could environmental factors (network conditions, hardware differences)
  explain your results?

## Phase 2: Implementation

### General principles
- Start simple. Get a minimal proof of concept working first.
- Add complexity one piece at a time. Test each change independently.
- Use established security tools and frameworks:
  - Fuzzing: AFL++, libFuzzer, honggfuzz
  - Binary analysis: angr, Ghidra, Binary Ninja
  - Network: Scapy, Wireshark, tcpdump
  - Crypto: OpenSSL, libsodium, PyCryptodome
  - Formal verification: ProVerif, Tamarin, CBMC, KLEE
  - Web security: Burp Suite, OWASP ZAP
  - Containers/VMs: Docker, QEMU for isolated testing
- Fix random seeds where applicable (fuzzing, ML-based detection, etc.)

### If implementing attacks
- Work in an isolated environment (VMs, containers, air-gapped networks)
- Document every step of the attack for reproducibility
- Capture evidence: packet captures, logs, screenshots, traces
- Measure timing precisely (use hardware counters for side channels)
- Test against multiple versions/configurations of the target

### If implementing defenses
- Instrument the system to measure performance overhead accurately
- Test with realistic workloads, not just microbenchmarks
- Include both benign and malicious traffic in your evaluation
- Measure overhead under varying load conditions (idle, normal, peak)
- Implement logging for detection events (true/false positives/negatives)

### If doing formal verification
- Start with a simplified model, then add complexity
- Document all abstractions and simplifications
- Verify known-secure protocols first to validate your model
- Report verification time, number of states explored, any manual steps

### If doing measurement studies
- Automate data collection for reproducibility
- Implement rate limiting and follow ethical scanning guidelines
- Log all raw data before processing
- Use established scanning tools (ZMap, Censys, Shodan APIs) when possible
- Follow responsible scanning practices (identify yourself, provide opt-out)

## Phase 3: Rigorous Evaluation

### Multiple runs (required where applicable)
- Run every non-deterministic experiment at least 3 times
- For attacks: report success rate over multiple attempts
- For defenses with ML components: use multiple random seeds
- For performance measurements: report median and percentiles (p50, p95, p99)
- For fuzzing: run for the same duration with different seeds

### Ablation studies (required for multi-component systems)
- Remove or disable each novel component one at a time
- Show quantitative impact: "without component X, detection rate drops
  from Y% to Z%" or "without optimization X, overhead increases from Y% to Z%"
- This proves every part of your system contributes

### Evasion analysis (required for defenses)
- Test against at least one adaptive adversary (see Phase 1, Section 1.6)
- Document which evasion strategies you considered and tested
- If your defense can be evaded, acknowledge it honestly and discuss
  the conditions under which evasion is possible
- Consider polymorphic attacks, mimicry attacks, and adversarial examples
  as applicable

### Performance overhead analysis (required for defenses)
- Measure overhead on realistic benchmarks:
  - Web servers: requests/second, response latency
  - Databases: transactions/second, query latency
  - Applications: end-to-end task completion time
  - System-level: CPU utilization, memory consumption, I/O overhead
- Compare: unprotected system vs. your defense vs. existing defenses
- Report both average case and worst case
- Test at multiple scales (1x, 10x, 100x typical load)

### Scalability analysis (recommended)
- How does your system scale with input size, network size, or attack complexity?
- Report resource consumption at multiple scales
- Identify bottlenecks and discuss practical deployment limits

### Statistical rigor
- Report confidence intervals for rates and proportions
- For measurement studies, account for sampling bias
- For comparisons, use appropriate statistical tests
- Distinguish statistical significance from practical significance

## Phase 4: Common Pitfalls

Security-specific pitfalls:

- DO NOT evaluate defenses only against non-adaptive adversaries
- DO NOT ignore false positive rates -- a defense with 99% detection but
  10% false positives is often useless in practice
- DO NOT test attacks only on default/misconfigured targets
- DO NOT measure performance overhead on toy workloads
- DO NOT claim formal security guarantees without a formal proof
- DO NOT use outdated or unrepresentative datasets
- DO NOT scan or attack systems without authorization
- DO NOT ignore ethical considerations (responsible disclosure, IRB approval)
- DO NOT compare defenses at different false positive rate operating points
- DO NOT cherry-pick attack scenarios where your defense works best
- DO NOT assume the attacker is static -- real attackers adapt
- DO NOT report only aggregate metrics -- break down by attack type/category

## Phase 5: What to Save

Save everything needed to write the paper:

```
results.json          # structured results (see format below)
figures/              # performance charts, ROC curves, overhead graphs
```

### results.json format

```json
{
  "method": {
    "detection_rate": {"mean": 0.9734, "std": 0.0021},
    "false_positive_rate": {"mean": 0.0023, "std": 0.0005},
    "latency_overhead_ms": {"mean": 2.3, "std": 0.4},
    "throughput_overhead_pct": {"mean": 4.7, "std": 0.8}
  },
  "baselines": {
    "baseline_defense_1": {
      "detection_rate": {"mean": 0.9102, "std": 0.0018},
      "false_positive_rate": {"mean": 0.0087, "std": 0.0012}
    }
  },
  "ablations": {
    "without_component_A": {
      "detection_rate": {"mean": 0.9201, "std": 0.0025}
    }
  },
  "evasion_analysis": {
    "adaptive_attack_1": {
      "detection_rate_under_evasion": {"mean": 0.8901, "std": 0.0034}
    }
  },
  "config": {
    "experiment_type": "defense_evaluation",
    "target_system": "system_name_and_version",
    "dataset": "dataset_name",
    "seeds": [42, 123, 456],
    "hardware": "description",
    "total_runtime_minutes": 240
  }
}
```

Adapt the structure to your experiment type. The key requirement:
structured, machine-readable, complete, and honest.

## Reproducibility Checklist

Before finishing, verify:
- [ ] Claim is clearly stated and testable
- [ ] Threat model is precisely defined
- [ ] Experiment type matches the claim
- [ ] Evaluation uses realistic systems/workloads, not toy examples
- [ ] At least 2 meaningful baselines compared fairly
- [ ] Results from multiple runs with mean +/- std (where applicable)
- [ ] Ablation study for each novel component
- [ ] Evasion analysis against adaptive adversaries (for defenses)
- [ ] Performance overhead measured on realistic benchmarks (for defenses)
- [ ] False positive rate reported (for detection systems)
- [ ] All configuration documented in results.json
- [ ] Figures saved for key results (ROC curves, overhead plots, etc.)
- [ ] Ethical considerations documented (responsible disclosure, IRB)
- [ ] Negative results reported honestly (if any)
