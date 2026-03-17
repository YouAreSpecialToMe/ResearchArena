# Idea Generation Guidelines (Databases)

How to go from a seed topic to a novel, feasible database research idea.

Distilled from database research methodology, SIGMOD/VLDB/ICDE submission
practices, the "Readings in Database Systems" (Hellerstein & Stonebraker),
and standard systems research principles.

## Step 1: Explore the field

Start by understanding what already exists. DO NOT skip this step.

### Search for existing work (newest to oldest)
- Search DBLP (dblp.org), ACM Digital Library (dl.acm.org), and PVLDB
  proceedings (vldb.org/pvldb) for papers in your seed area
- Also check Google Scholar (scholar.google.com) and Semantic Scholar
  (semanticscholar.org) for broader coverage
- Note: unlike ML, database research is primarily published at conferences
  and journals (SIGMOD, VLDB, ICDE, TODS, VLDB Journal), not arXiv. Preprints
  exist but the authoritative versions are in proceedings.
- **Start with the newest papers first** — sort by date, read the most
  recent work before going to older foundational papers. This ensures
  you know the current frontier before proposing something new.
- Recommended search order:
  1. Last 6 months — what appeared at the most recent SIGMOD, VLDB, ICDE?
  2. Last 1-2 years — what are the current state-of-the-art techniques?
  3. Foundational papers — what are the classic approaches (e.g., seminal
     work on B-trees, cost-based optimization, MVCC, column stores)?
- Look for:
  - Survey papers and tutorial papers — VLDB and SIGMOD tutorials summarize
    the landscape and highlight open problems
  - Highly-cited recent papers — they define current practice
  - Industry papers — SIGMOD Industry Track and CIDR papers reveal what
    practitioners actually need
  - Benchmark papers — they expose where current systems struggle

### Build a mental map
- What are the main approaches in this area?
  - Examples: cost-based vs. adaptive query optimization, row-store vs.
    column-store vs. hybrid layouts, pessimistic vs. optimistic concurrency
    control, LSM-tree vs. B-tree indexing
- What are the established benchmarks? (TPC-H, TPC-DS, TPC-C, YCSB,
  JOB, Star Schema Benchmark)
- What are the standard metrics? (query latency, throughput, storage
  overhead, index build time, memory consumption)
- What are the known limitations of current approaches?
- What recent hardware trends create new opportunities? (NVMe SSDs,
  persistent memory, RDMA networking, hardware accelerators, large DRAM)
- What recent workload shifts create new challenges? (cloud-native
  databases, serverless, multi-tenant, ML-in-DB, graph+relational)

### Find the gaps
- Read the "Limitations" and "Future Work" sections of recent papers
- Read CIDR and SIGMOD Record vision papers — they explicitly discuss
  open research directions
- Identify assumptions that current methods make — can you relax them?
  (e.g., static data distribution, uniform access patterns, single-node)
- Look for workload patterns where existing systems perform poorly:
  - Skewed data distributions
  - Mixed read-write workloads (HTAP)
  - Multi-way joins on real-world graphs
  - Evolving schemas and workload drift
  - Edge cases in concurrency control under contention

## Step 2: Generate candidate ideas

### Two approaches (choose one or combine)

**Goal-driven** (recommended): Start with a real database workload problem.
- "Current query optimizers produce poor plans for queries with many joins
  because cardinality estimation fails. How can we fix that?"
- "LSM-tree compaction causes write stalls under heavy ingestion. Can we
  eliminate or hide the stalls?"
- "Existing learned indexes assume static data. How do they handle inserts?"
- The real-world workload motivation constrains your search and makes the
  contribution clear. Database reviewers care deeply about practical relevance.

**Idea-driven**: Start with a technique and find where it applies.
- "Technique A (e.g., learned models) works well for B (index lookup). Could
  it also work for C (query optimization, cardinality estimation)?"
- Riskier — you may find the idea already exists or doesn't outperform
  carefully tuned traditional approaches.

### Types of contributions in database research
Database papers typically make one or more of these contributions:
- **New algorithm**: query optimization strategy, join algorithm, sorting
  algorithm, concurrency control protocol, recovery mechanism
- **New data structure**: index structure, storage layout, buffer management
  scheme, cache-friendly data organization
- **New system design**: novel architecture for a database component or
  an entire system (e.g., new query execution engine, new storage engine)
- **New query processing technique**: approximate query processing, adaptive
  query execution, query compilation
- **New transaction protocol**: isolation level implementation, distributed
  commit protocol, deterministic transaction processing
- **Formal analysis**: complexity bounds, correctness proofs for protocols,
  impossibility results
- **Empirical study**: large-scale evaluation comparing existing approaches
  under controlled conditions, revealing previously unknown tradeoffs

### What makes a good database research idea
- **Novel**: Not already done. You MUST verify this (Step 3).
- **Feasible**: Can be implemented and evaluated within your resource constraints.
- **Practically motivated**: There's a real workload pattern or system
  limitation that motivates the work.
- **Clear**: The contribution is easy to explain in one sentence.
- **Testable**: There's a concrete way to evaluate whether it works, using
  standard benchmarks and metrics.
- **Has both depth and breadth**: Ideally combines algorithmic contribution
  (new technique) with practical evaluation (benchmark results). A strong
  theoretical contribution (complexity analysis, correctness proof) adds
  further value.

### What makes a BAD database research idea
- Too broad ("improve databases") — needs a specific problem and approach
- Too incremental ("change buffer pool size from X to Y")
- No workload motivation ("we tried technique X on problem Y because no
  one has" — but WHY would anyone want to?)
- Pure engineering without intellectual contribution (reimplementing a known
  technique in a new language)
- Missing the systems context (proposing an algorithm without considering
  how it interacts with the rest of the database stack)
- Already exists (you didn't check DBLP and the proceedings)

## Step 3: Verify novelty (CRITICAL — do not skip)

Before committing to an idea, verify it hasn't been done:

### Search specifically for your idea
- Search DBLP, ACM DL, and Semantic Scholar with keywords from your
  proposed technique
- Search for the PROBLEM you're solving, not just your approach
- Check if your idea is a special case of something more general that exists
- Look at the "Related Work" sections of papers closest to your idea
- Check PhD dissertations — they often contain ideas that were not separately
  published as papers

### Common novelty traps
- Your idea exists but under a different name (terminology varies between
  the systems, theory, and ML-for-DB subcommunities)
- Your idea was tried in the 1980s/90s and abandoned for good reasons
  (check the Stonebraker and Hellerstein readings)
- Your idea is a minor variation of an existing approach
- A concurrent paper at the most recent SIGMOD/VLDB does the same thing
- An industry system already implements your idea (check system papers
  from Google, Amazon, Microsoft, Oracle, Snowflake, etc.)

### If your idea already exists
- DON'T give up immediately. Ask: can you improve on it? Apply it to a
  new workload? Combine it with a complementary technique? Adapt it to
  modern hardware? Prove tighter bounds?
- If it truly exists with no room for improvement, go back to Step 2

## Step 4: Refine and document

### Write your idea.json with:
- **description**: 1-3 sentences explaining what you're proposing
- **motivation**: what workload problem or system limitation you're addressing,
  and why existing approaches fall short
- **proposed_approach**: your high-level technique and why it should work;
  include algorithmic intuition (not just "we use ML")
- **related_work**: key existing papers and how your idea differs
  (use REAL papers you found in Steps 1 and 3 — include titles and authors)

### Sanity checks before moving on
- Can you explain the idea in one sentence to a non-expert?
- Is there a clear experiment using standard benchmarks that would test it?
- Do you have the resources (hardware, datasets, time) to do it?
- Does the idea have both an algorithmic contribution AND a plan for
  practical evaluation?
- Can you sketch the complexity analysis (time/space) for your approach?
- Is the expected contribution significant enough for a SIGMOD/VLDB paper?

## General principles

### From database research tradition
- Systems context matters — your algorithm must work within a real system,
  not just in isolation
- Workload-driven research produces the most impactful papers — start from
  what real users need
- Hardware-aware design is increasingly important — know your memory
  hierarchy, storage characteristics, and network constraints
- Correctness is non-negotiable for transaction processing and recovery —
  formal arguments or proofs are expected
- Reproducibility requires specifying hardware, OS, compiler flags, and
  system configuration — database performance is sensitive to all of these

### From "Readings in Database Systems" (Hellerstein & Stonebraker)
- Many "new" ideas in databases are rediscoveries — learn the history
- The tension between generality and performance is the central tradeoff
- Elegant solutions that work in practice beat complex solutions that
  are theoretically optimal but impractical
- The best database papers change how systems are built, not just how
  papers are written

### On learned components and ML-for-DB
- Learned indexes, learned query optimization, and learned tuning are active
  areas, but reviewers expect comparison against well-tuned traditional
  baselines (not strawman implementations)
- Training overhead, update cost, and robustness to workload shift must
  be evaluated — not just steady-state performance
- A learned approach that fails gracefully (falls back to traditional
  methods) is more convincing than one that doesn't
