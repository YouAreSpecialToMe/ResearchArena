# Idea Generation Guidelines (Systems)

How to go from a seed topic to a novel, feasible systems research idea.

Distilled from "How to Write a Great Research Paper" (Peyton Jones), "Writing Good
Systems Research Papers" guides, OSDI/SOSP/EuroSys CFPs, and the systems research
tradition of building real artifacts.

## Step 1: Explore the field

Start by understanding what already exists. DO NOT skip this step.

### Search for existing work (newest to oldest)
- Search USENIX proceedings (usenix.org), ACM Digital Library (dl.acm.org),
  IEEE Xplore (ieeexplore.ieee.org), and Google Scholar (scholar.google.com)
  for papers in your seed area
- Systems papers are published at conferences, NOT typically on arXiv.
  Search conference proceedings directly:
  - Operating systems / storage: OSDI, SOSP, EuroSys, FAST, ATC
  - Networking: NSDI, SIGCOMM, CoNEXT
  - Architecture: ISCA, MICRO, ASPLOS, HPCA
  - Databases: VLDB, SIGMOD, OSDI (increasingly)
  - Security: USENIX Security, Oakland, CCS
- **Start with the newest papers first** -- sort by date, read the most
  recent work before going to older foundational papers
- Recommended search order:
  1. Last 6 months -- what systems are people building right now?
  2. Last 1-2 years -- what are the current state-of-the-art systems?
  3. Foundational papers -- what are the classic designs that define the area?
- Look for:
  - Best Paper Award winners -- they define what the community values
  - Industry papers (Google, Meta, Microsoft, Amazon) -- they reveal real-world pain points
  - Measurement and characterization papers -- they expose new problems worth solving

### Build a mental map
- What are the dominant system designs in this area?
- What hardware trends are changing the landscape (new storage media, accelerators, network speeds)?
- What workload shifts are happening (AI training, microservices, edge computing)?
- What are the known bottlenecks -- where do current systems break down?
- What tradeoffs do existing systems make, and are those tradeoffs still justified?

### Find the gaps
- Read the "Limitations" and "Future Work" sections of recent papers
- Look for industry blog posts describing operational pain points that
  academic systems have not yet addressed
- Identify assumptions that current systems make -- have those assumptions
  been invalidated by new hardware, new workloads, or new scale?
- Look for areas where practitioners resort to ugly hacks -- that signals
  the current abstractions are wrong
- Check if a design from one domain (e.g., databases) could solve a problem
  in another (e.g., OS scheduling)

## Step 2: Generate candidate ideas

### The central question in systems research

Systems research is about building things that work. Every idea must answer:
**What real problem does this solve, and why can't existing systems solve it?**

If you cannot name a concrete workload or deployment scenario where current
systems fail, you do not yet have a systems research idea.

### Two approaches (choose one or combine)

**Problem-driven** (strongly recommended for systems):
- Start with a real-world pain point or performance bottleneck
- "System X breaks down under workload Y because of design decision Z. We
  can fix this by rethinking Z."
- "New hardware H makes old assumption A obsolete. A system redesigned
  around H's properties can achieve N times better performance."
- The best systems papers identify a problem that practitioners already
  feel but nobody has properly solved.

**Technique-driven** (higher risk):
- "Technique T from area A could be applied to build a better system for B."
- Must still have a clear problem it solves -- a technique looking for a
  problem is not enough.

### What makes a good systems research idea
- **Novel design**: Proposes a new system design, architecture, abstraction,
  or interface -- not just a parameter tweak to an existing system
- **Real problem**: Addresses a genuine bottleneck, failure mode, or
  limitation in real workloads
- **Buildable**: Can be implemented as a working prototype (systems papers
  require running code, not just theory)
- **Evaluable**: There are meaningful benchmarks or workloads to test against
- **Insightful**: The design embodies a non-obvious insight about WHY the
  problem exists and why existing approaches miss it

### Types of systems contributions
- **New system**: A complete system design that solves a problem existing
  systems cannot (e.g., Raft for consensus, Spark for iterative analytics)
- **New abstraction or interface**: A better way to expose functionality
  (e.g., MapReduce, exokernels, RDMA verbs)
- **Performance improvement**: Significant speedup through architectural
  redesign, not just implementation tuning (e.g., kernel bypass, zero-copy)
- **Reliability/availability**: New mechanisms for fault tolerance,
  consistency, or recovery (e.g., Paxos, PBFT, CRDTs)
- **Scalability**: Designs that work at scales where existing systems fail
- **Resource efficiency**: Doing more with less (memory, energy, network bandwidth)
- **Measurement and characterization**: Rigorous study of real workloads
  that reveals surprising findings and motivates new designs

### What makes a BAD systems idea
- No working system ("we propose but did not implement")
- Solves a problem nobody actually has (cool technique, no real use case)
- Marginal improvement that could be achieved by tuning the existing system
- Ignores practical constraints (assumes hardware that does not exist,
  workloads that are unrealistic)
- Reinvents an existing system without understanding why the original was
  designed that way

## Step 3: Verify novelty (CRITICAL -- do not skip)

Before committing to an idea, verify it has not been done:

### Search specifically for your idea
- Search ACM DL, USENIX proceedings, and IEEE Xplore with keywords from
  your proposed design
- Search for the PROBLEM you are solving, not just your approach
- Check if your design is a special case of a more general existing system
- Look at the "Related Work" sections of papers closest to your idea
- Check industry systems -- companies sometimes build what academia has not
  published (check engineering blog posts from major tech companies)

### Common novelty traps in systems
- Your system exists but is called something different (naming is inconsistent
  across OS, networking, and database communities)
- Your idea was tried in a different era with different hardware -- check if
  old ideas have been revisited with modern hardware
- An industry system already does this but has not been published academically
- Your improvement comes from better engineering, not a new design insight
  (important: systems papers need a design contribution, not just faster code)

### If your idea already exists
- Can you handle a workload or scale that the existing system cannot?
- Can you provide a stronger guarantee (consistency, durability, fault tolerance)?
- Can you achieve similar performance with a fundamentally simpler design?
- Can you adapt the idea to new hardware (persistent memory, CXL, SmartNICs)?
- If none of these, go back to Step 2

## Step 4: Refine and document

### Write your idea.json with:
- **description**: 1-3 sentences explaining what system you are building
  and what problem it solves
- **motivation**: the real-world pain point or bottleneck, with evidence
  (measurements, industry reports, published characterization studies)
- **proposed_approach**: your system design at a high level -- what are the
  key components and why each design decision is made
- **related_work**: key existing systems and papers, and how your design
  differs (use REAL papers and systems you found in Steps 1 and 3 --
  include titles, authors, and venues)

### Sanity checks before moving on
- Can you explain the system's key insight in one sentence?
- Can you build a working prototype?
- Is there a clear workload or benchmark to evaluate against?
- Can you name at least two real baseline systems to compare with?
- Is the expected improvement large enough to matter in practice
  (systems reviewers expect meaningful gains, not 5% improvements)?

## General principles

### From "Writing Good Systems Papers"
- The key contribution is the DESIGN, not the implementation. Your paper
  must explain WHY you made each design choice, not just WHAT you built.
- A system that works is necessary but not sufficient. You need a
  principled design with clear reasoning.
- Performance numbers alone do not make a contribution. Reviewers want to
  learn something generalizable from your system.

### From OSDI/SOSP tradition
- Build real systems that solve real problems for real users
- The best systems papers change how people think about a problem, not just
  how fast they can solve it
- Negative results are valuable when they expose fundamental limitations
  of existing approaches
- Simplicity is a virtue. A simpler design that achieves 90% of the
  performance of a complex one is often the better contribution.

### Problem selection
- Your ability to pick the right problem is more important than your
  implementation skill
- Talk to practitioners (or read their blog posts / conference talks) to
  find problems that matter in practice
- A well-motivated problem with a clean solution beats a poorly motivated
  problem with a sophisticated solution every time
- Watch what problems the best groups are working on, but also look for
  problems they are ignoring
