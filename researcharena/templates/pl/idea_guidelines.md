# Idea Generation Guidelines (Programming Languages)

How to go from a seed topic to a novel, feasible PL research idea targeting
PLDI, POPL, OOPSLA, or ICFP.

Distilled from best practices in SIGPLAN communities, the SIGPLAN Empirical
Evaluation Checklist, and established norms in type theory, static analysis,
compiler optimization, and program synthesis research.

## Step 1: Explore the field

Start by understanding what already exists. DO NOT skip this step.

### Search for existing work (newest to oldest)
- Search ACM Digital Library (dl.acm.org), DBLP (dblp.org), Semantic Scholar
  (semanticscholar.org), and Google Scholar (scholar.google.com) for papers in
  your seed area
- Search SIGPLAN proceedings specifically: PLDI, POPL, OOPSLA, ICFP, CC, CGO,
  ASPLOS (for systems-level PL work), and ECOOP
- **Start with the newest papers first** — sort by date, read the most recent
  work before going to older foundational papers. This ensures you know the
  current frontier before proposing something new.
- Recommended search order:
  1. Last 6 months — what's happening right now?
  2. Last 1-2 years — what are the current state-of-the-art techniques?
  3. Foundational papers — what are the classic results?
- Look for:
  - Survey papers and SoK (Systematization of Knowledge) papers
  - Distinguished papers from recent PLDI/POPL/OOPSLA/ICFP proceedings
  - Workshop papers from PLDI/POPL co-located workshops (e.g., SOAP, MAPL,
    TyDe, ML Workshop) — they contain early-stage ideas

### Build a mental map
- What are the main approaches in this area?
- What formal frameworks are used (type systems, abstract interpretation,
  separation logic, categorical semantics, etc.)?
- What are the established benchmarks and evaluation methodology?
- What are the known limitations of current techniques?
- What problems are considered "open" or "unsolved"?
- What recent techniques from adjacent areas (verification, databases,
  security, ML) could apply here?

### Find the gaps
- Read the "Limitations" and "Future Work" sections of recent papers
- Look for recurring weaknesses: scalability issues, restrictive assumptions,
  incomplete soundness arguments, narrow language support
- Identify assumptions that current methods make — can you relax them?
- Look for problems where current tools fail on real-world code (this signals
  the community hasn't cracked it yet)
- Check tool comparison papers and competition results (e.g., SV-COMP) for
  areas where no tool performs well

## Step 2: Generate candidate ideas

### Types of PL contributions

PL papers typically contribute one or more of:
- **New type system**: A type discipline that captures a property not previously
  enforced statically (ownership, effects, linearity, gradual types, etc.)
- **Static analysis**: A new analysis technique (abstract interpretation,
  dataflow analysis, pointer analysis) or significant precision/scalability
  improvement to an existing one
- **Compiler optimization**: A new optimization pass, intermediate
  representation, or compilation strategy with measurable performance gains
- **Program synthesis**: A technique for automatically generating programs from
  specifications, examples, or sketches
- **Verification method**: A new approach to proving program properties
  (safety, liveness, functional correctness, security)
- **Domain-specific language (DSL)**: A new language or language extension
  tailored to a specific domain with clear advantages over general-purpose
  alternatives
- **Runtime system**: Garbage collection, JIT compilation, concurrency
  primitives, memory management innovations
- **Semantics and metatheory**: New semantic frameworks, proof techniques, or
  foundational results about programs or languages
- **Empirical study**: Large-scale study of programming practices, bug
  patterns, or language feature usage that yields actionable insights

### Two approaches (choose one or combine)

**Goal-driven** (recommended): Start with a concrete problem.
- "Current analyses for X are unsound when Y happens. How can we fix that?"
- "Programs with property Z cannot be verified by existing tools. What
  formalism would enable this?"
- "Compilation of language feature W is too slow. Can we do better?"
- The goal constrains your search and makes the contribution clear.

**Idea-driven**: Start with a technique and find where it applies.
- "Abstract interpretation domain D works well for numerical properties. Could
  it also work for string properties?"
- "This type system idea from functional languages could apply to systems code."
- Riskier — you may find the idea already exists or doesn't work.

### Matching your idea to a venue

- **POPL**: Formal contributions with strong metatheory. Soundness proofs
  (ideally mechanized) are expected. Implementation and evaluation are valued
  but secondary to the formal contribution.
- **PLDI**: Implementation-oriented with strong evaluation. Formal foundations
  are welcome but the paper must demonstrate practical impact through
  benchmarks on real-world code.
- **OOPSLA**: Broad scope, accepts both formal and systems-oriented work.
  Expects solid evaluation. Particularly receptive to work on object-oriented,
  dynamic, and mainstream languages.
- **ICFP**: Functional programming focus. Values elegance of design, type
  theory contributions, and connections to mathematics. Mechanized proofs
  in Coq/Agda/Lean are highly valued.

### What makes a good PL research idea
- **Novel**: Not already done. You MUST verify this (Step 3).
- **Technically deep**: Has a non-trivial formal or engineering insight.
- **Sound**: Formal claims must be provable; empirical claims must be testable.
- **Practical**: Addresses a real problem in real programs or real languages
  (except for purely foundational work at POPL/ICFP).
- **Clear**: The contribution is easy to explain in one sentence.
- **Significant**: If it works, the community would care.

### What makes a BAD PL research idea
- Too broad ("improve static analysis") — needs a specific property, language,
  and analysis technique
- Too incremental ("add one more rule to an existing type system" without new
  insight)
- Unsound without acknowledging it (reviewers will find the counterexample)
- Requires resources you don't have (access to a proprietary compiler, massive
  industrial codebase)
- Already exists (you didn't check ACM DL and DBLP)
- All formalism, no motivation (what real problem does this solve?)

## Step 3: Verify novelty (CRITICAL -- do not skip)

Before committing to an idea, verify it hasn't been done:

### Search specifically for your idea
- Search ACM DL, DBLP, Semantic Scholar, and arXiv with keywords from your
  proposed technique
- Search for the PROBLEM you're solving, not just your approach
- Check if your idea is a special case of something more general that exists
- Look at the "Related Work" sections of papers closest to your idea
- Check PhD dissertations in the area — they often contain unpublished ideas

### Common novelty traps in PL
- Your type system is an instance of a known framework (e.g., refinement
  types, effect systems, coeffect systems)
- Your analysis is a known abstract domain applied to a known framework
- Your optimization exists in a different compiler (GCC vs LLVM vs GraalVM)
- A concurrent paper at a recent PLDI/POPL does the same thing
- Your idea was explored in the 1990s under different terminology (PL has a
  long history — search older proceedings too)
- Your "new" language feature exists in a less well-known language (Racket,
  Haskell, Scala, Rust, etc.)

### If your idea already exists
- DON'T give up immediately. Ask: can you make it sound where it was unsound?
  Scale to larger programs? Handle a richer language? Provide mechanized proofs
  where only pen-and-paper proofs existed? Combine it with something new?
- If it truly exists with no room for improvement, go back to Step 2.

## Step 4: Refine and document

### Consider the formal-practical balance

Every strong PL paper balances formalism with practicality:
- **Formal component**: Type rules, operational semantics, abstract domains,
  proof obligations, soundness theorems. What is the formal contribution?
- **Practical component**: Implementation, evaluation on real code, comparison
  with existing tools, usability. Does it work in practice?

POPL leans formal. PLDI/OOPSLA lean practical. ICFP values elegant design.
All venues expect both components to some degree.

### Write your idea.json with:
- **description**: 1-3 sentences explaining what you're proposing
- **motivation**: Why this problem matters, what gap you're filling. Include
  a concrete motivating example (a real or realistic program that illustrates
  the problem)
- **proposed_approach**: Your high-level method and why it should work.
  Sketch the key formal idea (e.g., "a type system with linear capabilities
  that tracks resource usage") and the evaluation plan.
- **related_work**: Key existing papers and how your idea differs (use REAL
  papers you found in Steps 1 and 3 — include titles and authors)

### Sanity checks before moving on
- Can you explain the key insight in one sentence to a PL researcher?
- Can you give a small example program that illustrates why existing approaches
  fail and yours succeeds?
- Is there a clear formal claim you can state and prove?
- Is there a clear empirical evaluation that would demonstrate practical value?
- Do you have the resources (benchmarks, tools, time) to do it?
- Is the expected contribution large enough for the target venue?

## General principles

### On choosing problems
- Your ability to choose the right problem is more important than raw skill
- Good PL problems often come from real pain points: programs that are hard to
  write correctly, bugs that are hard to find, code that is hard to optimize
- Watch which ideas prosper and which are forgotten — this develops taste
- Goal-driven research has lower scooping risk than idea-driven research
- There's no shame in working on ideas suggested by others or by the literature

### On formalism
- Formalism is a tool, not the goal. Use it to make your contribution precise,
  not to impress reviewers.
- A simple, clean formalism that captures the key insight is better than a
  complex one that handles every corner case
- Start with a core calculus, get the key ideas right, then extend
- If you can't prove soundness for the full system, prove it for a core
  subset and clearly state what's not covered

### On practicality
- The best PL papers change how people write, analyze, or compile programs
- Even theoretical papers benefit from a prototype implementation
- Show your technique works on code that real developers write, not just
  toy examples
- Consider: would a developer or tool builder actually use this?
