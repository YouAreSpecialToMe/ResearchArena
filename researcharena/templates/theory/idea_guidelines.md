# Idea Generation Guidelines (Theory)

How to go from a seed topic to a novel theoretical contribution in
algorithms, complexity, or combinatorics.

Distilled from practices in the STOC/FOCS/SODA/COLT communities, Oded
Goldreich's advice on theoretical research, and standard conventions in
theoretical computer science.

## Step 1: Explore the field

Start by understanding what is already known. Theory has a long history --
results may be decades old. DO NOT skip this step.

### Search for existing work

- Search ECCC (eccc.weizmann.ac.il) for complexity theory results
- Search arXiv sections: cs.DS (data structures and algorithms), cs.CC
  (computational complexity), cs.CG (computational geometry), cs.LG
  (learning theory, for COLT-style work)
- Search DBLP (dblp.org) for author and title lookups
- Search conference proceedings directly: STOC, FOCS, SODA, COLT, ICALP,
  ESA, APPROX/RANDOM, SoCG, CCC
- Search Semantic Scholar (semanticscholar.org) and Google Scholar for
  citation graphs and related work
- **Start with the most recent results** -- what are the current best bounds?
  Then trace back to the foundational papers that introduced the problem.
- Recommended search order:
  1. Last 1-2 years -- what are the newest upper and lower bounds?
  2. Last 5-10 years -- what techniques drove recent progress?
  3. Foundational papers -- who defined the problem and what were the
     original bounds?

### Build a mental map

- What is the best known upper bound for the problem?
- What is the best known lower bound?
- What is the conjectured optimal bound (if any)?
- What computational model are results stated in? (RAM, comparison model,
  cell probe, streaming, query complexity, circuit complexity, etc.)
- What are the main algorithmic paradigms used? (divide and conquer,
  dynamic programming, LP/SDP relaxations, spectral methods, algebraic
  techniques, primal-dual, multiplicative weights, etc.)
- What are the main barrier results? (oracle separations, natural proofs
  barrier, algebrization, communication complexity lower bounds, etc.)
- Are there conditional hardness results? (assuming P != NP, ETH, SETH,
  3SUM conjecture, OMv conjecture, etc.)

### Find the gaps

- Read survey papers and open problems lists (e.g., the Open Problems
  column in SIGACT News, workshop open problem sessions)
- Identify problems where there is a gap between upper and lower bounds
- Look for problems where the best algorithm uses a technique that "should"
  be improvable (e.g., brute-force enumeration as a subroutine)
- Check if known results hold in stronger or weaker models -- can you
  prove a result in a more general model, or a stronger result in a
  restricted model?
- Look for connections to other problems -- does a new result in area A
  have implications for area B?

## Step 2: Generate candidate ideas

### Types of theoretical contributions

Not all theory papers are alike. Identify which type of contribution you
are aiming for:

| Contribution type | What you prove | Example statement |
|---|---|---|
| Faster algorithm | Improved time or space | "We give an O(n log n) algorithm for X, improving the O(n^{3/2}) bound of [Author, Year]" |
| New lower bound | Impossibility result | "Any comparison-based algorithm for X requires Omega(n log n) time" |
| Improved approximation | Better ratio | "We give a 1.5-approximation for X, improving the 2-approximation of [Author, Year]" |
| Hardness of approximation | Approximation lower bound | "X is NP-hard to approximate within 1.99 unless P = NP" |
| New algorithmic technique | Framework or method | "We introduce a technique based on Y that yields improved algorithms for problems A, B, and C" |
| Structural / combinatorial | Property of objects | "Every graph with property X also has property Y" |
| Tight bound | Matching upper and lower | "We show that the complexity of X is Theta(n log n)" |
| New complexity class result | Separation or collapse | "We show that X is complete for class Y under Z reductions" |

### The contribution IS the theorem

In theory, the main contribution is a precisely stated theorem. Formulate
your idea as a concrete claim:

- BAD: "We study the X problem using new techniques"
- GOOD: "We give an O(n sqrt(log n)) algorithm for X in the Y model,
  improving the O(n log n) bound of [Author, Year]"

If you cannot state a precise theorem, your idea is not ready.

### What makes a good theory contribution

- **Fundamental problem**: The problem is well-studied, widely known, and
  progress on it would be noted by the community (e.g., maximum flow,
  shortest paths, satisfiability, graph coloring)
- **Significant improvement**: The improvement is more than a constant
  factor -- ideally a polynomial improvement or a qualitatively new bound
- **Novel technique**: The proof introduces a new idea, not a routine
  application of known methods
- **Clean result**: The theorem statement is elegant and easy to state
- **Opens doors**: The technique or result likely leads to further progress

### What makes a BAD theory contribution

- Marginal improvement on a non-fundamental problem (small constant factor
  on an obscure variant)
- Straightforward application of a known technique to a new problem with
  no new insight
- Result that follows easily from combining two known results
- Bounds that are far from tight with no evidence the approach could be
  pushed further
- Unverifiable claims -- the proof is incomplete or contains gaps

## Step 3: Verify novelty (CRITICAL -- do not skip)

Theory has a very long memory. A result may have been proved in the 1970s
in a different community or under a different name.

### Search specifically for your claimed result

- Search for the PROBLEM by all its known names (terminology varies across
  communities and decades)
- Search for the BOUND -- has someone already achieved O(n log n) for this
  problem?
- Check textbooks: Cormen et al. (CLRS), Vazirani (approximation),
  Arora-Barak (complexity), Motwani-Raghavan (randomized algorithms),
  Williamson-Shmoys (approximation), Grotschel-Lovasz-Schrijver,
  Schrijver (combinatorial optimization)
- Check if your result is implied by a more general result
- Search DBLP for all papers by the leading researchers on this problem

### Common novelty traps in theory

- Your result was proved in a different computational model and the
  translation is straightforward
- Your result is a special case of a known, more general theorem
- Your technique was used before for a related problem and the extension
  is routine
- The result is "folklore" -- known to experts but never formally published
  (search lecture notes, blog posts by researchers like Lipton, Trevisan,
  O'Donnell, etc.)
- A concurrent or very recent paper (last few months) proves the same
  or a stronger result

### If your result is already known

- Can you improve the bound further?
- Can you prove it in a stronger model or with weaker assumptions?
- Can you simplify the proof significantly? (Simpler proofs of known
  results ARE publishable in theory, especially for important theorems)
- Can you extend it to a more general setting?
- If none of these, go back to Step 2

## Step 4: Refine and document

### Write your idea.json with:

- **description**: The main theorem statement, stated precisely with
  asymptotic bounds, computational model, and any assumptions
- **motivation**: Why this problem matters, what gap in knowledge it fills,
  and what the previous best bounds were
- **proposed_approach**: The high-level proof strategy -- what technique
  you plan to use and why it should yield the claimed bound
- **related_work**: Key prior results with precise bounds and citations
  (use REAL papers -- include title, authors, venue, year). List the
  history of bounds for the problem.

### Sanity checks before moving on

- Is the theorem statement precise? (model, assumptions, bounds all specified)
- Is the claimed bound plausible? (Does it violate any known lower bound?
  Is it consistent with known special cases?)
- Do you have a proof sketch? (At minimum, know the high-level approach
  and why each step should work)
- Is the improvement significant enough for a venue like STOC/FOCS/SODA/COLT?
- Have you checked all known names and formulations of the problem?

## General principles

### On choosing problems

- Work on problems that are fundamental and well-studied -- a great result
  on a central problem is always valued
- Avoid inventing artificial problems just to solve them -- unless the
  problem captures something genuinely new and interesting
- A clean, surprising result on a simple problem is better than a
  complicated result on a complicated problem
- Closing gaps (matching upper and lower bounds) is inherently satisfying
  and valued by the community

### On techniques

- The technique is often as important as the result -- a new algorithmic
  idea that applies to many problems is more influential than an isolated
  improvement
- Understand why previous approaches got stuck -- this tells you what new
  ingredient is needed
- Look for connections across areas: algebraic, combinatorial, probabilistic,
  geometric, information-theoretic arguments often cross-pollinate
- Simple proofs are better than complicated ones, all else being equal

### On theory culture

- Correctness is paramount -- an incorrect proof is worse than no proof
- Elegance matters -- clean formulations and proofs are valued
- History matters -- cite the originators of problems and techniques,
  not just the most recent paper
- Precise statements matter -- vague or informal claims are not acceptable
