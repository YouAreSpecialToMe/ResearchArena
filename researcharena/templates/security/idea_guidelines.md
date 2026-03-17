# Idea Generation Guidelines (Computer Security)

How to go from a seed topic to a novel, feasible security research idea.

Distilled from practices at CCS, IEEE S&P, USENIX Security, and NDSS,
combined with the ResearchAgent methodology and "Can LLMs Generate Novel
Research Ideas?" (Si et al.).

## Step 1: Explore the field

Start by understanding what already exists. DO NOT skip this step.

### Search for existing work (newest to oldest)
- Search IEEE Xplore (ieeexplore.ieee.org), ACM Digital Library (dl.acm.org),
  USENIX proceedings (usenix.org/conferences), IACR ePrint (eprint.iacr.org),
  and Google Scholar (scholar.google.com) for papers in your seed area
- For vulnerability-related work, search CVE databases (cve.mitre.org, nvd.nist.gov)
  and exploit databases (exploit-db.com) to understand the real-world landscape
- **Start with the newest papers first** — sort by date, read the most
  recent work before going to older foundational papers. Security is a
  fast-moving field; attacks and defenses become outdated quickly.
- Recommended search order:
  1. Last 6 months — what attacks/defenses are current?
  2. Last 1-2 years — what are the state-of-the-art techniques?
  3. Foundational papers — what are the classic approaches?
- Look for:
  - Systematization of Knowledge (SoK) papers — they survey entire subfields
    and identify open problems (highly valued in security venues)
  - Top-tier venue papers (CCS, S&P, USENIX Security, NDSS) — they define
    the standard for rigorous security research
  - Industry reports and CVE advisories — they reveal real-world impact

### Build a mental map
- What are the main attack vectors in this area?
- What defenses exist, and what are their known limitations?
- What threat models do existing papers assume?
- What are the standard evaluation benchmarks and datasets?
- What are the known open problems?
- Are there techniques from adjacent fields (ML, PL, systems, crypto)
  that could apply here?

### Find the gaps
- Read the "Limitations" and "Discussion" sections of recent papers —
  security papers often acknowledge attacks they cannot handle
- Look for defenses that have not been tested against adaptive adversaries
- Identify threat models that are too weak or too strong for real-world use
- Look for areas where measurement studies are missing (how prevalent is X?)
- Check if formal guarantees have been established or if results are purely empirical

## Step 2: Generate candidate ideas

### Types of security contributions

Security research spans several distinct contribution types. Choose one:

| Type | Example | Key requirement |
|---|---|---|
| New attack | Exploiting a vulnerability in protocol X | Demonstrate practical feasibility and real impact |
| New defense | Detecting/preventing attack class Y | Show effectiveness AND acceptable performance overhead |
| Vulnerability analysis | Finding flaws in deployed system Z | Responsible disclosure, systematic methodology |
| Protocol analysis | Formal/empirical analysis of protocol P | Prove security properties or find violations |
| Privacy mechanism | New technique for data privacy | Formal privacy guarantees (e.g., differential privacy) |
| Measurement study | Large-scale analysis of phenomenon W | Statistically rigorous, representative data |
| Systematization of Knowledge | Survey and taxonomy of area A | Comprehensive coverage, new insights from synthesis |

### Two approaches (choose one or combine)

**Goal-driven** (recommended): Start with a security problem.
- "Defense X doesn't handle adaptive adversaries. Can we build one that does?"
- "Protocol Y assumes a trusted third party. Can we remove that assumption?"
- "Attack class Z is growing in the wild but no systematic study exists."
- The goal constrains your search and makes the contribution clear.

**Idea-driven**: Start with a technique and find security applications.
- "Technique A from ML/PL/crypto could detect attack class B."
- Riskier — verify the technique actually provides security guarantees.

### What makes a good security research idea
- **Novel**: Not already done. You MUST verify this (Step 3).
- **Feasible**: Can be implemented, tested, and evaluated within your constraints.
- **Clear threat model**: The attacker's capabilities, goals, and trust
  assumptions are precisely defined.
- **Testable**: There is a concrete evaluation plan (attack success rate,
  defense detection rate, performance overhead, formal proof).
- **Significant**: The attack is practical, the defense is deployable, or
  the analysis reveals important findings about real systems.

### What makes a BAD security research idea
- No clear threat model ("we defend against all attacks")
- Unrealistic threat model — too strong attacker (knows everything) or
  too weak attacker (nobody would bother)
- Attack on a system nobody uses or a protocol nobody deploys
- Defense that has unacceptable performance overhead (10x slowdown)
- Defense that doesn't consider adversarial evasion
- Measurement study on non-representative data
- Already exists (you didn't check the literature)
- Requires access to systems or data you don't have

### The threat model (CRITICAL for security research)

Every security idea MUST define a clear threat model. Before proceeding, answer:

1. **Who is the attacker?** (remote network attacker, malicious insider,
   compromised library, nation-state, etc.)
2. **What are the attacker's capabilities?** (can they modify network traffic?
   execute code on the target? control training data? access side channels?)
3. **What are the attacker's goals?** (steal data, deny service, evade detection,
   compromise integrity, violate privacy?)
4. **What are the trust assumptions?** (what components are trusted? what is
   outside the attacker's reach?)
5. **What is in scope and out of scope?** (physical attacks? social engineering?
   denial of service?)

A paper without a precise threat model will be rejected at any top venue.

## Step 3: Verify novelty (CRITICAL -- do not skip)

Before committing to an idea, verify it hasn't been done:

### Search specifically for your idea
- Search IEEE Xplore, ACM DL, USENIX proceedings, and IACR ePrint with
  keywords from your proposed attack/defense/analysis
- Search for the PROBLEM you're addressing, not just your approach
- Check if your attack variant has already been demonstrated
- Check if your defense mechanism is a special case of existing work
- Look at the "Related Work" sections of papers closest to your idea

### Common novelty traps in security
- Your attack exists but under a different name (security jargon varies
  across communities: systems security, network security, crypto, PL)
- Your defense was proposed but shown to be bypassable (check follow-up work)
- Your idea is a straightforward application of a known technique to a
  known problem without new insight
- A concurrent paper (posted in the last few months) does the same thing
- The vulnerability you found was already reported in a CVE

### If your idea already exists
- Can you demonstrate the attack on a newer or more realistic system?
- Can you improve the defense to handle evasion attacks?
- Can you provide formal guarantees where only empirical results existed?
- Can you scale the analysis to a larger, more representative dataset?
- If it truly exists with no room for improvement, go back to Step 2

## Step 4: Refine and document

### Write your idea.json with:
- **description**: 1-3 sentences explaining what you're proposing
- **motivation**: why this problem matters, what gap you're filling,
  what real-world impact this has
- **proposed_approach**: your high-level method and why it should work
- **threat_model**: attacker capabilities, trust assumptions, scope
- **related_work**: key existing papers and how your idea differs
  (use REAL papers you found in Steps 1 and 3 -- include titles and authors)

### Sanity checks before moving on
- Is the threat model clearly defined and realistic?
- Can you explain the idea in one sentence to a security researcher?
- Is there a clear experiment or proof that would validate the idea?
- For attacks: is the target system relevant and widely deployed?
- For defenses: is the performance overhead likely acceptable?
- Do you have the resources (systems, data, tools) to evaluate it?
- Is the expected contribution significant enough for CCS/S&P/USENIX/NDSS?

### Ethical considerations
- If your research discovers real vulnerabilities, plan for responsible
  disclosure BEFORE starting experiments
- Follow your institution's IRB requirements for any human-subjects research
- Consider dual-use implications: could your attack be misused?
- Document your ethical reasoning -- reviewers will evaluate it

## General principles

### From John Schulman (adapted for security)
- Your ability to choose the right problem is more important than raw skill
- Watch which security problems matter in the real world -- this develops taste
- Goal-driven research (solving a real security problem) has lower scooping
  risk than technique-driven research
- There's no shame in working on ideas from the literature or from real incidents

### From "Can LLMs Generate Novel Research Ideas?" (Si et al.)
- AI-generated ideas tend to be novel but lack feasibility -- ground yours
  in practical security constraints
- Vague threat models and hand-wavy security arguments are the #1 weakness
- Missing comparisons to existing defenses are a common failure
- Verify your idea against existing work -- reviewers WILL find the paper
  you missed

### Security-specific principles
- Think adversarially: if you propose a defense, immediately ask "how would
  I bypass this?" If you can think of an evasion, so will the reviewer.
- Realism matters: attacks on toy systems and defenses with unrealistic
  assumptions will be rejected
- Impact matters: connect your work to real-world systems, protocols,
  or deployments whenever possible
- Rigor matters: formal proofs, systematic evaluations, and honest
  limitations analysis are expected at top venues
