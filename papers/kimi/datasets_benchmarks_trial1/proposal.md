# CompViz: A Dynamic Benchmark for Compositional Visual Reasoning with Sub-100ms Generation

## Enabling Contamination-Free Evaluation Through Ultra-Fast Procedural Generation

---

## 1. Introduction

### 1.1 Context and Problem Statement

Vision-language models (VLMs) have demonstrated remarkable capabilities on standard benchmarks, but these static evaluations suffer from three critical limitations:

1. **Benchmark Saturation**: Models achieve near-ceiling performance on CLEVR (>97%) and static VQA datasets, making it difficult to distinguish capabilities or diagnose specific failure modes.

2. **Data Contamination**: Training datasets increasingly overlap with evaluation benchmarks, inflating scores without true generalization improvements. This is particularly acute for VLMs trained on web-scale data.

3. **Inability to Support Dynamic Evaluation**: Static benchmarks cannot adapt difficulty during evaluation, generate fresh test sets per run, or support curriculum learning—capabilities that require generation speeds incompatible with current procedural benchmarks.

Recent surveys (Ke et al., 2025) explicitly identify the need for benchmarks with "explicit quantification of reasoning difficulty levels." However, existing procedural benchmarks either:
- Require slow 3D rendering (InfiniBench: ~5 min/frame)
- Use complex agentic pipelines (MM-CondChain: hours per batch)
- Offer limited reasoning diversity (CVR: odd-one-out only)
- Focus on mathematical rather than logical reasoning (DynaMath: geometry, algebra, arithmetic)

### 1.2 Concurrent and Related Work

We explicitly acknowledge highly relevant concurrent and recent works:

**MM-CondChain** (Shen et al., 2026, March 12) implements multi-domain compositional reasoning with chain depth control through a sophisticated agentic synthesis pipeline using Verifiable Programmatic Intermediate Representation (VPIR). Their benchmark covers natural images, charts, and GUI trajectories with conditional control flow reasoning.

**MentisOculi** (Wiedemer et al., 2026, February) provides stratified difficulty for mental imagery tasks requiring visual state manipulation (Form Board, Rush Hour, Paper Fold).

**DynaMath** (Zou et al., 2024, October) is a dynamic visual math benchmark with 501 seed questions covering mathematical topics: Plane Geometry, Solid Geometry, Analytic Geometry, Algebra, Graph Theory, Statistics, and Arithmetic. Each seed is a handcrafted Python program generating question variants.

**Critical Insight**: While these benchmarks advance the field, they share fundamental constraints that CompViz addresses:
- **MM-CondChain**: Hours-long agentic generation prevents dynamic paradigms
- **MentisOculi**: Minutes per instance limits scalability  
- **DynaMath**: Handcrafted seed programs (501 total) limit diversity; focuses on mathematical reasoning rather than logical/quantitative reasoning with nested quantifiers

### 1.3 Key Innovation: Speed as Enabler of New Evaluation Paradigms

Our key insight is that **generation speed is not merely a convenience but enables entirely new evaluation capabilities**. By targeting **<100ms generation time** through lightweight 2D vector graphics, CompViz enables:

1. **Contamination-Free Per-Run Evaluation**: Generate fresh test sets for each evaluation run
2. **Dynamic Difficulty Adjustment**: Adapt task difficulty based on model performance in real-time
3. **Large-Scale Training Data**: Generate millions of training examples for curriculum learning
4. **Iterative Model Development**: Rapid hypothesis testing with custom-generated datasets

These capabilities are **infeasible** with MM-CondChain's hours-long generation, InfiniBench's minutes-per-frame rendering, or DynaMath's handcrafted seed programs.

### 1.4 Unique Reasoning Focus: Nested Quantification and Relational Chains

Beyond speed, CompViz targets **distinct reasoning challenges** from existing work:

| Reasoning Type | CompViz | MM-CondChain | MentisOculi | DynaMath |
|----------------|---------|--------------|-------------|----------|
| **Nested Quantifiers** | ∃x∀y: R(x,y) | ✗ | ✗ | ✗ |
| **Transitive Relations** | A→B→C chains | Limited | ✗ | ✗ |
| **Conditional Control Flow** | ✗ | ✓ | ✗ | ✗ |
| **Mental Imagery** | ✗ | ✗ | ✓ | ✗ |
| **Mathematical Reasoning** | ✗ | ✗ | ✗ | ✓ |

**Specific differentiators:**
- **Nested Quantification**: CompViz uniquely tests "Exists-Forall" reasoning (e.g., "Is there a shape that is larger than ALL red circles?"). This requires models to verify universal statements within existential contexts—a known challenge for neural models.
- **Transitive Relational Chains**: Multi-hop spatial reasoning ("Is the star left of the circle that is above the square?") with explicit composition depth control.
- **Quantitative Comparison**: Counting and comparative quantification ("Are there more striped triangles than solid circles?") with symbolic ground truth.

### 1.5 Differentiation from DynaMath

While both DynaMath and CompViz use procedural generation, they are **complementary benchmarks targeting different reasoning domains**:

| Aspect | DynaMath | CompViz |
|--------|----------|---------|
| **Domain** | Mathematical reasoning (geometry, algebra, arithmetic) | Logical/quantitative reasoning |
| **Question Source** | 501 handcrafted seed programs | Compositional grammar (unlimited) |
| **Visual Type** | Function plots, geometric diagrams, scientific figures | Abstract 2D shapes with attributes |
| **Core Challenge** | Mathematical problem-solving | Nested quantification, transitive relations |
| **Generation Speed** | Seconds per instance (handcrafted code) | **<100ms target** (optimized pipeline) |
| **Difficulty Control** | Pre-defined school levels (elementary, high school, undergrad) | Fine-grained composition depth (1-4+) |

**Key distinction**: DynaMath tests whether VLMs can solve math problems robustly across variations. CompViz tests whether VLMs can handle nested logical quantification and transitive relational chains—capabilities orthogonal to mathematical reasoning.

### 1.6 Hypothesis

A benchmark with:
1. **Sub-100ms generation** enabling dynamic evaluation paradigms
2. **Nested quantification and transitive relation challenges** distinct from conditional control flow and mathematical reasoning
3. **Fine-grained difficulty control** through composition depth
4. **2D abstract patterns** that isolate reasoning from perception challenges

will:
1. Demonstrate that speed enables practical contamination-free evaluation
2. Reveal specific failures on nested quantification and transitive reasoning invisible to existing benchmarks
3. Provide discrimination between VLMs even as they saturate existing benchmarks
4. Show failure modes complementary to MM-CondChain (logical/quantitative vs. conditional/branching) and DynaMath (logical vs. mathematical)

---

## 2. Proposed Approach

### 2.1 System Architecture

CompViz generates visual reasoning tasks through three stages:

```
Scene Generator (SVG-based) → Query Generator (Grammar-based) → Answer Computer (Symbolic)
      <50ms                        <30ms                           <20ms
```

**Target generation time: <100ms per instance** (to be validated in Section 2.5)

### 2.2 Visual Grammar

**Primitives:**
- Shapes: circle, square, triangle, star, pentagon, hexagon
- Colors: red, blue, green, yellow, purple, orange, cyan, magenta
- Attributes: size (small/medium/large), rotation (0-360°), texture (solid/striped/dotted/checkered), opacity (0.3-1.0)
- Relations: left-of, right-of, above, below, inside, touching, aligned-with, closer-to

**Compositional Operators:**
- Boolean: AND, OR, NOT, XOR
- **Quantification**: ∃ (exists), ∀ (for all), exactly-n, at-most-n, at-least-n
- **Comparison**: same-as, larger-than, different-from, count-comparison, ratio-comparison
- **Relational**: direct-relation, transitive-chain (A→B→C→D)

### 2.3 Task Taxonomy with Explicit Difficulty

| Task Type | Depth 1 | Depth 2 | Depth 3 | Depth 4+ |
|-----------|---------|---------|---------|----------|
| **Existential** | "Is there a red circle?" | "Is there a red circle above a blue square?" | "Is there a large red circle above a small blue square?" | "Is there an object larger than all red circles AND left of some blue square?" |
| **Universal** | "Are all circles red?" | "Are all red objects circles?" | "Are all objects above a red circle also small?" | "∀x: Red(x) → ∃y: Above(x,y) ∧ Blue(y)" |
| **Comparative** | "How many circles?" | "Are there more circles than squares?" | "Are there more red circles than green triangles?" | "Is the sum of large red objects greater than small blue objects?" |
| **Transitive** | — | "Is A left of B?" | "Is A left of B which is above C?" | "Is there a chain A→B→C→D where each relation differs?" |
| **Nested Quant** | — | — | "∃x∀y: Larger(x,y) if Blue(y)" | "∃x∀y∃z: Between(x,y,z) ∧ DifferentColor(y,z)" |

### 2.4 Query Naturalness and Comprehensibility Validation

**Acknowledged Challenge**: Nested quantification queries (e.g., "Is there a shape larger than ALL red circles?") may present linguistic complexity for both models and human annotators. This is a potential confound we explicitly address:

**Validation Approach:**
1. **Human Comprehension Pilot** (n=20): Test query understanding with human subjects
2. **Syntactic Complexity Metrics**: Measure query length, embedding depth, and quantifier nesting
3. **Alternative Formulations**: Generate semantically equivalent but syntactically simpler variants:
   - Complex: "Is there a shape larger than ALL red circles?"
   - Simplified: "Find a shape. Is it larger than every red circle?"
4. **Linguistic Controls**: Ensure comparable syntactic complexity across difficulty levels to isolate reasoning difficulty from parsing difficulty

**Mitigation Strategy**: If nested quantification proves too linguistically complex, we will:
- Use progressive phrasing (breaking into sub-questions)
- Compare performance on logically equivalent but syntactically different formulations
- Report parsing accuracy separately from reasoning accuracy

### 2.5 Target Timing and Validation Plan

**Target Performance** (based on architectural analysis):

| Component | Target | Basis |
|-----------|--------|-------|
| Scene Generation | <50ms | SVG primitive rendering with cairosvg |
| Query Generation | <30ms | Grammar sampling with pre-computed templates |
| Answer Computation | <20ms | Symbolic execution on scene graphs |
| **Total** | **<100ms** | **Conservative sum of components** |

**Validation Plan:**
- **Phase 1** (Week 1): Implement core pipeline and measure baseline timing
- **Phase 2** (Week 2): Optimize bottlenecks and validate across difficulty levels
- **Success Criteria**: Mean <100ms with σ<20ms on reference hardware

**Important Clarification**: The timing figures represent **engineering targets** based on comparable SVG generation benchmarks and will be validated during implementation. They are not pre-validated measurements.

### 2.6 Fine-Grained Difficulty Scaling

Difficulty scales along quantified dimensions:

| Dimension | Level 1 | Level 2 | Level 3 | Level 4 |
|-----------|---------|---------|---------|---------|
| **Objects** | 3-5 | 6-8 | 9-12 | 13-20 |
| **Composition Depth** | 1 | 2 | 3 | 4+ |
| **Distractors** | 0 | 1-2 | 3-4 | 5+ |
| **Attribute Conjunctions** | 1 | 2 | 3 | 4+ |
| **Relation Hops** | 0 | 1 | 2 | 3+ |

**Composition Depth** is formally defined as the maximum operator nesting in the functional program.

### 2.7 Generation Pipeline Failure Modes and Mitigations

**Acknowledged Pipeline Challenges:**

1. **Scene Incoherence**: Random placement may create visually cluttered or ambiguous scenes
   - *Mitigation*: Minimum object separation constraints; rejection sampling for layout validity

2. **Query Ambiguity**: Generated queries may have ambiguous answers due to edge cases
   - *Mitigation*: Symbolic verification of unique answers; rejection sampling for unambiguous queries

3. **Attribute Binding Failures**: Queries may reference non-existent attribute combinations
   - *Mitigation*: Scene-aware query generation ensuring referenced objects exist

4. **Quantifier Scope Errors**: Nested quantifiers may create vacuously true/false conditions
   - *Mitigation*: Automated logical verification; filter queries with trivial truth values

5. **Rendering Variability**: SVG rendering differences across platforms
   - *Mitigation*: Standardized rendering with cairosvg; fixed random seeds for reproducibility

---

## 3. Related Work and Positioning

### 3.1 Detailed Differentiation

**MM-CondChain** (Shen et al., 2026):
- **Domain**: Natural images, charts, GUI trajectories
- **Reasoning focus**: Conditional control flow (if-then-else branching)
- **Generation**: Agentic synthesis pipeline with VPIR verification
- **Speed**: Hours per batch

**Differentiators**: CompViz tests nested quantification and transitive relations; MM-CondChain tests conditional branching. CompViz is 10,000× faster, enabling dynamic evaluation.

**MentisOculi** (Wiedemer et al., 2026):
- **Task type**: Visual state manipulation (mental imagery puzzles)
- **Focus**: Whether models benefit from self-generated visualizations
- **Speed**: Moderate (minutes per instance)

**Differentiators**: CompViz tests direct visual reasoning; MentisOculi tests mental imagery. CompViz focuses on logical/quantitative; MentisOculi on spatial manipulation.

**DynaMath** (Zou et al., 2024):
- **Domain**: Mathematical reasoning (geometry, algebra, arithmetic)
- **Approach**: 501 handcrafted seed programs generating variants
- **Focus**: Mathematical reasoning robustness
- **Speed**: Seconds per instance

**Differentiators**: CompViz targets logical reasoning with nested quantifiers; DynaMath targets mathematical reasoning. CompViz uses compositional grammar for unlimited diversity; DynaMath uses handcrafted seeds. CompViz targets <100ms generation; DynaMath uses sophisticated rendering for mathematical figures.

### 3.2 Established Benchmarks

**CLEVR** (Johnson et al., 2017) pioneered procedural visual reasoning but has been saturated (>97% accuracy) and offers limited attribute diversity.

**CVR** (Zerroug et al., 2022) generates odd-one-out tasks for relation learning. We include direct comparison experiments in Section 4.3.

**InfiniBench** (Wang et al., 2025) generates photorealistic 3D scenes for spatial reasoning. At high quality: ~5 min/frame. Focuses on 3D spatial reasoning vs. our logical/quantitative focus.

### 3.3 Ecosystem Positioning

| Benchmark | Domain | Reasoning | Speed | Best For |
|-----------|--------|-----------|-------|----------|
| MM-CondChain | Natural images/charts/GUIs | Conditional control flow | Hours | Comprehensive workflow eval |
| MentisOculi | Mental imagery puzzles | Visual state maintenance | Minutes | Mental imagery research |
| DynaMath | Math diagrams/plots | Mathematical reasoning | Seconds | Math robustness testing |
| InfiniBench | 3D photorealistic | Spatial reasoning | ~5 min | 3D spatial evaluation |
| CVR | Abstract 2D | Odd-one-out relations | Fast | Relation learning |
| **CompViz** | **2D abstract** | **Logical/Quantitative/Nested** | **<100ms** | **Dynamic/contamination-free eval** |

---

## 4. Experiments

### 4.1 Research Questions

1. **RQ1 (Speed Validation)**: Can CompViz achieve <100ms generation across difficulty levels?

2. **RQ2 (Discriminative Power)**: Do VLMs show significant performance gaps between difficulty levels on CompViz?

3. **RQ3 (Nested Quantification)**: Which models struggle most with nested quantifiers (∃∀, ∀∃)?

4. **RQ4 (Transitive Relations)**: How does accuracy degrade with relational chain length (1-hop vs 2-hop vs 3-hop)?

5. **RQ4.5 (Query Naturalness)**: Do syntactically complex nested quantification queries affect performance independently of reasoning difficulty?

6. **RQ5 (Contamination-Free Eval)**: Does per-run test generation reveal different performance than fixed test sets?

7. **RQ6 (Failure Mode Analysis)**: Are CompViz failure modes complementary to MM-CondChain and DynaMath?

### 4.2 Experimental Setup

**Models:**
- Open-source: LLaVA-1.6-7B, InternVL2-4B, Qwen2-VL-7B, Phi-3.5-Vision
- API: GPT-4o Vision, Claude 3.5 Sonnet, Gemini 1.5 Flash/Pro

**Test Sets:**
- Fixed: 2,000 instances per difficulty level (for comparison studies)
- Dynamic: Fresh 1,000 instances generated per evaluation run (for contamination-free study)

### 4.3 Direct Comparison with CVR

To enable cross-benchmark analysis:
1. Implement CVR odd-one-out tasks in CompViz framework
2. Evaluate identical model suites on both
3. Measure correlation and identify incremental signal

### 4.4 Query Naturalness Validation Experiment

**Objective**: Isolate linguistic complexity from reasoning complexity.

**Method:**
1. Generate 100 Depth-3 nested quantification queries
2. Create syntactically simplified variants preserving semantics
3. Test same models on both versions
4. Measure performance gap

**Expected Outcome**: <10% accuracy difference indicates reasoning (not parsing) is the primary challenge.

### 4.5 Failure Mode Analysis vs MM-CondChain and DynaMath

We analyze published examples to identify complementary failure modes:

| Benchmark | Primary Challenge | Expected Error Pattern |
|-----------|-------------------|----------------------|
| MM-CondChain | Conditional branching | Path prediction errors |
| DynaMath | Mathematical reasoning | Calculation errors, formula misapplication |
| CompViz | Nested quantification | Scope errors, quantifier flipping |

**Analysis Methodology:**
Manually analyze 50 examples from each published dataset, categorizing:
1. Reasoning operation types (quantification vs. conditionals vs. math)
2. Error patterns expected for each benchmark

### 4.6 Experiment: Contamination-Free Per-Run Evaluation

**Objective**: Demonstrate practical contamination-free evaluation.

**Method:**
1. Run 5 independent evaluation runs
2. For each run, generate fresh 1,000 test instances with different random seeds
3. Measure performance variance across runs
4. Compare to performance on a fixed test set

**Expected Result**: Low variance across runs confirms stability; any run provides valid evaluation without risk of training set overlap.

### 4.7 Curriculum Learning (Reduced Scope)

**Note**: Full curriculum learning with LoRA fine-tuning is deferred due to timeline constraints. We include a **feasibility demonstration**:

**Method:**
1. Generate 10K training instances at Level 1 and 10K at Level 4
2. Generate 1K validation instances at Level 4
3. Demonstrate that generation completes in <15 minutes
4. This validates the *enabling capability* without requiring full training experiments

**Future Work**: Full curriculum learning experiments with fine-tuning will be conducted with additional compute resources.

### 4.8 Expected Results

**H1 (Speed)**: Mean generation time <100ms with σ<20ms across all difficulty levels.

**H2 (Discrimination)**: >30% accuracy gap between Level 1 and Level 4 for all tested models.

**H3 (Nested Quantification)**: ∃∀ and ∀∃ patterns show >25% accuracy drop compared to simple existential (∃) patterns.

**H4 (Transitive Relations)**: Accuracy follows inverse relationship with chain length: Acc ∝ 1/(1+0.3×hops).

**H5 (Query Naturalness)**: Performance difference between complex and simplified query formulations <10%.

**H6 (Contamination-Free)**: Standard deviation across 5 fresh runs <3%.

**H7 (Complementary Failures)**: Error patterns uncorrelated with MM-CondChain and DynaMath.

---

## 5. Success Criteria

### 5.1 Confirming Evidence

The hypothesis is supported if:
- [ ] Generation time <100ms with σ<20ms (validated during implementation)
- [ ] >30% accuracy gap between easiest and hardest difficulty
- [ ] Nested quantification shows unique difficulty profile vs. other reasoning types
- [ ] Transitive relation accuracy degrades predictably with chain length
- [ ] Query naturalness validation shows <10% parsing vs. reasoning confound
- [ ] Per-run evaluation shows <3% variance across fresh test sets
- [ ] Failure mode analysis shows distinct patterns from MM-CondChain and DynaMath

### 5.2 Refuting Evidence

The hypothesis is challenged if:
- [ ] All models achieve >90% on hardest difficulty (benchmark too easy)
- [ ] No performance difference between reasoning types
- [ ] Generation time >200ms (loses key differentiator)
- [ ] Perfect correlation with existing benchmarks (no new signal)
- [ ] Query naturalness confound >20% (linguistic complexity dominates)

---

## 6. Limitations and Future Work

### 6.1 Limitations

1. **Synthetic Visuals**: 2D abstract patterns do not test natural image understanding—a deliberate trade-off for speed.

2. **Language-Dependent**: Queries require language understanding; we cannot isolate pure visual reasoning.

3. **Query Complexity**: Nested quantification queries may be linguistically complex. We address this through validation (Section 2.4) but acknowledge this as a potential confound.

4. **2D Constraints**: Does not test 3D spatial reasoning (addressed by InfiniBench) or conditional control flow (addressed by MM-CondChain) or mathematical reasoning (addressed by DynaMath).

5. **Scope Constraints**: Full curriculum learning with fine-tuning is deferred due to 8-hour compute budget.

### 6.2 Future Work

1. **Cross-Benchmark Validation**: Direct comparison if MM-CondChain/DynaMath datasets become available
2. **Query Simplification**: Develop progressive/natural phrasings for nested quantification
3. **Full Curriculum Learning**: Complete training experiments with LoRA fine-tuning
4. **Hybrid Domain**: Combine procedural generation with domain-specific templates (scientific diagrams, geometric proofs)

---

## 7. References

1. Johnson, J., Hariharan, B., van der Maaten, L., et al. (2017). CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning. CVPR.

2. Zerroug, A., Vaishnav, M., Feather, J., et al. (2022). A Benchmark for Compositional Visual Reasoning. NeurIPS.

3. Wang, H., Xue, Q., & Gao, W. (2025). InfiniBench: Infinite Benchmarking for Visual Spatial Reasoning with Customizable Scene Complexity. arXiv:2511.18200.

4. Ke, F., Hsu, J., Cai, Z., et al. (2025). Explain Before You Answer: A Survey on Compositional Visual Reasoning. arXiv:2508.17298.

5. Shen, H., Yan, S., Xue, H., et al. (2026). MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning. arXiv:2603.12266.

6. Wiedemer, T., Li, F., Klein, T., et al. (2026). MentisOculi: Revealing the Limits of Reasoning with Mental Imagery. arXiv:2602.02465.

7. Zou, C., Guo, X., Yang, R., et al. (2024). DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models. arXiv:2411.00836.

8. Zou, B., Cai, M., Zhang, J., & Lee, Y. J. (2024). VGBench: Evaluating Large Language Models on Vector Graphics Understanding and Generation. EMNLP.

9. White, C., Dooley, S., Goldblum, M., et al. (2024). LiveBench: A Challenging, Contamination-Limited LLM Benchmark. arXiv:2406.19314.

10. Zhu, K., Chen, J., Wang, J., et al. (2024). DyVal: Dynamic Evaluation of Large Language Models for Reasoning Tasks. ICLR.

11. Liu, Y., Duan, H., Zhang, Y., et al. (2024). MMBench: Is Your Multi-modal Model an All-around Player? arXiv:2307.06281.

---

## 8. Resource Requirements

### 8.1 Compute Budget

- **Scene Generation**: CPU-based, targeting >10K instances/hour
- **Model Evaluation**:
  - Local models: ~40 examples/GPU hour on A6000
  - API calls: ~$80 for full evaluation suite
- **Timing Validation**: <1 hour for comprehensive profiling
- **Total**: Well within 8-hour budget (estimated ~5 hours)

### 8.2 Storage

- Generated scenes: ~30KB per instance (SVG)
- 10K test scenes: ~300MB
- Model outputs: ~30MB
- **Total: <1GB**

---

## 9. Expected Impact

CompViz contributes to the emerging ecosystem of next-generation benchmarks through **practical capabilities enabled by speed**:

1. **Dynamic Evaluation**: A visual reasoning benchmark practical for per-run test generation, enabling truly contamination-free evaluation

2. **Complementary Tool**: Explicitly positioned alongside MM-CondChain, MentisOculi, and DynaMath—researchers use CompViz for rapid iteration and dynamic evaluation, MM-CondChain for comprehensive multi-domain assessment, DynaMath for mathematical reasoning robustness

3. **Open Source**: Immediate code release to establish community adoption

**Key Contribution**: Not merely another compositional benchmark, but a demonstration that **speed enables new evaluation paradigms**—dynamic difficulty adjustment, curriculum learning at scale, and per-run contamination-free evaluation—that are computationally infeasible with existing procedural benchmarks, with **unique focus on nested quantification and transitive reasoning** orthogonal to mathematical reasoning (DynaMath) and conditional control flow (MM-CondChain).
