# ProgramClean: Zero-Shot Data Cleaning via LLM-Based Constraint Program Synthesis

## 1. Introduction

### 1.1 Problem Statement

Data cleaning remains a critical bottleneck in data science pipelines. Despite decades of research, most automated solutions face a fundamental trade-off:

- **Statistical methods** (e.g., Raha, Baran, HoloClean) are fast and scalable but semantically blind—they detect anomalies without understanding what the data *means*
- **LLM-based methods** (e.g., Cocoon, IterClean) bring semantic awareness but require expensive per-cell or per-batch LLM inference, making them impractical for large datasets

Recent work has attempted to bridge this gap:
- **Auto-Test (SIGMOD 2025)** learns Semantic-Domain Constraints from table corpora using statistical tests—powerful but requires large pre-existing datasets and cannot work zero-shot on new domains
- **Cocoon (Oct 2024)** decomposes cleaning into sub-tasks handled by LLMs—effective but requires iterative LLM prompting for detection decisions on data batches
- **Akella et al. (EMNLP 2025)** synthesizes code validators using LLMs, but relies on RAG and iteratively-generated few-shot examples, generating individual validators per rule rather than unified constraint programs with coupled detection-repair logic
- **IterClean (ACM-TURC 2024)** uses an iterative detector-verifier-repairer architecture with LLMs, but requires labeled tuples and performs per-batch LLM inference

**The fundamental question**: Can we achieve semantic-aware data cleaning at statistical-method speeds, without requiring training corpora, labeled examples, or per-cell LLM inference?

### 1.2 Key Insight

We propose treating the LLM not as a detector (that classifies each cell) nor as a statistical learner (that mines patterns from corpora), but as a **compiler** that generates executable constraint programs from semantic understanding.

**Our approach compared to related work**:
- **Auto-Test** → Statistical learning from corpora → generates constraints
- **Cocoon/IterClean** → LLM as detector → iteratively prompts LLM for decisions  
- **Akella et al.** → LLM generates individual validators → per-rule code synthesis with RAG and few-shot examples
- **SEED** → LLM as compiler → generates hybrid pipelines for general data curation tasks (entity resolution, annotation, imputation)
- **ProgramClean** → LLM as compiler → generates unified executable Python constraint programs per column with coupled detection-repair logic

By compiling semantic understanding into deterministic programs once per column (not once per cell or per rule), we achieve both semantic awareness and statistical efficiency.

### 1.3 Hypothesis

**Hypothesis**: LLMs can effectively compile column semantics and value patterns into executable Python constraint programs that detect and repair errors with accuracy comparable to iterative LLM methods but with computational cost comparable to statistical methods.

## 2. Related Work

### 2.1 Statistical Error Detection

**Raha and Baran (Mahdavi et al., 2019; Mahdavi & Abedjan, 2020)** pioneered configuration-free error detection using ensemble learning over multiple detection strategies. These methods are fast but lack semantic understanding—they detect "unusual" values without knowing whether those values are semantically invalid.

**HoloClean (Rekatsinas et al., 2017)** uses probabilistic inference to repair data by modeling integrity constraints and correlations. Requires manually specified denial constraints.

### 2.2 Learning-Based Constraint Discovery

**Auto-Test (He et al., SIGMOD 2025)** represents the state-of-the-art in learning semantic constraints. It learns Semantic-Domain Constraints (SDCs) from large table corpora using statistical tests. Key characteristics:
- **Approach**: Statistical learning from table corpora
- **Strength**: No LLM required, highly efficient
- **Limitation**: Requires large pre-existing table corpora for training; cannot work zero-shot on new domains

**Distinction from our work**: Auto-Test learns constraints statistically from data; we *synthesize* constraint programs semantically using LLMs as compilers. We require no training corpus.

### 2.3 LLM-as-Compiler for Data Curation

**SEED (Cui et al., arXiv:2310.00749, 2024)** presents an LLM-as-compiler approach for domain-specific data curation. SEED compiles natural language task descriptions into hybrid execution pipelines combining multiple modules: CacheReuse (vector-based caching), CodeGen (LLM-generated code), ModelGen (small models trained on LLM-annotated data), and raw LLM inference. SEED targets general data curation tasks including entity resolution, data annotation, data imputation, and data discovery.

**Why SEED does not solve the data cleaning problem**: While SEED shares the "LLM-as-compiler" framing, it addresses fundamentally different challenges:
1. **Task scope**: SEED targets general data curation (entity resolution, annotation, imputation); ProgramClean specifically targets constraint-based error detection and repair
2. **Input requirements**: SEED requires natural language task descriptions; ProgramClean works zero-shot from column metadata alone
3. **Output structure**: SEED generates hybrid pipelines with multiple independent modules; ProgramClean synthesizes unified constraint programs with coupled detection-repair logic
4. **Code generation**: SEED generates and ensembles multiple code snippets (M-best candidates); ProgramClean synthesizes single coherent constraint programs per column
5. **Optimization focus**: SEED uses an optimizer to trade off accuracy vs. cost; ProgramClean focuses on deterministic constraint satisfaction

**Distinction from SEED**: SEED is complementary—it addresses general data curation workflows while ProgramClean specifically targets constraint-based data cleaning with unified program synthesis and coupled detection-repair generation.

### 2.4 LLM-Based Data Cleaning

**Cocoon (Zhang et al., 2024)** introduces a system that leverages LLMs for semantic understanding combined with statistical detection. Key characteristics:
- **Approach**: Decomposes cleaning into sub-tasks (string outliers, pattern outliers, DMV, etc.)
- **Method**: Iteratively prompts LLMs for detection and cleaning decisions on data batches
- **Strength**: Strong semantic understanding through direct LLM inference
- **Limitation**: Requires per-batch LLM inference; computationally expensive

**Distinction from our work**: Cocoon uses LLMs as detectors (making decisions via batch-wise prompting); we use LLMs as compilers (generating programs that make decisions). Our approach shifts LLM inference from O(cells) at runtime to O(columns) at compile-time.

**IterClean (Ni et al., ACM-TURC 2024)** proposes an iterative cleaning framework with three LLM-based roles: error detector, verifier, and repairer. Key characteristics:
- **Approach**: Iterative detection-verification-repair cycle with LLMs
- **Method**: Batch prompting based on subject attribute grouping
- **Strength**: Handles multi-type errors through iteration; achieves high F1 with only 5 labeled tuples
- **Limitation**: Requires labeled tuples for few-shot prompting; per-batch LLM inference throughout cleaning

**Distinction from our work**: IterClean relies on few-shot learning from labeled tuples and performs iterative LLM inference at cleaning time. ProgramClean requires no labeled data and performs LLM inference only once per column during compilation.

**Akella et al. (EMNLP 2025)** combines statistical inlier detection with LLM-driven rule and code generation. Key characteristics:
- **Approach**: Three-stage framework (statistical inlier detection → rule generation → code synthesis)
- **Method**: Uses RAG with external knowledge sources and domain-specific few-shot examples retrieved via embedding similarity; generates validators through code-generating LLMs
- **Strength**: Combines statistical and semantic approaches; generates executable validators
- **Limitation**: Generates individual validators per rule (not unified programs); relies on RAG and iteratively-generated few-shot examples; does not synthesize repair functions

**Technical distinction from Akella et al.**:

| Aspect | Akella et al. | ProgramClean |
|--------|--------------|--------------|
| **Program structure** | Individual validators per rule | Unified constraint program per column (validation + repair functions) |
| **Synthesis methodology** | Rule-first: generates rules, then compiles to code | Semantic-profile-first: compiles semantic understanding directly to constraint programs |
| **External knowledge** | Requires RAG + few-shot examples from domain repository | True zero-shot: column metadata and samples only |
| **Repair capability** | Detection only (no automated repair generation) | Unified detection + repair function synthesis |
| **Execution model** | Multiple standalone validators | Single compiled program with coupled validation/repair logic |
| **Synthesis scope** | Per-rule code generation | Per-column unified program synthesis |

**ZS4C (Kabir et al., 2024)** performs zero-shot synthesis of compilable code for incomplete snippets using LLMs with validator feedback. While ZS4C demonstrates zero-shot code synthesis, it addresses a different problem (completing code snippets from Stack Overflow) and does not apply to data cleaning or constraint generation.

### 2.5 Key Gap

No existing method combines: (1) true zero-shot operation (no labeled data, no training corpus, no RAG, no few-shot examples), (2) unified constraint program synthesis (not individual validators), (3) coupled detection-repair generation, and (4) O(columns) LLM complexity with O(cells) native execution.

## 3. Proposed Approach: ProgramClean

### 3.1 Overview

ProgramClean operates in three phases:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Semantic       │     │  Constraint      │     │  Executable     │
│  Profiling      │────▶│  Program         │────▶│  Detection &    │
│  (LLM)          │     │  Synthesis (LLM) │     │  Repair         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
   One-time per            One-time per             Scalable execution
   column/table           column/table             (no LLM calls)
```

### 3.2 Phase 1: Semantic Profiling

Given a column, the LLM analyzes:
- Column name and context
- Sample values (statistical profile: min, max, distinct count, pattern distribution)
- Detected patterns via automatic profiling

Output: A structured semantic profile specifying:
- Semantic type (e.g., "US ZIP code", "email address", "date in MM/DD/YYYY format")
- Expected value constraints (range, format, reference sets)
- Common error patterns for this semantic type
- Relationship to other columns

**Example**:
```
Column: "zip"
Sample values: ["10001", "9021", "10003", "10004"]
Semantic Profile: {
  "type": "US ZIP code",
  "format": "5-digit or ZIP+5",
  "constraints": ["numeric", "length 5 or 10 (with hyphen)"],
  "valid_range": "00501-99950",
  "common_errors": ["missing leading zero", "missing hyphen in ZIP+5"]
}
```

### 3.3 Phase 2: Constraint Program Synthesis

**The key contribution**: The LLM compiles the semantic profile into a unified, executable Python program.

The generated program includes:

```python
def validate_zip(value):
    """Validates US ZIP code format."""
    if pd.isna(value):
        return True  # NULL handled separately
    
    # Pattern validation
    pattern = r'^\d{5}(-\d{4})?$'
    if not re.match(pattern, str(value)):
        return False
    
    # Range validation
    zip_num = int(str(value)[:5])
    if not (501 <= zip_num <= 99950):
        return False
    
    return True

def repair_zip(value, column_stats):
    """Attempts to repair invalid ZIP codes.
    
    Repair synthesis challenge: The LLM must generate repairs that:
    1. Are semantically consistent with the detected error type
    2. Use domain knowledge (e.g., ZIP code structure)
    3. Leverage data statistics for context-aware repairs
    4. Handle multiple error patterns in priority order
    """
    if pd.isna(value):
        return None
    
    val_str = str(value)
    
    # Priority 1: Missing leading zero (4 digits → 5 digits)
    if len(val_str) == 4 and val_str.isdigit():
        return f"0{val_str}"
    
    # Priority 2: Missing hyphen in ZIP+9
    if len(val_str) == 9 and val_str.isdigit():
        return f"{val_str[:5]}-{val_str[5:]}"
    
    # Priority 3: Common typo patterns (e.g., 'O' instead of '0')
    cleaned = val_str.upper().replace('O', '0').replace('I', '1')
    if validate_zip(cleaned):
        return cleaned
    
    # Priority 4: Value similarity using column statistics
    return suggest_similar_valid(value, column_stats, validate_zip)

def detect_and_repair(column_values, column_stats):
    """Unified detection and repair execution."""
    results = []
    for idx, value in column_values.items():
        if not validate_zip(value):
            repaired = repair_zip(value, column_stats)
            results.append({
                'index': idx,
                'original': value,
                'repaired': repaired,
                'confidence': compute_repair_confidence(value, repaired, column_stats)
            })
    return results
```

**Key properties**:
- **Unified**: Single program contains both validation and repair logic
- **Executable**: Runs as native Python code without LLM dependency
- **Interpretable**: Human-readable validation logic
- **Composable**: Multiple constraint programs can be combined for multi-column validation
- **No runtime LLM dependency**: Once compiled, executes without LLM calls

**Repair Synthesis Methodology**:
Repair functions are synthesized through a structured prompting approach:

1. **Error Pattern Analysis**: The LLM analyzes common error patterns for the semantic type (e.g., leading zeros, typos, format variations)

2. **Repair Strategy Generation**: For each error pattern, the LLM generates:
   - Detection condition (when to apply this repair)
   - Transformation logic (how to repair)
   - Priority ordering (which repairs to try first)

3. **Context Integration**: The repair function receives `column_stats` (value distribution, frequent values) to enable data-aware repairs (e.g., using similarity to known valid values)

4. **Confidence Scoring**: Each repair is annotated with confidence computation based on:
   - Edit distance to original value
   - Semantic validity of result
   - Statistical consistency with column distribution

### 3.4 Phase 3: Detection and Repair Execution

Execute the synthesized programs:
1. **Detection**: Apply validation functions to each cell
2. **Repair**: For invalid cells, apply repair functions in priority order
3. **Confidence scoring**: Combine constraint violations with statistical outlier scores

**Execution characteristics**:
- O(cells) operations run as native Python (no LLM calls)
- Execution time comparable to statistical methods
- Parallelizable across columns

### 3.5 Security and Sandboxing Considerations

**Security Implications**: ProgramClean executes LLM-generated Python code, which introduces inherent security risks: arbitrary code execution, file system access, and potential data exfiltration. We address these through a defense-in-depth sandboxing strategy:

**Sandboxing Architecture**:
1. **Process Isolation**: Generated code executes in separate subprocesses with restricted privileges
2. **Resource Limits**: CPU time (30s max), memory (512MB max), and recursion depth limits
3. **Module Whitelist**: Only standard library modules (re, datetime, math) and pandas/numpy are permitted; network, file system, and OS access modules are blocked
4. **Restricted Built-ins**: `eval()`, `exec()`, `compile()`, `__import__()`, and file I/O operations are removed from the execution namespace
5. **Input Data Isolation**: Original datasets are never modified in-place; all operations work on copies

**Validation Before Execution**:
1. **Static Analysis**: AST parsing to detect prohibited operations before execution
2. **Pattern Matching**: Regex-based filtering for dangerous keywords (import, open, socket, etc.)
3. **Fallback Strategy**: If sandboxing fails, fall back to statistical methods (Raha/Baran)

**Security-Utility Trade-off**: These restrictions may prevent some legitimate but complex constraint programs (e.g., those requiring external API calls for validation). We accept this trade-off: constrained but safe execution is preferable to unrestricted but potentially dangerous pipelines, particularly for automated data processing workflows.

### 3.6 Failure Modes and Mitigation

**Failure Mode 1: Syntactically Valid but Semantically Incorrect Programs**

The LLM may generate code that passes Python syntax checks but encodes incorrect semantic constraints (e.g., wrong regex pattern for ZIP codes, incorrect range bounds).

*Example failure*: 
```python
# Generated code (incorrect)
def validate_email(value):
    pattern = r'^.*@.*$'  # Too permissive—allows invalid emails
    return re.match(pattern, str(value)) is not None
```

*Mitigation strategies*:
1. **Self-validation via sample execution**: Execute the validator on the sample values used for profiling; flag programs that reject >50% of samples
2. **Cross-validation with data statistics**: Compare constraint boundaries (min/max from program) against actual data statistics; flag mismatches
3. **Program testing on synthetic violations**: Generate synthetic violations (e.g., append "_invalid" to valid values) and verify the validator catches them
4. **Confidence scoring**: Lower confidence for constraints that don't align with observed data patterns

**Failure Mode 2: Overly Restrictive Constraints**

The LLM may generate validators that reject valid but unusual values (e.g., rejecting rare but legitimate domain values).

*Mitigation*:
- Statistical outlier detection alongside constraint validation
- Configurable strictness levels (strict vs. permissive validation modes)
- User review for flagged constraints before deployment

**Failure Mode 3: Incorrect Repair Functions**

Repairs may transform values into semantically invalid results (e.g., "90210" → "090210" by incorrectly adding a leading zero).

*Mitigation*:
1. **Repair validation**: Always validate repaired values through the detection function
2. **Edit distance limits**: Reject repairs with excessive transformation
3. **Confidence thresholds**: Only apply high-confidence repairs automatically; flag others for review

**Failure Mode 4: Execution Errors**

Generated code may raise runtime exceptions (e.g., type errors, undefined variables).

*Mitigation*:
1. **Sandboxed execution**: Run generated code in isolated environment with exception handling
2. **Static analysis**: Use Python AST parsing to detect undefined names before execution
3. **Fallback to statistical methods**: If program execution fails, fall back to Raha/Baran

## 4. Experimental Plan

### 4.1 Research Questions

1. **RQ1 (Accuracy)**: Does ProgramClean achieve detection/repair accuracy comparable to iterative LLM methods (Cocoon, IterClean)?

2. **RQ2 (Efficiency)**: Does ProgramClean achieve runtime performance comparable to statistical methods (Raha/Baran)?

3. **RQ3 (Zero-shot Generalization)**: Can ProgramClean handle novel semantic types not seen in training corpora (unlike Auto-Test)?

4. **RQ4 (Ablation - Program Synthesis Value)**: Does explicit program synthesis outperform direct LLM validation without compilation?

### 4.2 Datasets

**Standard Benchmarks**:
- Hospital (Rekatsinas et al., 2017): 1,000 rows, known error patterns
- Flights (Rekatsinas et al., 2017): Temporal data with FD violations
- Beers (Mahdavi et al., 2019): Product data with semantic errors
- Movies (Magellan Data): Real-world dirty data
- Rayyan (Ouzzani et al., 2016): Systematic review data

**Novel Domain Test**: Custom datasets with emerging semantic types:
- Cryptocurrency addresses (BTC, ETH)
- JWT tokens
- Modern TLD email addresses
- Unicode domain names

These test true zero-shot capability—types unlikely to appear in training corpora.

### 4.3 Baselines

1. **Cocoon**: Iterative LLM-based cleaning (accuracy benchmark)
2. **IterClean**: Iterative detector-verifier-repairer with LLMs (ACM-TURC 2024)
3. **Auto-Test**: Learned SDCs (where applicable - requires corpus)
4. **Raha + Baran**: Statistical error detection and correction
5. **GPT-4 Direct**: Cell-by-cell LLM classification (costly baseline)
6. **Akella et al. approach**: Statistical inlier detection + LLM rule generation with RAG
7. **SEED**: LLM-as-compiler for data curation (adapted for cleaning)

### 4.4 Ablation Study: Direct Validation vs. Program Synthesis

To isolate the value of the compilation step, we compare:

| Variant | Description |
|---------|-------------|
| **ProgramClean (full)** | LLM compiles semantic profile → executable program → native execution |
| **Direct Validation** | LLM directly validates each cell value via prompting (no program synthesis) |
| **Naive Code Gen** | LLM generates code without semantic profiling step (direct code generation from samples) |

**Expected outcome**: ProgramClean matches Direct Validation accuracy but is 100x+ faster; Naive Code Gen has lower accuracy due to lack of structured semantic understanding.

### 4.5 Metrics

- **Precision, Recall, F1**: Standard error detection metrics
- **Repair Accuracy**: Percentage of repairs that match ground truth
- **Runtime**: End-to-end execution time
- **LLM API calls**: Number of LLM invocations (our approach: O(columns); baselines: O(cells) or O(batches))
- **Cost**: Estimated API cost for cloud LLMs
- **Program validity**: Percentage of synthesized programs that execute without errors
- **Semantic correctness**: Manual audit of 50 programs for semantic correctness

### 4.6 Expected Results

**Expected Findings**:
- **Accuracy (RQ1)**: Within 5-10% F1 of Cocoon/IterClean; significantly better than Raha/Baran on semantic errors
- **Speed (RQ2)**: 100-1000x faster than Cocoon/IterClean, comparable to Raha/Baran
- **Zero-shot (RQ3)**: Successfully handles novel semantic types that Auto-Test cannot (lacks corpus)
- **Ablation (RQ4)**: ProgramClean matches Direct Validation accuracy at fraction of cost; validates compilation value

### 4.7 Implementation Details

**LLM Backend**: Open-source models (Llama 3.1-8B, CodeLlama) running locally via vLLM to avoid API costs.

**Computational Resources**:
- CPU: 2 cores (primary execution)
- RAM: 128GB available
- No GPU required (LLM inference done once per column, can use CPU offloading)

**Time Budget Justification**:
Given the 8-hour CPU-only constraint, we scope experiments as follows:
- Use smaller open-source models (Llama 3.1-8B, CodeLlama-7B) for faster CPU inference
- Limit evaluation to 5 standard benchmarks (~50 columns total)
- Novel domain tests limited to 10 semantic types
- Ablation studies run on subset (Hospital + Beers) only
- Estimated total: 6-7 hours for full experimental suite, leaving buffer for debugging

**Runtime Budget**: 
- Profiling + Synthesis: ~30 seconds per column (one-time)
- Detection: ~1 second per 10K rows (scalable execution)

## 5. Success Criteria

### 5.1 Confirmation Criteria

The hypothesis is **confirmed** if:
1. ProgramClean achieves F1 score within 10% of Cocoon on standard benchmarks
2. ProgramClean runs 100x faster than per-cell or per-batch LLM methods
3. ProgramClean successfully handles semantic types not present in Auto-Test's training corpus
4. Synthesized programs execute without errors on >90% of test columns
5. Ablation shows ProgramClean matches or exceeds Direct Validation accuracy

### 5.2 Refutation Criteria

The hypothesis is **refuted** if:
1. Synthesized constraint programs achieve <70% precision on standard benchmarks
2. LLM-synthesized programs have execution errors in >10% of cases
3. Zero-shot program synthesis fails to capture domain semantics (repair accuracy <50%)
4. Semantic bugs (correct syntax, wrong logic) occur in >20% of programs and escape detection

## 6. Significance and Contributions

### 6.1 Key Contributions

1. **LLM-as-Compiler for Data Cleaning**: Application of the LLM-as-compiler paradigm specifically for constraint-based data cleaning, extending the SEED framework's general curation approach to unified constraint program synthesis

2. **Zero-Shot Unified Program Synthesis**: First approach to synthesize unified constraint programs (not individual validators) truly zero-shot—without labeled data, training corpora, RAG, or few-shot examples

3. **Coupled Detection-Repair Synthesis**: Unified generation of validation and repair functions from semantic profiles, addressing a gap in prior work that focuses on detection only

4. **Security-Aware Execution**: Systematic sandboxing strategy for executing LLM-generated constraint code, addressing an overlooked security dimension in prior LLM-for-data-cleaning work

5. **Failure Mode Analysis**: Systematic characterization of failure modes for LLM-synthesized constraint programs with mitigation strategies

### 6.2 Impact

ProgramClean bridges the gap between statistical and LLM-based cleaning:
- **For practitioners**: Get LLM-level accuracy at statistical-method speeds without labeled data
- **For researchers**: Demonstrates application of the LLM-as-compiler paradigm to data quality constraints

### 6.3 Distinction from Prior Work

| System | Approach | Requires Corpus | Requires Labels | External Knowledge | LLM Usage | Program Structure | Runtime Cost |
|--------|----------|-----------------|-----------------|-------------------|-----------|-------------------|--------------|
| Auto-Test | Statistical learning | Yes | No | No | None | N/A | Low |
| Cocoon | Iterative LLM detection | No | No | No | Per-batch | N/A | High |
| IterClean | Iterative LLM cleaning | No | Yes (few) | No | Per-batch | N/A | High |
| Akella et al. | RAG + few-shot rules | No | No | RAG required | Per-rule | Individual validators | Medium |
| SEED | LLM-as-compiler (general) | No | No | User description | Pipeline compilation | Multiple modules | Variable |
| **ProgramClean** | **LLM-as-compiler (cleaning)** | **No** | **No** | **None** | **Per-column** | **Unified programs** | **Low** |

## 7. Limitations and Future Work

### 7.1 Limitations

1. **Pre-trained knowledge dependence**: Our "zero-shot" approach relies on LLM pre-training—semantic types not well-represented in training data may produce incorrect constraints
2. **Sandboxing overhead**: Security constraints may limit legitimate constraint complexity
3. **Single-column focus**: Current design focuses on single-column constraints; multi-column dependencies (functional dependencies) require extension
4. **CPU inference limits**: Without GPU, large-scale evaluation with state-of-the-art models is impractical

### 7.2 Future Directions

1. **Multi-column constraint synthesis**: Extend to functional dependencies and cross-column validation
2. **Human-in-the-loop refinement**: Interactive constraint editing and validation
3. **Learned repair confidence**: Train confidence estimators on repair outcomes
4. **Integration with data lakes**: Apply to data discovery and cataloging scenarios

## 8. References

1. He, Y., Wong, R. C. W., Cui, W., Ge, S., Zhang, H., Zhang, D., & Chaudhuri, S. (2025). Auto-Test: Learning Semantic-Domain Constraints for Unsupervised Error Detection in Tables. SIGMOD 2025.

2. Zhang, S., Huang, Z., Guo, J., Deng, D., & Wu, E. (2024). Data Cleaning Using Large Language Models. arXiv:2410.15547.

3. Mahdavi, M., Abedjan, Z., Castro Fernandez, R., Madden, S., Ouzzani, M., Stonebraker, M., & Tang, N. (2019). Raha: A Configuration-Free Error Detection System. SIGMOD 2019.

4. Mahdavi, M., & Abedjan, Z. (2020). Baran: Effective Error Correction via a Unified Context Representation and Transfer Learning. VLDB 2020.

5. Rekatsinas, T., Chu, X., Ilyas, I. F., & Ré, C. (2017). HoloClean: Holistic Data Repairs with Probabilistic Inference. VLDB 2017.

6. Akella, A., Kaul, A., Narayanam, K., & Mehta, S. (2025). Quality Assessment of Tabular Data using Large Language Models and Code Generation. EMNLP 2025 Industry Track.

7. Ni, W., Zhang, K., Miao, X., Zhao, X., Wu, Y., & Yin, J. (2024). IterClean: An Iterative Data Cleaning Framework with Large Language Models. ACM-TURC 2024.

8. Cui, W., Cao, L., Madden, S., Kraska, T., Shang, Z., Fan, J., Tang, N., Gu, Z., Liu, C., & Cafarella, M. (2024). Domain-Specific Data Curation With Large Language Models. arXiv:2310.00749.

9. Qi, D., & Wang, J. (2024). CleanAgent: Automating Data Standardization with LLM-based Agents. arXiv:2403.08291.

10. Naem, Z. A., Ahmad, M. S., Eltabakh, M., Ouzzani, M., & Tang, N. (2024). RetClean: Retrieval-Based Data Cleaning Using LLMs and Data Lakes. VLDB 2024.

11. Kabir, A., Wang, S., Tian, Y., Chen, T., Asaduzzaman, M., & Zhang, W. (2024). ZS4C: Zero-Shot Synthesis of Compilable Code for Incomplete Code Snippets Using LLMs. ACM Transactions on Software Engineering and Methodology.

12. Ouzzani, M., Hammady, H., Fedorowicz, Z., & Elmagarmid, A. (2016). Rayyan: A Systematic Reviews. Systematic Reviews, 5(1), 210.
