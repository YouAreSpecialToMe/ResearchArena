# LayerSelect: Attention-Aware Multi-Objective Hub Selection for Cross-Layer KV Cache Sharing

## Introduction

### Problem Statement

Large Language Models (LLMs) with long context capabilities face a critical memory bottleneck: the KV cache grows linearly with sequence length, number of layers, and batch size. Cross-layer KV cache sharing—where multiple transformer layers share a single set of key-value activations—has emerged as a promising direction for reducing memory consumption. However, existing approaches face fundamental limitations:

1. **Fixed patterns (CLA, YOCO)**: Uniformly grouping adjacent layers ignores the heterogeneous information needs across different layers.

2. **Similarity-based merging (MiniCache, xKV)**: These methods assume that layers with similar KV cache representations should be merged. MiniCache uses spherical linear interpolation based on cosine similarity, while xKV uses CKA to identify layers with aligned singular vectors for SVD-based compression. However, recent work by Yang et al. (2024) in KVSharer discovered a counterintuitive phenomenon: sharing *dissimilar* KV caches can sometimes better preserve model performance.

3. **Index sharing (HShare)**: Shares critical token indices rather than KV values, which doesn't reduce the KV cache memory footprint directly.

These observations reveal a critical gap: current methods lack a principled way to balance coverage (ensuring spoke layers get the information they need) with diversity (avoiding redundant hub selection) while considering how queries actually attend to KV caches.

### Key Insight and Hypothesis

Our key insight is that **optimal cross-layer KV sharing should be determined by how query layers actually use information from KV caches**, not just by the similarity of KV cache representations. We frame hub layer selection as a **multi-objective optimization problem** that simultaneously:

1. **Maximizes coverage**: Each spoke layer should have access to KV caches that its queries strongly attend to
2. **Maximizes hub diversity**: Selected hub layers should be informationally diverse to avoid redundancy (addressing KVSharer's finding)
3. **Minimizes attention misalignment**: The routing of spoke layers to hub layers should preserve attention patterns

We hypothesize that this attention-aware, multi-objective approach will achieve better memory-accuracy tradeoffs than similarity-based methods (xKV, MiniCache), uniform grouping (CLA), or greedy selection (HShare).

### Contributions

1. **Query-KV Dependency Graph**: A novel bipartite graph representation that captures how queries at each layer attend to KV caches, enabling data-driven hub selection based on actual usage patterns rather than just KV cache similarity.

2. **Multi-Objective Hub Selection**: A principled optimization formulation that balances coverage, diversity, and attention alignment—addressing the tension between similarity-based and dissimilarity-based sharing discovered by KVSharer.

3. **Attention-Weighted Dynamic Routing**: A mechanism for spoke layers to dynamically aggregate information from multiple hub layers based on attention scores, rather than simply copying or interpolating KV values.

## Proposed Approach

### Overview

LayerSelect operates in three phases:

1. **Profiling Phase** (~10 minutes): Build a Query-KV Dependency Graph by profiling the model on calibration prompts
2. **Optimization Phase**: Solve a multi-objective optimization problem to select optimal hub layers
3. **Deployment Phase**: Use attention-weighted dynamic routing during inference

### Phase 1: Query-KV Dependency Graph Construction

Unlike xKV which uses CKA to measure KV cache similarity, we construct a **bipartite dependency graph** between query layers and KV cache layers:

**Nodes**: 
- Query layer nodes $Q = \{q_1, q_2, ..., q_L\}$ (one per transformer layer)
- KV cache layer nodes $K = \{k_1, k_2, ..., k_L\}$ (one per transformer layer)

**Edges**: For each edge $(q_i, k_j)$, we compute the **cross-layer attention dependency** $A_{ij}$:

$$A_{ij} = \frac{1}{N \cdot H} \sum_{n=1}^{N} \sum_{h=1}^{H} \text{Attn}_{i,h}^{(n)}[j]$$

where $\text{Attn}_{i,h}^{(n)}[j]$ is the average attention weight that query layer $i$, head $h$, token $n$ places on tokens from KV cache $j$ (computed by substituting KV cache $j$ for KV cache $i$ during the calibration phase).

This captures how much query layer $i$ would "prefer" to use KV cache $j$ if given the choice—providing a usage-based measure rather than just representation similarity.

### Phase 2: Multi-Objective Hub Selection

Given the dependency matrix $A \in \mathbb{R}^{L \times L}$, we formulate hub selection as a **multi-objective optimization**:

**Decision Variables**: Binary vector $s \in \{0,1\}^L$ where $s_j = 1$ if layer $j$ is selected as a hub.

**Objective 1 - Coverage**: Maximize the fraction of dependency weights covered by selected hubs:
$$f_{\text{coverage}}(s) = \frac{\sum_{i=1}^{L} \max_{j: s_j=1} A_{ij}}{\sum_{i,j} A_{ij}}$$

**Objective 2 - Diversity**: Maximize the dissimilarity between selected hub layers (preventing redundancy):
$$f_{\text{diversity}}(s) = \frac{1}{(\sum s_j)^2} \sum_{j_1, j_2: s_{j_1}=s_{j_2}=1} (1 - \text{CKA}(k_{j_1}, k_{j_2}))$$

This explicitly incorporates KVSharer's finding that dissimilar sharing can be beneficial.

**Objective 3 - Attention Alignment**: Minimize attention distortion when routing spoke layers to hubs:
$$f_{\text{alignment}}(s) = -\sum_{i=1}^{L} \sum_{j: s_j=1} r_{ij} \cdot \text{KL}(\text{Attn}_i || \text{Attn}_{i \rightarrow j})$$

where $r_{ij}$ indicates that spoke layer $i$ is routed to hub $j$, and $\text{Attn}_{i \rightarrow j}$ is the attention pattern when layer $i$ uses KV cache from layer $j$.

**Constraint**: Budget on number of hubs: $\sum_j s_j \leq B$

We solve this using **multi-objective evolutionary algorithms** (NSGA-II) to obtain a Pareto frontier of solutions, from which users can select based on their desired compression-accuracy tradeoff.

### Phase 3: Attention-Weighted Dynamic Routing

Once hubs are selected, we need a routing mechanism for spoke layers to access hub KV caches. Unlike:
- **CLA**: Simply copies KV from the bottom layer of each group
- **xKV**: Uses SVD to compress multiple layers into a shared subspace
- **MiniCache**: Interpolates based on cosine similarity

We propose **attention-weighted dynamic routing**:

For spoke layer $i$ routed to hub set $\mathcal{H}_i \subseteq \{j: s_j=1\}$:

$$\text{KV}_i^{\text{eff}} = \sum_{j \in \mathcal{H}_i} w_{ij} \cdot \text{KV}_j$$

where weights $w_{ij}$ are computed based on the dependency scores:

$$w_{ij} = \frac{\exp(A_{ij} / \tau)}{\sum_{j' \in \mathcal{H}_i} \exp(A_{ij'} / \tau)}$$

with temperature $\tau$ controlling the sharpness of the routing.

This allows spoke layers to dynamically blend information from multiple hubs based on their actual attention patterns.

## Related Work

### Cross-Layer KV Cache Sharing

**Cross-Layer Attention (CLA)** [Brandon et al., NeurIPS 2024]: Introduces architectural modifications to share KV caches uniformly across adjacent layer groups. Requires model retraining from scratch. Our method is training-free and uses data-driven selection rather than fixed patterns.

**YOCO** [Sun et al., 2024]: Uses a decoder-decoder architecture where the top-half layers reuse KV caches from a single global cache. Requires architectural changes and retraining. Our method works with existing pretrained models.

### Training-Free Cross-Layer Methods

**MiniCache** [Liu et al., 2024]: Observes high angular similarity in KV caches of middle-to-deep layers and merges adjacent layers using SLERP based on cosine similarity. Limited to merging adjacent layer pairs with similar KV caches.

**xKV** [Chang et al., arXiv 2025]: Uses CKA to identify layers with aligned singular vectors and applies cross-layer SVD compression. While effective, xKV focuses solely on KV cache similarity through CKA/SVD, not on how queries actually use the KV caches. Our Query-KV Dependency Graph and attention-weighted routing provide a fundamentally different approach based on usage patterns rather than just representation similarity.

**KVSharer** [Yang et al., arXiv 2024]: Discovers that sharing *dissimilar* KV caches can better preserve performance. Uses a simple search strategy to find sharing pairs. Our multi-objective optimization explicitly incorporates diversity as an objective, providing a principled way to leverage this counterintuitive finding.

**HShare** [Wu et al., ICLR 2025]: Proposes hierarchical sharing of critical token indices (which tokens to keep) across layers, heads, and queries using a greedy algorithm. This is complementary to our approach: HShare reduces which tokens are stored, while we reduce which layers store unique KV caches. Our multi-objective optimization provides a more principled selection mechanism than greedy search.

### Key Distinctions

| Method | Selection Strategy | Sharing Mechanism | Training Required |
|--------|-------------------|-------------------|-------------------|
| CLA | Fixed uniform grouping | Direct KV copy | Yes |
| MiniCache | Cosine similarity of adjacent layers | SLERP interpolation | No |
| xKV | CKA + SVD on singular vectors | Cross-layer SVD compression | No |
| KVSharer | Search for dissimilar pairs | Direct KV copy | No |
| HShare | Greedy similarity-based | Critical token index sharing | No |
| **LayerSelect (Ours)** | **Multi-objective optimization** | **Attention-weighted dynamic routing** | **No** |

Our key differentiators:
1. **Usage-based vs. representation-based**: We use Query-KV attention dependency rather than just KV-KV similarity (xKV, MiniCache)
2. **Multi-objective vs. single-objective**: We balance coverage and diversity explicitly, addressing KVSharer's finding
3. **Dynamic routing vs. static sharing**: Attention-weighted aggregation vs. direct copy (CLA, KVSharer) or SVD compression (xKV)

## Experiments

### Experimental Setup

**Models**: 
- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct
- Mistral-7B-v0.3

**Baselines**:
- Full KV cache (no compression)
- CLA-2 (2 layers share 1 KV cache, uniformly grouped)
- MiniCache (adjacent layer merging with SLERP)
- xKV (cross-layer SVD compression)
- KVSharer (dissimilar layer sharing via search)

**Benchmarks**:
- RULER (long-context benchmark): 4K, 8K, 16K sequence lengths
- LongBench: long-document QA, summarization
- Perplexity: WikiText-2, C4

**Metrics**:
- Accuracy (task-specific metrics)
- KV cache memory usage (GB)
- Compression ratio (relative to full cache)
- Inference throughput (tokens/sec)

### Expected Results

**Hypothesis 1**: LayerSelect will achieve better accuracy-compression tradeoffs than similarity-based methods (xKV, MiniCache) by considering actual usage patterns.

Expected: At 2× compression, LayerSelect achieves <1% accuracy drop vs. 2-3% for MiniCache and xKV on RULER tasks.

**Hypothesis 2**: The multi-objective formulation will outperform greedy selection (HShare) and simple search (KVSharer).

Expected: LayerSelect achieves higher diversity scores while maintaining comparable coverage, leading to better generalization across tasks.

**Hypothesis 3**: Attention-weighted routing will reduce attention distortion compared to direct KV sharing (CLA, KVSharer).

Expected: KL divergence between original and compressed attention patterns is 30-50% lower for LayerSelect vs. CLA.

### Ablation Studies

1. **Component ablation**: Compare full method vs. coverage-only vs. diversity-only vs. alignment-only
2. **Graph construction**: Compare Query-KV dependency vs. KV-KV similarity (xKV-style)
3. **Routing mechanisms**: Compare attention-weighted vs. direct copy vs. SVD reconstruction
4. **Budget sensitivity**: Performance across different hub budgets (compression ratios)

## Success Criteria

### Confirming Evidence

The hypothesis would be confirmed if:

1. **Accuracy**: LayerSelect achieves ≥2× compression with <2% accuracy drop on long-context benchmarks, outperforming baselines at the same compression ratio.

2. **Pareto dominance**: LayerSelect dominates baselines on the accuracy-compression Pareto frontier.

3. **Component validation**: Ablation studies show each objective (coverage, diversity, alignment) contributes meaningfully to final performance.

4. **Attention preservation**: Lower attention distribution divergence compared to direct sharing methods.

### Refuting Evidence

The hypothesis would be refuted if:

1. Query-KV dependency provides no additional benefit over KV-KV similarity (xKV approach is sufficient)
2. Multi-objective optimization does not outperform simple greedy selection
3. Attention-weighted routing introduces unacceptable latency overhead
4. The method fails to generalize across different model architectures

## Limitations and Future Work

1. **Computational overhead**: The profiling and optimization phases add ~10-15 minutes of one-time cost. Future work could explore amortizing this across multiple deployments.

2. **Calibration sensitivity**: Performance depends on the representativeness of calibration prompts. Future work could investigate robust calibration selection.

3. **Dynamic contexts**: Current method uses static hub selection. Future work could explore adaptive hub selection based on input characteristics.

## References

1. Brandon, W., Mishra, M., Nrusimha, A., Panda, R., & Ragan-Kelley, J. (2024). Reducing Transformer Key-Value Cache Size with Cross-Layer Attention. *NeurIPS 2024*.

2. Chang, C. C., Lin, C. Y., Akhauri, Y., Lin, W. C., Wu, K. C., Ceze, L., & Abdelfattah, M. S. (2025). xKV: Cross-Layer SVD for KV-Cache Compression. *arXiv:2503.18893*.

3. Liu, A., Liu, J., Pan, Z., He, Y., Haffari, G., & Zhuang, B. (2024). MiniCache: KV Cache Compression in Depth Dimension for Large Language Models. *arXiv:2405.14366*.

4. Sun, Z., Li, Y., Gan, Z., Liu, Z., Wang, Z., Li, C., ... & Wang, J. (2024). You Only Cache Once: Decoder-Decoder Architectures for Language Models. *arXiv:2405.05254*.

5. Wu, H., Li, L., Huang, H., Tu, Y., Zhang, J., Yu, M., & Yan, J. (2025). HShare: Fast LLM Decoding by Hierarchical Key-Value Sharing. *ICLR 2025*.

6. Yang, Y., Cao, Z., Chen, Q., Qin, L., Yang, D., Zhao, H., & Chen, Z. (2024). KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing. *arXiv:2410.18517*.

7. Zhang, P., Zeng, G., Li, T., & Lu, F. (2024). TinyLLaVA: A Framework of Small-scale Large Multimodal Models. *arXiv preprint*.

8. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *ICML 2019*.
