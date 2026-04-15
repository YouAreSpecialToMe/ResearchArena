# Entropy-Guided Adaptive Token Merging for Robust and Efficient Vision Transformers

## Introduction

### Context
Vision Transformers (ViTs) have become the dominant architecture for image recognition, achieving state-of-the-art results across classification, detection, and segmentation tasks. However, their quadratic attention complexity makes inference expensive, motivating a large body of work on *token reduction* — methods that prune or merge tokens during the forward pass to reduce computation while maintaining accuracy. Recent surveys (Kong et al., 2025) argue that token reduction should transcend its traditional efficiency-oriented role, serving broader purposes including robustness and alignment — a perspective our work directly instantiates.

Token Merging (ToMe) (Bolya et al., 2023) is the most widely adopted approach: it uses bipartite soft matching on key cosine similarity to gradually merge similar tokens across layers, achieving 2x+ throughput with minimal accuracy loss on clean ImageNet. Other methods include EViT (Liang et al., 2022), DynamicViT (Rao et al., 2021), A-ViT (Yin et al., 2022), and more recent image-adaptive approaches like AToM (Shin et al., 2025) and AdapTiV (Yoo et al., 2024).

### Problem Statement
Despite their widespread use, **token reduction methods have never been systematically evaluated under distribution shift** — the very conditions that deployed vision systems must handle. Real-world inputs regularly contain corruptions (noise, blur, weather artifacts), domain shifts (sketch, art, new sensor), and other perturbations. We hypothesize that token merging methods suffer *disproportionate* accuracy degradation under these conditions, grounded in three mechanistic arguments:

1. **Similarity metric degradation**: ToMe merges tokens whose keys are cosine-similar. Input perturbations propagate through the key projection as $W_K \cdot \delta$, distorting the cosine similarity structure used for bipartite matching. Under corruption, tokens that should remain separate may appear similar (and vice versa), leading to *incorrect merging decisions*. This effect is amplified in deeper layers where corrupted representations have compounded. Empirical evidence supports this concern: Guo et al. (ICCV 2023) showed that under corruption, ViT attention patterns become "highly diverging" — the attention similarity structure that token merging relies on is precisely what degrades.
2. **Token overfocusing amplification**: Guo et al. (ICCV 2023) demonstrated that ViTs under corruption suffer from "token overfocusing" — attention concentrates on a few tokens that are not robust, with cosine similarity between attention vectors increasing pathologically. Token merging can amplify this by eliminating the diverse tokens that would otherwise provide redundancy and robustness.
3. **No shift-aware mechanism**: Current methods — including recent adaptive approaches like AToM and AdapTiV that adapt merging for efficiency, and SATA (Nikzad et al., CVPR 2025) which improves robustness via spatial autocorrelation grouping — do not adapt the *amount* of merging based on input degradation. They use strategies agnostic to distribution shift severity.

### Key Insight
**The attention entropy of a ViT layer is an input-dependent signal that correlates with distribution shift severity.** Under clean inputs, attention distributions are sharp (low entropy) — the model is confident about which tokens matter. Under corrupted or shifted inputs, attention becomes more diffuse (high entropy) — the model is uncertain. This connection is well-grounded: Guo et al. (ICCV 2023) showed that corruption causes attention to either overfocus (collapse) or diffuse, and Zhai et al. (ICML 2023) established that attention entropy is a meaningful diagnostic of transformer behavior. We exploit this signal to adapt the token merging ratio at inference time: merge aggressively when attention is confident, conservatively when it is uncertain.

### Hypothesis
An entropy-guided adaptive token merging strategy that adjusts the merging ratio based on per-layer attention entropy will maintain the efficiency gains of standard ToMe on clean inputs while significantly reducing accuracy degradation under distribution shift, without any additional training.

## Proposed Approach

### Overview
We propose **Entropy-Guided Token Merging (EGTM)**, a training-free, drop-in replacement for standard ToMe that makes token reduction robust to distribution shift. The core contribution is an **adaptive merging ratio** driven by attention entropy, which naturally reduces token merging under distribution shift where merging decisions become unreliable.

### Method Details

#### Core Component: Entropy-Adaptive Merging Ratio

**Step 1: Calibration.** Given a small set of clean images (e.g., 500 from ImageNet validation), we run forward passes through the ViT with standard ToMe and record per-layer attention entropy statistics. To avoid the overhead of materializing the full $N \times N$ attention matrix, we use a **CLS-row entropy proxy**: at each transformer layer $l$, we compute the entropy of the CLS token's attention distribution over all other tokens, averaged across heads:

$$H_l = -\frac{1}{N_h} \sum_{h=1}^{N_h} \sum_{j} A^{(h)}_{\text{CLS},j} \log A^{(h)}_{\text{CLS},j}$$

where $A^{(h)}_{\text{CLS},j}$ is the attention weight from the CLS token to token $j$ in head $h$. This requires only the CLS row of the attention matrix ($1 \times N$ per head), which is already computed during the forward pass, adding negligible overhead. We store the mean $\bar{H}_l$ and standard deviation $\sigma_l$ for each layer from the calibration set.

**Rationale for CLS-row proxy**: The CLS token aggregates global information and its attention distribution reflects overall input uncertainty. Under corruption, the CLS attention becomes more diffuse as the model struggles to identify which tokens carry reliable information. We validate this proxy against full-matrix entropy in our ablation study, comparing both the CLS-row variant and the full attention entropy variant.

**Step 2: Inference-time adaptation.** At inference, for each input and each layer, we compute $H_l$ from the CLS attention row and derive an *entropy deviation score*:

$$\delta_l = \max\left(0, \frac{H_l - \bar{H}_l}{\sigma_l}\right)$$

The adaptive merging ratio $r_l$ is then:

$$r_l = r_0 \cdot \max\left(\alpha, \exp(-\beta \cdot \delta_l)\right)$$

where $r_0$ is the base merging ratio (e.g., 8 tokens per layer as in standard ToMe), $\alpha \in (0,1)$ is a minimum ratio floor (ensuring some merging always occurs for efficiency), and $\beta > 0$ controls the sensitivity to entropy deviation. When attention entropy is normal ($\delta_l \approx 0$), the merging ratio is close to $r_0$ (standard efficiency). When entropy is elevated (distribution shift detected), the ratio decreases exponentially, preserving more tokens.

**Computational overhead analysis**: The CLS-row entropy computation adds only $O(N)$ per head per layer, where $N$ is the current token count. Since the attention scores are already computed in the forward pass, the additional cost is one entropy computation over a pre-existing softmax output — negligible compared to the $O(N^2 d)$ cost of the attention computation itself. In contrast, full attention matrix entropy would require $O(N^2)$ per head, potentially negating throughput gains at high token counts. We benchmark both variants in our efficiency analysis.

#### Optional Enhancement: Importance-Aware Token Protection
Before merging, we can additionally identify tokens with high cumulative attention (tokens that receive attention from many other tokens):

$$\text{importance}_i = \sum_j A_{ji}$$

Tokens in the top-$k$ by importance score are excluded from the merging candidate set. This preserves critical discriminative features even when the overall merging ratio is high. We treat this as an optional enhancement and ablate its contribution separately.

### Key Innovations
1. **First systematic study** of token reduction robustness under distribution shift — existing adaptive methods (AToM, AdapTiV) adapt for efficiency, not for robustness; SATA (Nikzad et al., CVPR 2025) improves ViT robustness through spatial analysis but does not address token merging degradation specifically
2. **Training-free, shift-aware mechanism** — uses CLS-row attention entropy as a lightweight proxy for input difficulty, unlike prior entropy-based ViT work (Lin et al., IJCV 2026) which uses transfer entropy for layer-wise condensation during training
3. **Input-adaptive computation** — naturally allocates more compute to harder (shifted) inputs
4. **Drop-in replacement** — same API as ToMe, applicable to DeiT, ViT, and other standard architectures

### Limitations and Failure Modes
We identify two scenarios where EGTM may fail or underperform:

1. **Overconfident wrong attention under corruption**: Some corruption types (e.g., extreme contrast changes, adversarial perturbations) may cause attention entropy to *decrease* rather than increase — the model becomes confidently wrong, concentrating attention on spurious features. In this case, EGTM would maintain aggressive merging, potentially worsening performance. We explicitly test for this in our per-corruption analysis and report which corruption types violate the entropy-increase assumption.

2. **Mild corruptions within calibration variance**: If a corruption's effect on attention entropy falls within the normal variance of clean data ($\delta_l \approx 0$), EGTM will not trigger adaptation. This is by design (avoiding false positives on natural image variation), but means EGTM provides no benefit for very mild corruptions where standard ToMe may already perform adequately.

3. **Calibration distribution mismatch**: If the deployment domain differs substantially from the calibration domain (e.g., calibrated on ImageNet natural images but deployed on medical imaging), the entropy statistics may be miscalibrated. The $z$-score normalization provides some robustness to this, but significant domain gaps would require re-calibration.

## Related Work

### Token Reduction for Vision Transformers
- **ToMe** (Bolya et al., ICLR 2023): Uses bipartite soft matching on key similarity to merge tokens. Training-free, widely adopted. Uses a fixed merging ratio; does not consider robustness.
- **EViT** (Liang et al., ICLR 2022): Keeps top-k attentive tokens, fuses the rest into a single token. The original method requires fine-tuning; we evaluate a training-free variant (labeled "EViT-style") that applies the same token selection criterion without fine-tuning, and discuss this limitation (see Experiments section).
- **DynamicViT** (Rao et al., NeurIPS 2021): Learns a prediction module for token importance. Requires training.
- **A-ViT** (Yin et al., CVPR 2022): Adaptive halting mechanism per token using ACT. Requires training.
- **Token Fusion** (Kim et al., WACV 2024): Bridges pruning and merging by switching strategy based on model linearity. Does not study robustness.

### Surveys on Token Reduction
- **Kong et al. (arXiv 2025)**: "Token Reduction Should Go Beyond Efficiency in Generative Models — From Vision, Language to Multimodality." Argues that token reduction transcends computational optimization, influencing model robustness, alignment, and training stability. Our work directly instantiates this vision by showing token reduction impacts robustness under distribution shift and proposing an entropy-guided solution.

### Image-Adaptive Token Merging
- **AToM** (Shin et al., IEEE Trans. Computers 2025): Proposes image-adaptive, fine-grained token merging with algorithm-architecture co-design. Focuses on *efficiency optimization* — adapting the merging pattern per image for better accuracy-throughput tradeoffs on clean data. Does not study behavior under distribution shift.
- **AdapTiV** (Yoo et al., IEEE/ACM MICRO 2024): Uses sign-similarity based image-adaptive token merging with hardware co-design. Also focuses on *efficiency* (reducing merging overhead, concealing latency within LayerNorm). Does not study robustness under corrupted or shifted inputs.

**Our difference from AToM/AdapTiV**: These methods adapt *which* tokens to merge for better clean-data efficiency. We adapt *how many* tokens to merge based on a shift-detection signal (attention entropy). The motivation is fundamentally different: they optimize throughput, we optimize robustness. The two approaches are complementary — EGTM's entropy-guided ratio could be combined with their adaptive matching strategies.

### Training-Free Robustness Methods for ViTs
- **SATA** (Nikzad et al., CVPR 2025): Enhances ViT robustness by analyzing spatial autocorrelation of tokens using Moran's I metric. Groups tokens by spatial dependency before the FFN block, improving robustness on ImageNet-C (mCE=13.6%) without retraining. SATA is a strong, directly comparable baseline: it is training-free, operates on tokens, and targets robustness. However, SATA addresses robustness by modifying the FFN computation via spatial grouping, while we address a different problem — the degradation of token *merging* decisions under shift. The two methods are complementary: SATA could be applied alongside EGTM.

### Entropy-Based ViT Methods
- **Entropy-Guided Condensing** (Lin et al., IJCV 2026): Uses transfer entropy to identify and remove uninformative *layers* (depth-wise condensation) via "Dilution Learning." This is a training-time method that reduces layers, not tokens, and uses a different notion of entropy (transfer entropy between layers, not attention entropy within a layer).
- **Zhai et al. (ICML 2023)**: Showed that attention entropy is a meaningful diagnostic of transformer behavior — pathologically low entropy (collapse) correlates with training instability. We use attention entropy in a complementary way: as an inference-time signal for distribution shift detection.

**Our difference**: We use per-layer CLS-row attention entropy at inference time to adapt the token merging *ratio* — a fundamentally different mechanism, purpose (robustness vs. efficiency/stability), and granularity (per-input adaptation vs. architecture modification).

### Robustness of Vision Transformers
- **Zhang et al. (CVPR 2022)**: Showed ViTs generalize better than CNNs under distribution shift due to shape bias, but did not study efficient ViTs.
- **Guo et al. (ICCV 2023)**: Identified "token overfocusing" in ViTs under corruption — attention concentrates on few non-robust tokens with highly diverging patterns. Proposed TAP+ADL to diversify attention during training. Does not address token reduction but provides key evidence that attention patterns are disrupted under corruption, supporting our entropy-based detection mechanism.
- **Mao et al. (CVPR 2022)**: Proposed methods towards robust vision transformers, studying adversarial robustness. Does not study token reduction.
- **RSPC** (Cheng et al., CVPR 2023): Reduces sensitivity to patch corruptions via training-time augmentation. Requires retraining.

**Our difference**: We study the intersection of efficiency and robustness — how token reduction degrades under shift and how to fix it without retraining.

### Test-Time Adaptation with Token Methods
- **TCA** (Wang et al., ICCV 2025): Token condensation for test-time adaptation in VLMs (CLIP). VLM-specific, different from our focus on standard ViTs and on making token reduction itself robust.

## Experiments

### Planned Setup

**Models** (all pretrained, no training required):
- DeiT-Small (22M params)
- DeiT-Base (86M params)

**Token Reduction Methods** (all baselines):
- No reduction (upper bound on accuracy)
- ToMe (Bolya et al., 2023) — fixed ratio r=8
- EViT-style (training-free variant) — top-k attentive token keeping applied without fine-tuning. **Note**: The original EViT (Liang et al., 2022) requires fine-tuning. We implement the token selection criterion (keep top-k by CLS attention, fuse rest) without fine-tuning to ensure a fair training-free comparison. We clearly label this as "EViT-style (training-free)" in all results and discuss the gap vs. fine-tuned EViT as a limitation.
- SATA (Nikzad et al., CVPR 2025) — spatial autocorrelation token analysis (training-free robustness baseline)
- Random token dropping — lower bound on token reduction
- Our EGTM — entropy-guided adaptive

**Evaluation Benchmarks**:
- *Clean*: ImageNet-1K validation set (50K images)
- *Common corruptions*: ImageNet-C (15 corruption types x 5 severity levels)
- *Natural distribution shift*: ImageNet-R (renditions)

**Metrics**:
- Top-1 Accuracy (clean and per-corruption)
- mean Corruption Error (mCE) — standard robustness metric
- Relative Accuracy Drop: $(Acc_{clean} - Acc_{corrupt}) / Acc_{clean}$ — measures disproportionate degradation from token reduction
- Throughput (images/sec on A6000 GPU) — efficiency metric
- Entropy overhead: wall-clock time for entropy computation vs. total inference time

### Planned Experiments

**Experiment 1: Systematic Robustness Benchmark** (~3 hours)
- Apply each token reduction method to DeiT-S and DeiT-B
- Evaluate on ImageNet val (clean), ImageNet-C (severity 3 and 5 for all 15 corruptions), ImageNet-R
- Report accuracy, mCE, relative accuracy drop, throughput
- Key question: Does token merging degrade disproportionately under shift?
- 1 seed for baselines, 3 seeds for EGTM (varying calibration subsets)

**Experiment 2: Per-Corruption Analysis** (~included in Exp 1)
- Break down results by corruption type (noise, blur, weather, digital)
- Identify which corruption types cause the largest degradation for token merging
- Analyze the relationship between corruption severity and merging degradation
- **Critical**: Identify corruptions where entropy *decreases* (overconfident wrong attention), testing the failure mode discussed above

**Experiment 3: Ablation Study** (~1.5 hours)
- EGTM with CLS-row entropy (default) vs. full attention matrix entropy
- EGTM with adaptive ratio only (core) vs. adaptive ratio + importance protection
- Sensitivity to hyperparameters: $\alpha \in \{0.1, 0.3, 0.5\}$, $\beta \in \{0.5, 1.0, 2.0\}$
- DeiT-S on ImageNet-C (severity 5, 3 corruption types: Gaussian noise, motion blur, snow)

**Experiment 4: Efficiency Analysis** (~0.5 hours)
- Measure actual throughput (images/sec) for each method, including EGTM entropy overhead
- Compare CLS-row entropy vs. full-matrix entropy overhead explicitly
- Plot accuracy-throughput Pareto curves for clean and corrupted settings
- Show EGTM achieves better Pareto frontier under shift

**Experiment 5: Analysis and Visualization** (~0.5 hours)
- Visualize CLS attention entropy distributions for clean vs. corrupted inputs
- Show how EGTM adapts its merging ratio across layers for different inputs
- Compute Spearman correlation between entropy and corruption severity
- Report per-corruption entropy direction (increase vs. decrease) to validate the entropy-increase assumption

### Runtime Budget (~6.5 hours total)
| Component | Time |
|-----------|------|
| Setup + calibration | 0.5h |
| Exp 1: Main benchmark (2 models x 6 methods x ImageNet-val + ImageNet-C + ImageNet-R) | 3.5h |
| Exp 3: Ablations (DeiT-S, entropy variants, 3 corruptions, hyperparameter sweep) | 1.5h |
| Exp 4-5: Efficiency + analysis | 1.0h |
| **Total** | **6.5h** |

### Expected Results
1. Standard ToMe will show measurably larger relative accuracy drop on corrupted images compared to no-reduction baseline (hypothesis: 2-5% gap)
2. EGTM will recover most of this degradation while maintaining >80% of ToMe's throughput gains on clean images
3. The adaptive ratio will naturally merge fewer tokens on corrupted inputs (verifiable by measuring average reduction rate)
4. CLS-row entropy will be a sufficient proxy for full-matrix entropy with <5% of the computational overhead
5. Ablations will show that the adaptive ratio is the primary contributor; importance protection provides additional but smaller gains
6. Some corruption types (particularly contrast-related) may violate the entropy-increase assumption — we report these transparently

### Risk Mitigation
If the disproportionate degradation from ToMe is <1% (i.e., the core hypothesis doesn't hold), we pivot to reporting this as a negative/null result with a thorough empirical study of token reduction robustness — still a valuable contribution as no prior work has characterized this. The benchmark itself (systematic evaluation of token reduction under shift) has standalone value. If attention entropy does not reliably increase under corruption, we explore alternative shift-detection signals (e.g., key similarity variance, token norm statistics).

## Success Criteria

### Confirms hypothesis if:
- Token merging (ToMe) shows statistically significant *disproportionate* accuracy degradation under distribution shift compared to uncompressed ViTs (relative accuracy drop gap > 2%)
- EGTM reduces this disproportionate degradation by >50% compared to standard ToMe on ImageNet-C
- EGTM maintains >80% of ToMe's throughput improvement on clean images
- Results are consistent across both DeiT-S and DeiT-B
- Attention entropy correlates positively with corruption severity (Spearman rho > 0.5) for the majority (>10/15) of corruption types

### Refutes hypothesis if:
- Token merging does NOT degrade disproportionately under shift (i.e., the accuracy-efficiency tradeoff is the same for clean and corrupted inputs)
- Attention entropy does NOT correlate with distribution shift severity for the majority of corruption types
- Adaptive merging ratio provides no benefit over fixed ratio under shift

## References

1. Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, Judy Hoffman. "Token Merging: Your ViT But Faster." ICLR 2023.
2. Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Herve Jegou. "Training data-efficient image transformers & distillation through attention." ICML 2021.
3. Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, Pengtao Xie. "Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations." ICLR 2022.
4. Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, Cho-Jui Hsieh. "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification." NeurIPS 2021.
5. Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz, Pavlo Molchanov. "A-ViT: Adaptive Tokens for Efficient Vision Transformer." CVPR 2022.
6. Dan Hendrycks, Thomas Dietterich. "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations." ICLR 2019.
7. Chongzhi Zhang, Mingyuan Zhang, Shanghang Zhang, Daisheng Jin, Qiang Zhou, Zhongang Cai, Haiyu Zhao, Xianglong Liu, Ziwei Liu. "Delving Deep into the Generalization of Vision Transformers under Distribution Shifts." CVPR 2022.
8. Xiaofeng Mao, Gege Qi, Yuefeng Chen, Xiaodan Li, Ranjie Duan, Shaokai Ye, Yuan He, Hui Xue. "Towards Robust Vision Transformer." CVPR 2022.
9. Yong Guo, David Stutz, Bernt Schiele. "Robustifying Token Attention for Vision Transformers." ICCV 2023.
10. Minchul Kim, Shangqian Gao, Yen-Chang Hsu, Yilin Shen, Hongxia Jin. "Token Fusion: Bridging the Gap between Token Pruning and Token Merging." WACV 2024.
11. Zixin Wang, Dong Gong, Sen Wang, Zi Huang, Yadan Luo. "Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation." ICCV 2025.
12. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
13. Mingjie Cheng, Xinwen Hou, Chunhui Hao. "Improving Robustness of Vision Transformers by Reducing Sensitivity to Patch Corruptions." CVPR 2023.
14. J. Shin, M. Kang, Y. Han, J. Park, L.-S. Kim. "AToM: Adaptive Token Merging for Efficient Acceleration of Vision Transformer." IEEE Transactions on Computers, 2025.
15. Seungjae Yoo, Hangyeol Kim, Joo-Young Kim. "AdapTiV: Sign-Similarity Based Image-Adaptive Token Merging for Vision Transformer Acceleration." IEEE/ACM MICRO 2024.
16. Shangqun Lin, Peng Lyu, Dongliang Liu et al. "Entropy-Guided Condensing for Vision Transformer." International Journal of Computer Vision, 134:86, 2026.
17. Nick Nikzad, Yi Liao, Yongsheng Gao, Jun Zhou. "SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers." CVPR 2025.
18. Zhenglun Kong, Yize Li, Fanhu Zeng, Lei Xin, Shvat Messica, Xue Lin, Pu Zhao, Manolis Kellis, Hao Tang, Marinka Zitnik. "Token Reduction Should Go Beyond Efficiency in Generative Models — From Vision, Language to Multimodality." arXiv:2505.18227, 2025.
19. Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, Joshua M. Susskind. "Stabilizing Transformer Training by Preventing Attention Entropy Collapse." ICML 2023.
