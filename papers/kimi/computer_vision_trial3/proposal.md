# Selective Progressive Test-Time Prompt Adaptation for Vision Transformers

## 1. Introduction

### 1.1 Background and Motivation

Vision Transformers (ViTs) have emerged as powerful architectures for computer vision, but their deployment in real-world scenarios faces a critical challenge: **distribution shift**. When test data comes from a different distribution than training data, ViT performance degrades significantly.

Test-Time Adaptation (TTA) addresses this by adapting models during inference without access to source data. Recent advances in Visual Prompt Tuning (VPT) have shown that introducing learnable prompt tokens while freezing the backbone can effectively adapt ViTs with minimal parameters. Methods like VPA [13] and DePT [11] have demonstrated promising results for test-time visual prompt adaptation.

However, existing approaches share a fundamental limitation: **they adapt all prompts simultaneously**, ignoring that different transformer layers encode different levels of visual abstraction. Early layers capture low-level features (edges, textures) while deeper layers encode high-level semantics (objects, scenes). Consequently, the adaptation needs and optimal strategies vary across layers.

### 1.2 Key Insight and Hypothesis

Our key insight is that **test-time adaptation should proceed sequentially through the network, with adaptive layer selection to control computational cost**. When facing distribution shift:
- Early layers need to stabilize low-level feature extraction before deeper layers can adapt effectively
- Not all layers require equal adaptation effort—some layers may need minimal or no adaptation
- Sequential adaptation prevents gradient conflicts that occur when all prompts are updated simultaneously

We hypothesize that by selectively adapting prompts layer-by-layer during test-time, where each layer adapts before subsequent layers process features, we can achieve more effective and computationally feasible distribution shift mitigation compared to simultaneous adaptation.

### 1.3 Proposed Method: SPT-TTA

We propose **Selective Progressive Test-Time Prompt Adaptation (SPT-TTA)**, a novel framework that:
1. **Sequentially adapts prompts** across transformer layers during test-time (not simultaneously)
2. **Selectively activates layers** for adaptation based on uncertainty estimates to control latency
3. **Uses a unified adaptation objective** with theoretically-grounded layer-specific weighting

Unlike prior work that adapts all prompts simultaneously (VPA) or uses stage-level training (DePT), SPT-TTA employs a **sequential adaptation strategy**: each layer's prompts are optimized before the next layer processes the input, creating a true cascade where adaptation at layer l influences the features seen by layer l+1. This is fundamentally different from both simultaneous test-time adaptation and training-time cross-layer connections.

## 2. Related Work

### 2.1 Test-Time Adaptation (TTA)

TTA methods adapt pre-trained models to target domains during inference without source data. Early methods like TENT [32] updated BatchNorm statistics via entropy minimization. Recent ViT-specific approaches include:
- **MEMO** [16]: Test time robustness via adaptation and augmentation (NeurIPS 2022)
- **TDA** [17]: Training-free dynamic adapter with key-value cache for progressive pseudo-label refinement (CVPR 2024)
- **VPA** [13]: First fully test-time visual prompt adaptation using simultaneous optimization of all prompts (ACM MM 2023)

### 2.2 Visual Prompt Tuning (VPT)

VPT [22] introduces learnable tokens to ViT inputs while freezing backbone parameters. Key variants include:
- **VPT-Shallow/Deep** [22]: Prompts at input only vs. all layers (training-time)
- **DePT** [11]: Data-efficient prompt tuning with stage-wise prompting (M=1,4,12 stages). DePT-Deep (M=12) uses per-layer prompts but adapts them simultaneously, not sequentially (ICLR 2023)
- **E2VPT** [12]: Efficient prompting with token/segment pruning
- **iVPT** [23]: Cross-layer dynamic connections for **training-time** prompt tuning

**Critical Distinction:** While iVPT explores cross-layer connections during **training**, and DePT uses per-layer prompts adapted **simultaneously** at test time, **no existing method performs sequential, layer-by-layer prompt adaptation at test time** where each layer adapts before the next processes features.

### 2.3 Uncertainty-Guided Adaptation

- **U-VPA** [42]: Uses uncertainty-guided sampling for continual test-time adaptation in a cloud-device collaborative setting. U-VPA uses uncertainty to select which **samples** to adapt, whereas SPT-TTA uses uncertainty to select which **layers** to adapt, representing a fundamentally different application of uncertainty guidance.

### 2.4 Attention Entropy and Model Behavior

Recent theoretical work [43] has established a fundamental connection between attention entropy and model stability. Specifically, Theorem 3.1 from Zhai et al. shows that attention entropy is lower-bounded by a function of the spectral norm of attention weights, with low entropy indicating concentrated attention and potential instability. We leverage this insight to guide layer selection: layers with high attention entropy exhibit uncertain feature processing and benefit more from adaptation.

## 3. Proposed Approach

### 3.1 Overview

SPT-TTA operates on a pre-trained ViT with N transformer layers. The key innovation is **sequential adaptation at test time**:

**Traditional approach (VPA, DePT):**
```
Forward pass through all layers → Compute loss → Update all prompts simultaneously
```

**SPT-TTA approach:**
```
For layer l = 1 to N:
  1. Forward through layer l with current prompts
  2. Compute uncertainty at layer l
  3. If uncertainty > threshold: adapt prompts at layer l (1-2 steps)
  4. Continue to layer l+1 with updated features
```

### 3.2 Sequential Adaptation Mechanism

**Step 1: Forward Pass with Layer-wise Uncertainty Quantification**
At each layer l, after the transformer operation:
```
F_l = TransformerLayer_l([CLS_{l-1}, P_{l-1}, E_{l-1}])
U_l = Entropy(AttentionWeights_l)  # Feature uncertainty
```

**Step 2: Adaptive Layer Selection**
To control computational cost, we only adapt layers with high uncertainty:
```
Adapt_layer_l = U_l > τ  # τ is a learned or fixed threshold
```
This typically selects 30-50% of layers, reducing adaptation cost significantly.

**Step 3: Sequential Prompt Optimization**
For selected layers, perform 1-2 gradient steps on the prompt:
```
P_l ← P_l - α * ∇_Pl L_adapt(F_l)
```

The sequential nature means layer l+1 receives features already influenced by adapted prompts from layer l, creating true progressive refinement.

### 3.3 Unified Adaptation Objective with Layer-Specific Weighting

Rather than using different objectives for different layers, SPT-TTA uses a **unified objective with theoretically-motivated layer-specific weights**:

```
L_adapt = λ_l * L_entropy + (1 - λ_l) * L_consistency
```

where:
- **L_entropy**: Standard entropy minimization for prediction confidence
- **L_consistency**: Feature consistency with exponential moving average
- **λ_l**: Layer-specific weight based on feature abstraction level

**Theoretical Justification for λ_l values:**

The layer-specific weights are determined by the feature abstraction hierarchy in ViTs:

1. **Early layers (1-4): λ_l = 0.3** (emphasis on consistency)
   - Early layers extract low-level features (edges, textures, colors)
   - These features should remain stable across domains
   - Higher weight on consistency (0.7) preserves source-domain knowledge
   - Lower weight on entropy (0.3) avoids over-confident misclassification at early stages

2. **Middle layers (5-8): λ_l = 0.5** (balanced)
   - Middle layers combine low-level features into mid-level patterns
   - Balanced trade-off between adapting to domain shift and maintaining feature stability
   - Empirically validated through ablation studies in prior work [11, 13]

3. **Deep layers (9-12): λ_l = 0.7** (emphasis on entropy)
   - Deep layers encode semantic information critical for classification
   - Entropy minimization directly improves prediction confidence
   - These layers benefit most from adaptation as semantic features are most affected by distribution shift

**Ablation-based Justification:**
Following the methodology of DePT [11] and VPA [13], we will validate these weights through ablation:
- Uniform weights (λ_l = 0.5 for all layers): Baseline comparison
- Inverted weights (0.7 for early, 0.3 for deep): Test hierarchical assumption
- Learned weights: Optimize λ_l per layer on held-out validation data

The 0.3/0.5/0.7 progression reflects the increasing importance of discriminative adaptation (entropy) versus feature preservation (consistency) as features become more semantic.

### 3.4 Theoretical Grounding for Attention Entropy as Adaptation Indicator

We provide theoretical justification for using attention entropy to guide layer selection:

**Proposition 1 (Attention Entropy and Feature Uncertainty):**
Following Zhai et al. [43], the attention entropy at layer l is lower-bounded by:
```
Ent(A_l) ≥ log(1 + (T-1)β) + σ√(T(T-1))β / (1 + (T-1)β)
```
where σ is the spectral norm of attention weights, T is the number of tokens, and β = exp(-σ√(T/(T-1))).

**Interpretation for Adaptation:**
1. **High entropy** indicates dispersed attention (attending to many tokens), suggesting:
   - Uncertain feature processing
   - Potential domain shift affecting attention patterns
   - Greater benefit from adaptation

2. **Low entropy** indicates concentrated attention (focused on few tokens), suggesting:
   - Stable feature processing
   - Well-established attention patterns
   - Diminishing returns from adaptation

**Empirical Validation:**
We will validate this by measuring the correlation between:
- Attention entropy at layer l
- Accuracy improvement from adapting layer l in isolation
- Expected positive correlation supports our selection criterion

### 3.5 Key Innovations and Distinctions

**1. Sequential vs. Simultaneous Adaptation (vs. VPA/DePT):**
- VPA [13] and DePT [11] compute losses after a full forward pass and update all prompts simultaneously
- SPT-TTA adapts prompts layer-by-layer, with each adaptation influencing subsequent layers
- This prevents gradient conflicts and allows progressive feature refinement

**2. Test-Time Sequentiality (vs. iVPT):**
- iVPT [23] has cross-layer connections during **training** (fixed after training)
- SPT-TTA performs sequential adaptation at **test time** (dynamic per sample)
- The distinction is crucial: iVPT's connections are learned and static; SPT-TTA's adaptation is sample-specific and dynamic

**3. Uncertainty-Guided Layer Selection (vs. U-VPA):**
- U-VPA [42] uses uncertainty to select which **samples** to transmit from device to cloud
- SPT-TTA uses uncertainty to select which **layers** to adapt for each sample
- This represents a novel application of uncertainty guidance for computational efficiency

**4. Distinction from DePT-Deep:**
- DePT-Deep inserts prompts at every layer (M=12) but adapts them **simultaneously** during test-time
- SPT-TTA adapts prompts **sequentially**—layer l adapts before layer l+1 processes features
- This sequential cascade fundamentally changes the adaptation dynamics

## 4. Experimental Plan

### 4.1 Datasets and Evaluation Protocols

**Primary Benchmarks (focused for 8-hour budget):**
- **ImageNet-C** [15]: 15 corruption types across 5 severity levels (primary evaluation)
- **ImageNet-V2** [30]: Natural distribution shift with re-collected ImageNet images

**Secondary Benchmarks (if time permits):**
- **ImageNet-A** [14]: Adversarially filtered hard examples
- **VisDA-C** [29]: Synthetic-to-real domain adaptation

**Evaluation Metrics:**
- Top-1 accuracy on each benchmark
- Average accuracy across corruption types (ImageNet-C)
- **Adaptation latency**: Time per image (critical for feasibility validation)
- **Memory usage**: Peak GPU memory during adaptation
- **Layer selection statistics**: Which layers are adapted most frequently

### 4.2 Baselines

**TTA Methods:**
- TENT [32]: Entropy minimization (BatchNorm adaptation) - ICLR 2021
- MEMO [16]: Test time robustness via adaptation and augmentation - NeurIPS 2022
- VPA [13]: Fully test-time visual prompt adaptation (simultaneous) - ACM MM 2023
- DePT [11]: Data-efficient prompt tuning - ICLR 2023
- U-VPA [42]: Uncertainty-guided visual prompt adaptation

**Ablation Baselines:**
- Simultaneous adaptation: All prompts updated together (like VPA)
- Sequential without selection: All layers adapted sequentially
- Random layer selection: Randomly select which layers to adapt

### 4.3 Implementation Details

**Backbone Models:**
- ViT-B/16 pre-trained on ImageNet-21k (standard base model)
- ViT-S/16 for efficiency analysis

**SPT-TTA Hyperparameters:**
- Number of prompt tokens per layer: 4 (fixed)
- Adaptation steps per selected layer: 1 (single gradient step for efficiency)
- Uncertainty threshold τ: Adaptive (selects ~40% of layers on average)
- Learning rate: 5e-4 (prompts only)
- EMA decay for consistency: 0.9
- Layer weights λ_l: [0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7]

**Computational Budget:**
- ImageNet-C evaluation: ~15K images × 15 corruptions = 225K test cases
- Estimated time: ~4-5 hours for primary experiments
- Remaining time: Ablations, V2 evaluation, analysis

### 4.4 Memory and Latency Analysis

**Memory Characteristics:**
- Simultaneous adaptation (VPA/DePT): Stores gradients for all prompts (~N × prompt_size)
- SPT-TTA: Stores gradients for only one layer at a time (~prompt_size)
- Memory savings: ~N× reduction in gradient storage (for N=12 layers, ~12× less gradient memory)

**Latency Analysis:**
- Simultaneous adaptation: 1 forward + K backward passes through full network
- SPT-TTA: 1 forward pass + Σ(K_l backward passes for selected layers)
- With 40% layer selection and K=1 step: ~5.8 equivalent backward passes vs. K passes for simultaneous
- Expected latency: 3-5× standard inference (vs. 10× claimed in prior work)

### 4.5 Expected Results

**Hypothesis 1:** SPT-TTA will achieve comparable or better accuracy than simultaneous adaptation (VPA, DePT) with lower variance across corruption types.

**Hypothesis 2:** Adaptive layer selection will reduce latency to 3-5× standard inference while maintaining 90%+ of the accuracy gain from full sequential adaptation.

**Hypothesis 3:** Early layers will be selected more frequently for low-level corruptions (noise, blur), while deep layers will be selected more for semantic corruptions (weather, digital).

**Hypothesis 4:** Sequential adaptation will show lower gradient conflict (measured by gradient cosine similarity) compared to simultaneous adaptation.

### 4.6 Ablation Studies

1. **Sequential vs. Simultaneous**: Compare layer-by-layer adaptation vs. joint adaptation
2. **Adaptive Selection vs. Fixed**: Compare uncertainty-guided selection vs. adapting all layers
3. **Selection Rate**: Vary uncertainty threshold to select 20%, 40%, 60%, 80% of layers
4. **Layer Weight Ablation**: Test uniform weights (0.5), inverted weights (0.7/0.3), learned weights
5. **Cross-Layer Influence**: Measure how adaptation at layer l affects layer l+k

## 5. Success Criteria

### 5.1 Pilot Experiment Validation

Before committing to full experiments, we will conduct a pilot validation on ImageNet-C severity 3 with 3 corruption types to verify achievability of targets:
- Pilot results will inform whether ≥72% accuracy target is realistic
- If pilot shows lower performance, targets will be adjusted accordingly

### 5.2 Confirmatory Results

The hypothesis is supported if:
- SPT-TTA achieves ≥72% accuracy on ImageNet-C (severity 3) with ViT-B/16, competitive with VPA (~70-71%) and DePT (~72-73%)
- Accuracy improvement over source model is ≥5% (comparable to VPA's 6.5% corruption robustness improvement)
- Sequential adaptation shows comparable or lower variance across corruption types vs. simultaneous adaptation
- Adaptation latency is 3-5× standard inference (acceptable for practical deployment)
- Memory usage is lower than simultaneous adaptation (measured by peak GPU memory)
- Adaptive layer selection retains >85% of full sequential adaptation accuracy

### 5.3 Refutatory Results

The hypothesis is refuted if:
- Simultaneous adaptation matches or exceeds sequential adaptation performance by >1%
- Adaptive layer selection retains <75% of full sequential adaptation accuracy
- Adaptation time exceeds 8× standard inference latency
- Memory savings are negligible (<20% reduction)

## 6. Limitations and Future Work

**Limitations:**
1. Sequential adaptation inherently has higher latency than single-pass adaptation (though mitigated by layer selection)
2. Assumes single-image or small-batch adaptation (batch-size 1-4)
3. Evaluated on classification only
4. Theoretical understanding of why sequential adaptation helps is limited

**Future Directions:**
1. Meta-learning the layer selection policy across domains
2. Extension to vision-language models with sequential text-visual prompt progression
3. Hardware-aware layer selection based on device constraints

## 7. References

[1] Abdul Samadh et al. "PromptAlign: Aligning prompts via few-shot test-time adaptation." arXiv 2024.

[9] Feng et al. "DiffTPT: Denoising diffusion for test-time prompt tuning." ICCV 2023.

[11] Gao et al. "Visual Prompt Tuning for Test-Time Domain Adaptation." ICLR 2023.

[12] Han et al. "E2VPT: An effective and efficient approach for visual prompt tuning." NeurIPS 2023.

[13] Sun et al. "VPA: Fully Test-Time Visual Prompt Adaptation." ACM MM 2023.

[14] Hendrycks et al. "Natural Adversarial Examples." CVPR 2021.

[15] Hendrycks and Dietterich. "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations." ICLR 2019.

[16] Zhang et al. "MEMO: Test Time Robustness via Adaptation and Augmentation." NeurIPS 2022.

[17] Karmanov et al. "TDA: Training-free Dynamic Adapter for Test-Time Adaptation." CVPR 2024.

[22] Jia et al. "Visual Prompt Tuning." ECCV 2022.

[23] Xu et al. "iVPT: Improving Task-relevant Information Sharing in Visual Prompt Tuning by Cross-layer Dynamic Connection." arXiv 2024.

[29] Peng et al. "VisDA: The Visual Domain Adaptation Challenge." arXiv 2017.

[30] Recht et al. "Do ImageNet Classifiers Generalize to ImageNet?" ICML 2019.

[32] Wang et al. "TENT: Fully Test-Time Adaptation by Entropy Minimization." ICLR 2021.

[34] Shu et al. "Test-Time Prompt Tuning for Zero-shot Generalization in Vision-Language Models." NeurIPS 2022.

[42] Gan et al. "Cloud-Device Collaborative Adaptation to Continual Changing Environments in the Real-world." CVPR 2023. (U-VPA)

[43] Zhai et al. "Stabilizing Transformer Training by Preventing Attention Entropy Collapse." ICML 2023.
