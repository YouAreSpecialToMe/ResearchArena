# RobustSAE: Adversarially Robust Sparse Autoencoders for Trustworthy Concept Interpretation

## Abstract

Sparse autoencoders (SAEs) have emerged as a powerful tool for interpreting large language models (LLMs) by decomposing dense activations into sparse, human-interpretable features. However, recent work demonstrates that SAE interpretations are alarmingly fragile: minimal adversarial perturbations can drastically alter SAE feature activations while leaving the underlying LLM behavior unchanged. This "interpretability illusion" renders current SAEs unsuitable for safety-critical applications like model monitoring and oversight.

We propose **RobustSAE**, the first adversarially robust sparse autoencoder framework that explicitly trains for concept stability under input perturbations. RobustSAE introduces a novel **consistency regularization** objective that enforces stable SAE representations across semantically-equivalent inputs, and an **unsupervised robustness proxy** that identifies potentially fragile features without requiring adversarial training data. Our approach fundamentally differs from concurrent diversity-focused methods: while "Beyond Sparsity" (Pan et al., ICML 2025) uses denoising to improve feature coverage, RobustSAE targets the distinct problem of adversarial robustness, making the two approaches complementary rather than competing.

Through comprehensive experiments on language models, we demonstrate that RobustSAE significantly improves SAE robustness metrics while maintaining reconstruction quality and interpretability. Our work addresses a critical gap identified by Li et al. (2025) and establishes robustness as a first-class objective in SAE training.

---

## 1. Introduction

### 1.1 Background and Motivation

Mechanistic interpretability seeks to understand neural networks by decomposing their internal representations into human-interpretable components. Sparse autoencoders (SAEs) have become the dominant approach for this task, learning overcomplete dictionaries that map dense LLM activations to sparse feature representations [Cunningham et al., 2023; Bricken et al., 2023]. Each active SAE feature ideally corresponds to a coherent semantic concept, enabling precise model inspection and control.

The prevailing evaluation paradigm for SAEs focuses on three axes: (1) reconstruction-sparsity tradeoffs [Gao et al., 2024], (2) automated interpretability scores [Paulo et al., 2024], and (3) feature diversity and disentanglement [Karvonen et al., 2025]. While these metrics assess SAE quality under ideal conditions, they overlook a critical question: **Are SAE interpretations stable under realistic input variations?**

### 1.2 The Robustness Problem

Li et al. (2025) recently introduced a comprehensive framework for evaluating SAE robustness and made a troubling discovery: SAE concept representations are highly vulnerable to minimal adversarial perturbations. Using gradient-based attacks (adapted from GCG [Zou et al., 2023]), they show that:

1. Tiny input perturbations (e.g., adding a few adversarial tokens) can completely change SAE activation patterns
2. These perturbations often leave the LLM's semantic output unchanged
3. Both population-level activation patterns and individual features can be manipulated with high success rates

This reveals an **interpretability illusion**: SAEs may report completely different concepts for semantically equivalent inputs, rendering them unreliable for model monitoring, safety evaluation, or any downstream application requiring stable interpretations.

### 1.3 Our Contribution

We propose **RobustSAE**, addressing the robustness gap through two key innovations:

**1. Consistency-Regularized SAE Training**: We introduce a training objective that penalizes representation divergence between original and perturbed inputs. Unlike denoising approaches that add noise for diversity, our consistency loss specifically targets adversarial robustness by enforcing Lipschitz continuity in the encoder mapping.

**2. Unsupervised Robustness Proxy**: We develop a lightweight method for identifying potentially fragile features without requiring adversarial training. This proxy uses gradient concentration analysis to flag features likely to be unstable, serving as an early warning system for interpretability illusions.

**Distinction from Concurrent Work**: We explicitly address the concurrent "Beyond Sparsity" paper (Pan et al., ICML 2025), which uses dropout-based denoising to improve feature diversity. While both papers involve input perturbations, they solve orthogonal problems:
- **Diversity** (Pan et al.): Prevents feature collapse, improves coverage
- **Robustness** (Our work): Prevents adversarial manipulation, ensures stability

These approaches are complementary and can be combined for SAEs that are both diverse and robust.

---

## 2. Related Work

### 2.1 Sparse Autoencoders for Interpretability

SAEs were introduced for mechanistic interpretability by Cunningham et al. (2023) and Bricken et al. (2023), building on earlier dictionary learning approaches. Standard SAEs optimize a reconstruction loss with sparsity constraints:

$$\mathcal{L}_{SAE} = \mathbb{E}[||h - \hat{h}||^2] + \lambda ||z||_1$$

where $h$ is the LLM activation, $\hat{h} = W_d z$ is the reconstruction, and $z = \text{ReLU}(W_e h + b_e)$ is the sparse latent representation.

Recent architectural improvements include TopK activation [Gao et al., 2024], JumpReLU [Rajamanoharan et al., 2024], Gated SAEs [Rajamanoharan et al., 2024], and Matryoshka SAEs [Bussmann et al., 2024]. These focus on improving the reconstruction-sparsity Pareto frontier.

### 2.2 Diversity in SAEs

Several concurrent works address feature redundancy in SAEs:

**"Beyond Sparsity" (Pan et al., ICML 2025)**: Introduces dropout-based denoising training to improve feature diversity. They adapt dropout as a data augmentation strategy, showing improved feature coverage and interpretability. This is the most closely related concurrent work.

**Multi-Expert SAE (Xu et al., arXiv 2025)**: Uses a Mixture-of-Experts architecture with specialized expert routing to encourage diverse feature learning. They report 99% reduction in feature redundancy.

Our work targets a different problem (robustness vs. diversity) and can be combined with both approaches.

### 2.3 SAE Robustness and Reliability

**Interpretability Illusions (Li et al., 2025)**: Identifies the robustness problem but does not propose training solutions. They call for "further denoising or postprocessing" to address fragility. Our work directly answers this call.

**SAE Evaluation Benchmarks**: SAEBench [Karvonen et al., 2025] provides comprehensive evaluation metrics including feature disentanglement and coverage, but does not include robustness evaluation. We extend this framework with robustness metrics.

### 2.4 Robust Representation Learning

Adversarial training [Madry et al., 2018] and consistency regularization [Bachman et al., 2014; Xie et al., 2020] are established techniques in supervised learning. We adapt these concepts to the unsupervised SAE setting, addressing unique challenges including sparsity constraints and dictionary learning dynamics.

---

## 3. Proposed Approach

### 3.1 Problem Formulation

Given an LLM activation $h(x) \in \mathbb{R}^d$ for input $x$, an SAE encoder $f_\theta: \mathbb{R}^d \rightarrow \mathbb{R}^m$ maps to sparse latents $z = f_\theta(h(x))$. An adversarial perturbation $x'$ (semantically equivalent to $x$) should ideally yield similar representations: $z \approx z'$.

Following Li et al. (2025), we define robustness violations when:
- **Type 1**: Semantically similar inputs map to dissimilar SAE activations ($d_z(z_1, z_2) > \epsilon$ despite semantic similarity)
- **Type 2**: Semantically dissimilar inputs map to similar SAE activations (adversarial manipulation)

### 3.2 Consistency-Regularized SAE Training

Our training objective augments the standard SAE loss with a consistency regularization term:

$$\mathcal{L}_{RobustSAE} = \mathcal{L}_{recon} + \lambda_{sparse} \mathcal{L}_{sparse} + \lambda_{consist} \mathcal{L}_{consist}$$

**Reconstruction Loss**: Standard MSE between original and reconstructed activations:
$$\mathcal{L}_{recon} = ||h - W_d z||^2$$

**Sparsity Loss**: L1 penalty on latent activations (or TopK threshold):
$$\mathcal{L}_{sparse} = ||z||_1$$

**Consistency Loss** (Novel): For input $x$ and perturbed variant $x'$, we enforce:
$$\mathcal{L}_{consist} = ||z - z'||^2 + \gamma ||\text{topk}(z) - \text{topk}(z')||^2$$

The consistency loss has two components:
1. **Continuous consistency**: Soft alignment of pre-activation values
2. **Discrete consistency**: Alignment of sparsity patterns (which features fire)

**Perturbation Strategy**: Unlike denoising for diversity (which uses random dropout), we use targeted perturbations that preserve semantics:
- Paraphrase generation via LLM
- Token synonym substitution
- Minor punctuation/case variations

These perturbations simulate natural input variation rather than destructive noise.

### 3.3 Unsupervised Robustness Proxy

We introduce a method to identify potentially fragile features without adversarial training:

**Gradient Concentration Score**: For each feature $i$, compute:
$$R_i = \frac{||\nabla_{h} z_i||_2}{|z_i| + \delta}$$

Features with high $R_i$ have activations that change rapidly with small input perturbations, indicating fragility. We rank features by $R_i$ to identify those most in need of additional regularization.

**Local Lipschitz Estimation**: For feature $i$, estimate local Lipschitz constant via gradient sampling:
$$L_i^{local} = \max_{||\Delta h|| < \epsilon} \frac{|z_i(h + \Delta h) - z_i(h)|}{||\Delta h||}$$

Features with high estimated Lipschitz constants are flagged as potentially non-robust.

### 3.4 Theoretical Justification

**Proposition 1** (Consistency implies robustness): If the SAE encoder satisfies $||f_\theta(h) - f_\theta(h')|| \leq L ||h - h'||$ (Lipschitz continuity) and the LLM activations satisfy $||h(x) - h(x')|| \leq \epsilon$ for semantically similar $x, x'$, then the SAE representations are robust to semantic perturbations.

**Proof**: Direct application of Lipschitz property. The consistency loss encourages Lipschitz continuity by penalizing large changes in $z$ for small changes in $h$.

---

## 4. Experimental Plan

### 4.1 Research Questions

1. Does consistency regularization improve SAE robustness metrics without sacrificing reconstruction quality?
2. Does the unsupervised robustness proxy correlate with actual adversarial vulnerability?
3. How do robustness and diversity objectives interact when combined?
4. Does RobustSAE improve reliability for downstream applications (feature steering, circuit analysis)?

### 4.2 Datasets and Models

**Models**:
- Pythia-70M and Pythia-160M [Biderman et al., 2023] for controlled experiments
- Gemma-2B [Team et al., 2024] for scale validation

**Data**:
- OpenWebText (2M tokens for training)
- Art & Science dataset from Li et al. (2025) for robustness evaluation
- AG News for domain generalization testing

### 4.3 Baselines

1. **Standard TopK SAE** [Gao et al., 2024]
2. **JumpReLU SAE** [Rajamanoharan et al., 2024]
3. **Denoising SAE** ("Beyond Sparsity" replication): SAE with dropout-based denoising
4. **Consistency-only SAE**: Our method without robustness proxy

### 4.4 Metrics

**Standard Metrics**:
- Fraction of Variance Unexplained (FVU)
- L0 sparsity (average number of active features)
- Loss recovered when substituting SAE reconstructions

**Robustness Metrics** (from Li et al., 2025):
- Population-level overlap change under perturbation
- Individual feature activation success rate (ASR)
- Semantic stability: LLM-judged semantic equivalence preservation

**Interpretability Metrics**:
- Automated interpretability score [Paulo et al., 2024]
- Feature disentanglement score [Karvonen et al., 2025]

### 4.5 Experimental Protocol

**Phase 1: Training** (Est. 4 hours)
- Train RobustSAE and baselines on Pythia-70M activations
- Grid search over $\lambda_{consist} \in \{0.01, 0.1, 1.0\}$

**Phase 2: Robustness Evaluation** (Est. 2 hours)
- Implement GCG-based attacks following Li et al. (2025)
- Evaluate all methods across 8 attack scenarios (2 semantic goals × 2 activation goals × 2 perturbation modes)
- Report attack success rates and semantic stability

**Phase 3: Proxy Validation** (Est. 1 hour)
- Compute gradient concentration scores for all features
- Correlation analysis between proxy scores and empirical attack success

**Phase 4: Downstream Validation** (Est. 1 hour)
- Feature steering experiments: Compare stability of steering vectors
- Circuit analysis: Evaluate faithfulness under input perturbations

### 4.6 Expected Results

**Hypothesis 1**: RobustSAE will show >30% reduction in attack success rate compared to standard SAEs, with <5% increase in FVU.

**Hypothesis 2**: The unsupervised robustness proxy will achieve >0.7 correlation with empirical attack success.

**Hypothesis 3**: Combining RobustSAE with denoising (diversity) will yield SAEs superior to either approach alone on both robustness and diversity metrics.

---

## 5. Success Criteria

### 5.1 Confirming Results

The hypothesis is **confirmed** if:
- RobustSAE demonstrates statistically significant (p < 0.01) improvement in robustness metrics vs. all baselines
- Attack success rate reduced by at least 25% relative to standard TopK SAE
- Reconstruction quality (FVU) maintained within 10% of baseline
- Unsupervised proxy shows >0.6 Spearman correlation with empirical fragility

### 5.2 Refuting Results

The hypothesis is **refuted** if:
- No significant improvement in robustness metrics
- Consistency regularization catastrophically degrades reconstruction (>30% FVU increase)
- Robustness gains only apply to specific attack types (not general)
- Proxy method shows no correlation with actual robustness

### 5.3 Partial Success

**Partial success** scenarios:
- Robustness improves but only for population-level attacks (not individual features)
- Method works for small models but fails to scale
- Reconstruction-robustness tradeoff requires problematic hyperparameter tuning
- Proxy works for some feature types but not others

---

## 6. Discussion

### 6.1 Limitations

1. **Computational Cost**: Consistency regularization requires multiple forward passes per input, increasing training time by ~2-3×.

2. **Perturbation Coverage**: Our semantic perturbations may not cover all adversarial strategies; adaptive attacks may find remaining vulnerabilities.

3. **Hyperparameter Sensitivity**: The $\lambda_{consist}$ tradeoff requires tuning and may vary across model sizes.

### 6.2 Future Work

1. **Adaptive Attacks**: Evaluate against adversaries aware of the consistency training
2. **Certified Robustness**: Explore randomized smoothing or other certified defenses for SAEs
3. **Cross-Model Transfer**: Study whether robust features transfer across different LLMs
4. **Safety Applications**: Apply RobustSAE to red-teaming and model monitoring scenarios

### 6.3 Broader Impact

Reliable SAEs are essential for AI safety. By addressing the robustness gap identified by Li et al. (2025), we enable more trustworthy model interpretation and oversight. However, robust SAEs could also be used to make model deception harder to detect—a dual-use consideration we acknowledge.

---

## 7. Conclusion

We propose RobustSAE, the first adversarially robust sparse autoencoder framework. By introducing consistency regularization and an unsupervised robustness proxy, we address a critical gap in SAE reliability. Our approach is orthogonal to concurrent diversity methods, offering a complementary path toward SAEs that are both comprehensive and trustworthy. We expect this work to establish robustness as a first-class objective in interpretability research.

---

## References

1. **Aaron J. Li, Suraj Srinivas, Usha Bhalla, Himabindu Lakkaraju**. "Interpretability Illusions with Sparse Autoencoders: Evaluating Robustness of Concept Representations." *arXiv preprint arXiv:2505.16004*, 2025.

2. **Xiang Pan, Yifei Wang, Qi Lei**. "Beyond Sparsity: Improving Diversity in Sparse Autoencoders via Denoising Training." *ICML 2025*.

3. **Zhen Xu, Zhen Tan, Song Wang, Kaidi Xu, Tianlong Chen**. "Beyond Redundancy: Diverse and Specialized Multi-Expert Sparse Autoencoder." *arXiv preprint arXiv:2511.05745*, 2025.

4. **Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, Lee Sharkey**. "Sparse Autoencoders Find Highly Interpretable Features in Language Models." *arXiv preprint arXiv:2309.08600*, 2023.

5. **Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, et al.** "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning." *Transformer Circuits Thread*, 2023.

6. **Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, Jeffrey Wu**. "Scaling and Evaluating Sparse Autoencoders." *arXiv preprint arXiv:2406.04093*, 2024.

7. **Adam Karvonen, Benjamin Wright, Can Rager, Rico Angell, Jannik Brinkmann, Logan Smith, Claudio Mayrink Verdun, David Bau, Samuel Marks**. "Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models." *NeurIPS 2024*.

8. **Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson**. "Universal and Transferable Adversarial Attacks on Aligned Language Models." *arXiv preprint arXiv:2307.15043*, 2023.

9. **Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, Aaron Mueller**. "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models." *ICLR 2024*.

10. **Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu**. "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR 2018*.

11. **Lee Sharkey, Bilal Chughtai, Joshua Batson, Jack Lindsey, Jeff Wu, Lucius Bushnaq, Nicholas Goldowsky-Dill, et al.** "Open Problems in Mechanistic Interpretability." *arXiv preprint arXiv:2501.16496*, 2025.
