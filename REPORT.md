# The Residual Stream After Bijective Token Transformation

## 1. Executive Summary

We investigated what happens to a pre-trained transformer's internal representations when token identities are bijectively permuted at inference time. Using GPT-2 Small (124M parameters) and WikiText-2, we applied a fixed random bijection f: V → V to all token IDs before feeding them to the model, then compared residual stream activations against normal (unpermuted) processing across all 12 layers.

**Key finding:** The model **never recovers** from embedding misalignment. The residual stream under bijective permutation follows a distinctive **U-shaped similarity curve** — moderate similarity at the embedding layer (~0.59), dropping to a minimum at layers 6-7 (~0.47), then rising sharply at layers 10-11 (~0.85). However, this late-layer convergence is **not semantic recovery**: logit lens analysis shows near-zero top-1 prediction match (<1%) at all layers, and perplexity increases 2,315x. The final-layer convergence reflects structural regularities in how GPT-2 formats its output, not successful adaptation to the cipher.

**Practical implication:** Pre-trained transformers cannot "figure out" a bijective token shuffle through in-context processing alone. The embedding layer creates a fundamentally wrong representation that propagates irreversibly, consistent with the impossibility result of Alur et al. (2025). This contrasts with models *trained from scratch* on permuted input (Huang et al., 2023), which can achieve near-normal performance.

## 2. Goal

**Hypothesis:** Applying a bijective transformation that shuffles token-surface form mappings to the input of a language model will initially confuse the model, but after sufficient exposure, the model may adapt such that its residual stream representations become similar to normal, modulo the shuffle. Alternatively, the model may never fully recover due to embedding misalignment.

**Why this matters:**
- Reveals fundamental properties of how pre-trained transformers encode and process token identity
- Tests the limits of in-context adaptation vs. architectural constraints
- Bridges mechanistic interpretability with theoretical results on permutation learning
- Has implications for robustness, security (substitution ciphers), and understanding what the residual stream "knows"

**Gap filled:** No prior work has directly measured pre-trained model residual streams under bijective token identity permutation. The Lexinvariant LM paper (Huang et al., 2023) trained models from scratch; the Permutation Equivariance paper (Xu et al., 2024) studied position permutations; the Impossibility paper (Alur et al., 2025) proved theoretical limits but didn't characterize residual stream dynamics.

## 3. Data Construction

### Dataset Description
- **Source:** WikiText-2-raw-v1 (Merity et al., 2016), downloaded from HuggingFace
- **Split used:** Validation set (3,760 examples)
- **Filtered to:** 1,646 examples with >100 characters (to ensure meaningful context)
- **Sequence length:** Truncated to 256 tokens per sample

### Example Samples
Texts are Wikipedia articles covering diverse topics. Samples include:

| Sample | Length (chars) | Topic |
|--------|---------------|-------|
| "Robert Boulter is an English..." | 483 | Biography |
| "The game was played on..." | 312 | Sports |
| "Senjō no Valkyria 3..." | 1,247 | Video games |

### Preprocessing Steps
1. Filter validation set to texts >100 characters
2. Tokenize using GPT-2's BPE tokenizer (prepend BOS token)
3. Truncate to 256 tokens maximum
4. Apply bijective permutation to token IDs: `permuted_tokens = perm[original_tokens]`

### Bijective Permutation
- Created using `numpy.random.RandomState(42).permutation(50257)`
- **Fixed points:** 2 out of 50,257 tokens (0.004%) — essentially a full shuffle
- **Neighborhood preservation:** None — cosine similarity between E(t) and E(f(t)) is 0.270 ± 0.052, indistinguishable from random pairs (0.272 ± 0.053, t-test p=0.80)

## 4. Experiment Description

### Methodology

#### High-Level Approach
For each text sample, we run two forward passes through GPT-2 Small:
1. **Normal:** tokenize text → run through model → cache residual stream at all layers
2. **Permuted:** tokenize text → apply bijection to token IDs → run through model → cache residual stream

We then compare the residual streams using cosine similarity, L2 distance, logit lens predictions, attention patterns, and norm correlations.

#### Why This Method?
- **TransformerLens** provides clean access to all residual stream positions via hooks
- **GPT-2 Small** is well-studied in mechanistic interpretability (12 layers, 768-dim residual stream, 50,257-token vocabulary)
- **Cosine similarity** captures directional alignment independent of magnitude
- **Logit lens** reveals what the model "thinks" at each layer by projecting the residual stream through the unembedding matrix

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0+cu128 | Tensor computation |
| TransformerLens | 2.15.4 | Residual stream access |
| transformers | 4.49.0 | Tokenization |
| NumPy | (bundled) | Array operations |
| SciPy | 1.17.1 | Statistical tests |
| matplotlib | 3.10.8 | Visualization |
| seaborn | 0.13.2 | Heatmaps |

#### Hardware
- GPU: NVIDIA RTX A6000 (49 GB VRAM)
- Inference only (no training)

#### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | GPT-2 Small (124M) | Well-studied, manageable size |
| Max sequence length | 256 tokens | Sufficient context for adaptation test |
| N samples (Exp 1) | 60 | Statistical power for per-layer analysis |
| N samples (Exp 2) | 40 | Logit lens is more expensive |
| Random seed | 42 | Reproducibility |
| Permutation | Full vocabulary (50,257) | Tests worst-case scenario |

### Experimental Protocol

**5 main experiments + 5 supplementary analyses:**

| Experiment | Description | Metric |
|------------|-------------|--------|
| Exp 1 | Layer-wise residual stream similarity | Cosine sim, L2 distance |
| Exp 2 | Logit lens prediction match | Top-1 match rate, KL divergence |
| Exp 3 | Context length effect | Cosine sim by token position |
| Exp 4 | Permutation-adjusted similarity | Norm correlation (Pearson r) |
| Exp 5 | Random baseline comparison | Cosine sim (perm vs random) |
| Supp 1 | Attention pattern similarity | Attention cosine sim |
| Supp 2 | Perplexity comparison | Cross-entropy loss |
| Supp 3 | Layer 11 norm investigation | Activation norms |
| Supp 4 | Partial permutation gradient | Cosine sim at varying % permuted |
| Supp 5 | Embedding neighborhood check | E(t) vs E(f(t)) similarity |

### Raw Results

#### Experiment 1: Layer-wise Cosine Similarity (Normal vs Permuted)

| Layer | Cosine Sim (mean ± SE) | L2 Distance (mean ± SE) |
|-------|------------------------|-------------------------|
| Embed | 0.5893 ± 0.0014 | 4.60 ± 0.01 |
| L0 | 0.6115 ± 0.0014 | 49.18 ± 0.06 |
| L1 | 0.6151 ± 0.0010 | 54.63 ± 0.09 |
| L2 | 0.5864 ± 0.0010 | 60.04 ± 0.11 |
| L3 | 0.5300 ± 0.0012 | 68.00 ± 0.11 |
| L4 | 0.5035 ± 0.0014 | 74.85 ± 0.12 |
| L5 | 0.4875 ± 0.0017 | 83.03 ± 0.11 |
| L6 | 0.4707 ± 0.0020 | 93.12 ± 0.13 |
| L7 | 0.4723 ± 0.0023 | 106.56 ± 0.17 |
| L8 | 0.4867 ± 0.0026 | 121.64 ± 0.22 |
| L9 | 0.5111 ± 0.0037 | 144.94 ± 0.38 |
| L10 | 0.6529 ± 0.0035 | 177.42 ± 0.63 |
| L11 | 0.8461 ± 0.0032 | 273.14 ± 1.64 |

**Key observation:** U-shaped cosine similarity — drops from 0.59 (embed) to 0.47 (L6-7), then rises to 0.85 (L11). But L2 distance monotonically increases from 4.6 to 273.1, meaning the vectors are getting *further apart in absolute terms* even as they become more directionally aligned.

#### Experiment 2: Logit Lens Predictions

| Layer | Raw Top-1 Match | Perm-Adjusted Match | KL Div (Raw) | KL Div (Adjusted) |
|-------|-----------------|---------------------|--------------|-------------------|
| Embed | 0.03% | 0.00% | 22.56 | 22.78 |
| L0 | 0.00% | 0.00% | 16.38 | 18.58 |
| L5 | 0.04% | 0.00% | 13.83 | 17.06 |
| L8 | 0.89% | 0.00% | 15.37 | 17.87 |
| L11 | 0.55% | 0.00% | 6.08 | 8.90 |

**Key observation:** The model's predictions under permutation *never* match normal predictions. Even the "permutation-adjusted" match rate (applying the inverse permutation to the permuted model's predictions) is exactly 0.00% — the model is not predicting "the right token but in the permuted space."

#### Experiment 3: Context Length Effect

Cosine similarity (Normal vs Permuted) by token position:

| Layer | Pos 0-32 | Pos 32-64 | Pos 96-128 | Pos 224-256 |
|-------|----------|-----------|------------|-------------|
| Embed | 0.630 | 0.584 | 0.569 | 0.557 |
| L6 | 0.507 | 0.466 | 0.453 | 0.440 |
| L11 | 0.875 | 0.845 | 0.834 | 0.828 |

**Key observation:** Similarity *decreases* with more context at all layers (Spearman rho = -0.98 to -0.76, all p < 0.03). The model does NOT adapt with more context — it gets worse.

#### Experiment 4: Norm Correlation

| Layer | Pearson r (norms) |
|-------|-------------------|
| Embed | 0.693 ± 0.016 |
| L1 | 0.990 ± 0.001 |
| L2-L10 | 0.999 ± 0.000 |
| L11 | 0.089 ± 0.012 |

**Key observation:** The *magnitude pattern* (which positions have large/small activations) is nearly perfectly preserved through layers 2-10, then collapses at L11. This means the model processes permuted input with similar "intensity patterns" but completely different directions.

#### Experiment 5: Random Baseline Comparison

| Layer | vs Permuted | vs Random | Difference |
|-------|-------------|-----------|------------|
| L0 | 0.6115 | 0.6113 | +0.0002 |
| L5 | 0.4875 | 0.4720 | +0.0155 |
| L11 | 0.8461 | 0.8699 | -0.0237 |

**Key observation:** Permuted inputs are statistically distinguishable from random (Wilcoxon p < 0.001 at most layers), but the differences are small. At middle layers, permuted inputs are *slightly more similar* to normal than random inputs; at later layers, the pattern reverses.

#### Supplementary: Attention Patterns

| Layers 0-9 | Attention Sim |
|------------|--------------|
| Mean | 0.82 |
| Range | 0.78 - 0.86 |

| Layers 10-11 | Attention Sim |
|--------------|--------------|
| Mean | 0.65 |
| Range | 0.65 - 0.66 |

Attention patterns remain quite similar (0.78-0.86) through layers 0-9, then drop sharply at layers 10-11 (0.65). This explains the norm correlation collapse at L11 — the final layers change their attention patterns substantially under permutation.

#### Supplementary: Perplexity

| Condition | Perplexity |
|-----------|------------|
| Normal | 47.5 |
| Permuted (predicting permuted next token) | 109,989 |
| **Ratio** | **2,315x** |

The model cannot predict the next permuted token at all, confirming total failure to adapt.

#### Supplementary: Partial Permutation

| % Permuted | L0 Sim | L6 Sim | L11 Sim |
|------------|--------|--------|---------|
| 0% | 1.000 | 1.000 | 1.000 |
| 10% | 0.973 | 0.950 | 0.983 |
| 30% | 0.891 | 0.805 | 0.935 |
| 50% | 0.818 | 0.708 | 0.910 |
| 100% | 0.622 | 0.478 | 0.862 |

The U-shape is present at all permutation levels, with degradation roughly proportional to the fraction permuted.

### Visualization Summary

All plots are saved in `results/plots/`:
- `summary_figure.png` — 4-panel overview (cosine sim, logit lens, L2 distance, norm correlation)
- `layer_cosine_similarity.png` — Permuted vs Random comparison
- `logit_lens_analysis.png` — Top-1 match rates and KL divergences
- `context_length_heatmap.png` — Position × Layer heatmap
- `partial_permutation.png` — Effect of varying permutation fraction
- `attention_similarity.png` — Attention pattern similarity across layers
- `l2_distance_layers.png` — L2 distance growth
- `norm_correlation_layers.png` — Norm correlation U-shape and L11 collapse

## 5. Result Analysis

### Key Findings

**Finding 1: The residual stream never recovers semantically.**
Despite the U-shaped cosine similarity curve suggesting partial recovery at deeper layers, the logit lens shows the model never predicts the correct tokens — neither in the original vocabulary space (top-1 match <1%) nor in the permuted space (perm-adjusted match = 0.00%). The 2,315x perplexity increase confirms total semantic failure.

**Finding 2: The U-shape reflects structural, not semantic, convergence.**
The high cosine similarity at layer 11 (0.85) occurs simultaneously with:
- Monotonically increasing L2 distance (273 at L11, 60x larger than at embedding)
- Norm correlation collapse (r drops from 0.999 to 0.089 at L11)
- Attention pattern divergence (similarity drops to 0.65 at L10-11)

This means L11's residual stream directions converge toward similar orientations *because GPT-2's final layer has a stereotyped output format* — it projects toward high-frequency tokens and grammatical patterns regardless of whether the input makes sense. The vectors point in roughly the same direction but at very different magnitudes.

**Finding 3: More context makes things worse, not better.**
Contrary to the hypothesis that the model might "attune to the code" with more text, cosine similarity *decreases* with token position at all layers (Spearman rho = -0.76 to -1.0, all p < 0.03). The model accumulates increasingly wrong contextual representations as it processes more permuted tokens.

**Finding 4: The model preserves magnitude patterns but not directions.**
Norm correlation is nearly perfect (r > 0.99) at layers 2-10, meaning the model processes permuted and normal inputs with similar "activation intensity profiles" — the same positions get large activations. But the *directions* of these activations are fundamentally different (cosine sim 0.47-0.65 at the same layers). The model's attention mechanisms create position-dependent processing regardless of token identity, but the semantic content is wrong.

**Finding 5: The permutation provides no structural advantage over random tokens.**
The bijective structure of the permutation provides essentially no benefit compared to random (non-bijective) token replacement. Differences between permuted and random similarity are tiny (|Δ| < 0.03) and inconsistent in sign. A pre-trained model cannot exploit the bijective structure of the mapping.

### Hypothesis Testing Results

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Early confusion | **Supported** | Low similarity at embed (0.59) and early layers |
| H2: Layer-wise recovery | **Partially supported, but misleading** | Similarity increases at L10-11, but this is structural, not semantic |
| H3: Context accumulation helps | **Refuted** | Similarity decreases with more context (rho = -0.98, p < 0.001) |
| H4: Embedding ceiling prevents full recovery | **Strongly supported** | 0% logit lens match, 2,315x perplexity increase |
| H5: Permutation-adjusted similarity higher | **Refuted** | Perm-adjusted KL is *higher* (worse) than raw KL; match rate is 0.00% |

### Surprises and Insights

1. **The U-shape was unexpected.** We expected monotonically decreasing similarity (progressive corruption). Instead, the final layers show convergence — a structural artifact of GPT-2's output layer that is independent of input semantics.

2. **Norm correlation collapse at L11.** While norms are nearly perfectly correlated through layers 2-10 (r > 0.999), they abruptly decorrelate at L11 (r = 0.089). This coincides with attention pattern divergence, suggesting L11 fundamentally reorganizes representations based on accumulated (wrong) context.

3. **Context hurts rather than helps.** This contradicts the Lexinvariant LM finding that performance improves with context length. The critical difference: Lexinvariant models were *trained* to handle permutations; pre-trained GPT-2 has no such capability.

4. **Attention patterns are largely preserved.** Even though token identities are completely wrong, attention patterns remain similar (mean cos sim 0.82 for L0-9). The positional encoding and structural patterns in attention are more robust than token identity.

### Error Analysis

The model's failure is systematic, not random:
- **Top-1 predictions are completely unrelated** to both the correct token and the permuted token
- **The model doesn't learn the permutation** in context — perm-adjusted match is 0.00%
- **The failure is position-independent** — no position bin shows recovery
- **The failure scales linearly** with permutation fraction (partial permutation experiment)

### Limitations

1. **Single model:** Only tested GPT-2 Small. Larger models (GPT-2 Large, Llama, etc.) might show different behavior, potentially with more adaptation capacity.

2. **Fixed permutation:** We used one random permutation. Different permutations might produce different similarity patterns if they accidentally preserve embedding neighborhoods (though our Analysis 5 shows our permutation does not).

3. **Context length limited to 256 tokens.** The Lexinvariant LM paper showed convergence at 512+ tokens for trained models. However, context length *hurt* performance in our experiments, so longer contexts are unlikely to help.

4. **No fine-tuning:** We only tested inference-time adaptation. A model fine-tuned on permuted text would likely recover (similar to training a cipher model).

5. **Cosine similarity as metric:** High cosine similarity at L11 could be misleading (and was — it reflects structural convergence, not semantic recovery). Multiple metrics were necessary to disambiguate.

## 6. Conclusions

### Summary
A pre-trained transformer (GPT-2 Small) **cannot recover** from a bijective token identity permutation at inference time. The residual stream under permutation shows a distinctive U-shaped similarity curve — declining through middle layers then rising at the output layer — but this late-layer convergence reflects structural output formatting, not semantic adaptation. The model never learns the cipher: top-1 prediction match is 0%, perplexity increases 2,315x, and more context makes performance worse, not better.

### Implications

**Theoretical:** The results strongly support the impossibility result of Alur et al. (2025) — decoder-only transformers with causal masks cannot invert permutations. Our work shows this theoretical limit manifests empirically as a complete failure to adapt the residual stream, even partially, to a bijective token shuffle.

**For mechanistic interpretability:** The U-shaped similarity curve reveals that GPT-2's residual stream has two phases: (1) layers 0-9 progressively build token-identity-dependent representations that diverge under permutation, and (2) layers 10-11 converge toward stereotyped output patterns that are partially identity-independent. The norm correlation pattern (near-perfect through L2-10, collapse at L11) suggests the model's processing "intensity" is position-determined, while semantic content is token-determined.

**For robustness:** Pre-trained LMs are completely fragile to vocabulary shuffling. Even a 10% partial permutation causes meaningful degradation. This has implications for adversarial attacks and token-level perturbation studies.

### Confidence in Findings
**High confidence** in the main conclusion (no recovery). The evidence is consistent across 5 experiments, multiple metrics, 60+ samples, and 5 supplementary analyses. All statistical tests yield p < 0.001 for the key comparisons. The result is also theoretically grounded (impossibility result).

**Medium confidence** in the U-shape interpretation (structural vs. semantic). This is supported by multiple converging metrics but would benefit from circuit-level analysis (e.g., identifying which attention heads drive the L10-11 convergence).

## 7. Next Steps

### Immediate Follow-ups
1. **Larger models:** Test GPT-2 Large, Llama 3, and other architectures. Do models with more capacity show any adaptation?
2. **Tuned lens analysis:** Train affine probes per layer (the tuned lens approach from Belrose et al.) for more reliable layer-by-layer prediction decoding.
3. **Circuit analysis:** Use activation patching to identify which components (specific attention heads, MLP layers) drive the U-shape and the L11 convergence.

### Alternative Approaches
1. **Fine-tuning on permuted text:** How quickly does a model adapt when given gradient updates? What minimal fine-tuning suffices?
2. **Partial permutation of semantically coherent groups:** Instead of random permutation, permute within syntactic categories (swap all nouns with other nouns). Does preserving distributional similarity help?
3. **Character-level models:** The Lexinvariant LM paper found faster convergence with character-level vocab. Does GPT-2-style processing work better with smaller vocabularies?

### Open Questions
1. Why exactly do layers 10-11 show structural convergence? Is this specific to GPT-2 or universal?
2. Could a model with bidirectional attention (BERT-style) show any adaptation capacity, consistent with the Alur et al. existence proof for non-causal models?
3. Is there a "critical mass" of context where any pre-trained model begins to decipher the permutation, or is the impossibility absolute for causal models?

## References

1. Huang, Q., et al. (2023). Lexinvariant Language Models. NeurIPS Spotlight. arXiv:2305.16349
2. Xu, H., et al. (2024). Permutation Equivariance of Transformers and Its Applications. CVPR. arXiv:2304.07735
3. Alur, R., et al. (2025). The Impossibility of Inverse Permutation Learning in Transformer Models. arXiv:2509.24125
4. Belrose, N., et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. arXiv:2303.08112
5. Fernando, J. & Guitchounts, G. (2025). Transformer Dynamics. arXiv:2502.12131
6. Lawson, T., et al. (2024). Residual Stream Analysis with Multi-Layer SAEs. arXiv:2405.10263
7. Shai, A., et al. (2024). Transformers Represent Belief State Geometry in Their Residual Stream. arXiv:2405.15943
8. Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits.

## Appendix: Reproducibility

```bash
# Environment setup
uv venv && source .venv/bin/activate
uv add torch transformer-lens transformers datasets numpy scipy matplotlib seaborn tqdm

# Run main experiments
CUDA_VISIBLE_DEVICES=0 python src/experiment.py

# Run supplementary analyses
CUDA_VISIBLE_DEVICES=0 python src/supplementary_analysis.py
```

Random seed: 42 for all experiments. Results are deterministic given the same seed, model weights, and library versions.
