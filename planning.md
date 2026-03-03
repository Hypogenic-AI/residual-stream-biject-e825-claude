# Research Plan: The Residual Stream After Bijective Token Transformation

## Motivation & Novelty Assessment

### Why This Research Matters
Language models encode knowledge through learned token embeddings that map surface forms to semantic vectors. Understanding how these models handle systematic disruption of this mapping — via a bijective permutation of the vocabulary — reveals fundamental properties of how transformers build internal representations. This has implications for model robustness, security (cipher attacks), and mechanistic interpretability.

### Gap in Existing Work
Based on the literature review:
- **Lexinvariant LMs** (Huang et al., 2023) showed models *trained from scratch* with random embeddings can recover near-normal performance, but didn't analyze the residual stream of *pre-trained* models.
- **Permutation Equivariance** (Xu et al., 2024) proved equivariance for position/feature permutations but not for embedding-level identity permutations.
- **Impossibility results** (Alur et al., 2025) showed decoder-only transformers can't invert permutations, but this hasn't been linked to residual stream dynamics.
- **No one has studied what happens to a pre-trained model's residual stream when token identities are bijectively shuffled at inference time.**

### Our Novel Contribution
We directly measure how a pre-trained transformer's residual stream responds to bijective token identity permutation at inference time. We track layer-by-layer whether the model's internal representations "recover" toward normal processing as context accumulates, or whether embedding misalignment permanently distorts the residual stream.

### Experiment Justification
- **Experiment 1 (Residual Stream Similarity)**: Measures cosine similarity between normal and permuted residual streams at each layer — directly answers whether the residual stream "looks normal modulo the shuffle."
- **Experiment 2 (Logit Lens Analysis)**: Tracks what the model "thinks" at each layer — reveals whether internal predictions converge toward correct tokens despite permuted input.
- **Experiment 3 (Context Length Effect)**: Tests whether more context helps the model adapt — addresses the "after a lot of text" part of the hypothesis.
- **Experiment 4 (Permutation-Adjusted Similarity)**: Compares residual streams after accounting for the known permutation — tests the "modulo the shuffle" hypothesis directly.

## Research Question
When a pre-trained language model receives input where token IDs have been bijectively permuted, does the residual stream eventually resemble normal processing (modulo the permutation), or does embedding misalignment permanently prevent recovery?

## Hypothesis Decomposition
1. **H1 (Early confusion)**: At early token positions and early layers, the residual stream under permutation will be highly dissimilar to normal processing.
2. **H2 (Layer-wise recovery)**: Deeper layers may show increasing similarity as the model's attention patterns extract statistical regularities.
3. **H3 (Context accumulation)**: With more context tokens, the model may progressively adapt, reducing the gap between normal and permuted residual streams.
4. **H4 (Embedding ceiling)**: Due to the embedding misalignment (wrong vectors enter the residual stream), there may be a fundamental ceiling on recovery — the residual stream can never fully normalize.
5. **H5 (Permutation-adjusted similarity)**: If we account for the known permutation when comparing residual streams, similarity should be higher than raw comparison, suggesting the model preserves some structure "modulo the shuffle."

## Proposed Methodology

### Approach
Use GPT-2 small (124M params, 12 layers) via TransformerLens to:
1. Process normal text and cache all residual stream activations
2. Apply a fixed bijective permutation to token IDs and process the permuted text
3. Compare residual streams across conditions using multiple metrics
4. Vary context length to study adaptation dynamics

### Experimental Steps

1. **Setup**: Load GPT-2 small via TransformerLens, load WikiText-2
2. **Permutation Generation**: Create a fixed random bijective permutation of the full vocabulary (50257 tokens)
3. **Residual Stream Caching**: For each text sample:
   - Tokenize normally → cache residual stream at all layers
   - Apply permutation to token IDs → cache residual stream at all layers
4. **Experiment 1 — Layer-wise Similarity**: Compute cosine similarity between normal and permuted residual streams at each of the 13 residual stream positions (after embedding + after each of 12 layers)
5. **Experiment 2 — Logit Lens**: Apply the unembedding matrix to residual streams at each layer; compare top-k predictions between normal and permuted runs
6. **Experiment 3 — Context Length**: Measure how similarity changes as a function of token position within a sequence (early tokens vs. late tokens)
7. **Experiment 4 — Permutation-Adjusted Comparison**: For the permuted run, apply the inverse permutation to the logit lens predictions before comparing — tests whether the model's predictions are correct "modulo the shuffle"

### Baselines
- **Normal (unpermuted)**: Control condition
- **Random embeddings**: Replace token embeddings with random vectors (no bijective structure)
- **Identity permutation**: Sanity check that unpermuted input gives identical results

### Evaluation Metrics
- Cosine similarity between residual streams (layer-by-layer)
- Top-1 and Top-5 accuracy of logit lens predictions
- KL divergence between logit distributions
- Perplexity under permutation as function of context length
- Permutation-adjusted metrics (applying inverse permutation before comparison)

### Statistical Analysis Plan
- Report means ± standard errors across multiple text samples (N≥50)
- Use paired t-tests or Wilcoxon signed-rank tests for layer comparisons
- Bootstrap confidence intervals for key metrics
- Multiple comparison correction via Bonferroni where needed

## Expected Outcomes
- **H1 supported**: Low cosine similarity at early layers/positions → the model is confused
- **H2 partially supported**: Some increase in similarity at deeper layers, but likely modest
- **H3 partially supported**: Slight improvement with more context, but probably not dramatic for a pre-trained model (unlike trained-from-scratch lexinvariant models)
- **H4 supported**: A fundamental ceiling on recovery because the embedding vectors are wrong — the model never "remaps" them
- **H5 supported**: Permutation-adjusted similarity should be higher than raw similarity, showing the model preserves some structural relationships

## Timeline and Milestones
- Phase 1 (Planning): 15 min ✓
- Phase 2 (Environment + Data): 15 min
- Phase 3 (Implementation): 60 min
- Phase 4 (Experiments): 60 min
- Phase 5 (Analysis + Visualization): 30 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- TransformerLens memory usage with full residual stream caching → use batch processing
- GPT-2's vocabulary size (50257) makes permutation computationally trivial but analysis rich
- Need to carefully handle special tokens (BOS, EOS, PAD) in permutation
- Logit lens may be unreliable at early layers (known issue) → use as relative comparison

## Success Criteria
1. Clear quantitative evidence for or against residual stream recovery
2. Layer-by-layer and position-by-position analysis with statistical significance
3. Visualizations showing the trajectory of similarity across layers and context
4. Comparison of raw vs. permutation-adjusted metrics to test "modulo the shuffle" hypothesis
