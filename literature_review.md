# Literature Review: The Residual Stream After Bijective Token Transformation

## Research Area Overview

This research investigates what happens to a transformer's internal representations — specifically the residual stream — when the input undergoes a bijective (one-to-one) transformation of token identities. The hypothesis is that applying a permutation to token-surface form mappings will initially confuse the model, but with sufficient exposure, the model may adapt so that its residual stream representations become similar to normal (modulo the shuffle), or alternatively, the model may never fully recover due to embedding misalignment.

This topic sits at the intersection of three active research areas:
1. **Mechanistic interpretability** of transformers, particularly residual stream analysis
2. **Token embedding permutation and invariance** properties of transformers
3. **Lexical flexibility and in-context adaptation** to novel token mappings

## Key Papers

### Paper 1: Lexinvariant Language Models
- **Authors**: Qian Huang, Eric Zelikman, Sarah Li Chen, Yuhuai Wu, Gregory Valiant, Percy Liang
- **Year**: 2023 (NeurIPS Spotlight)
- **Source**: arXiv:2305.16349
- **Key Contribution**: Demonstrates that a language model can function without any fixed token embeddings by using random Gaussian vectors per-sequence, achieving lexical invariance to all permutations.
- **Methodology**: Replace standard embeddings with per-sequence random Gaussian vectors E(x) ~ N(0, I_d). Same token → same random embedding within a sequence, different across sequences. Train decoder-only Transformer (150M params, 12 layers, 8 heads) on The Pile.
- **Datasets Used**: The Pile, Wiki-40B, GitHub subset of The Pile
- **Results**:
  - Perplexity gap between lexinvariant and standard LMs shrinks from 9x to <1x with 512 context tokens (char-level vocab)
  - Convergence rate: O((d/T)^{1/4}) where d=vocab size, T=context length
  - 99.6% accuracy recovering substitution cipher keys via probing
  - 4x improvement on synthetic reasoning tasks (LookUp, Permutation)
  - Code converges faster than natural language (more structured)
  - Semi-lexinvariance (embedding dropout p=0.2) improves BIG-bench by 25%
- **Code Available**: No (not publicly released)
- **Relevance to Our Research**: This is the most directly relevant paper. It shows that a model trained from scratch with bijective token permutations (random embeddings = implicit permutation each sequence) can recover near-normal language modeling performance. The key question for our research is: what does the residual stream look like in such a model? The lexinvariant LM implicitly performs Bayesian deciphering in-context, and the residual stream must encode this permutation inference process. Our research extends this by examining what happens to a *pre-trained* model's residual stream when exposed to bijective transformations.

### Paper 2: Permutation Equivariance of Transformers and Its Applications
- **Authors**: Hengyuan Xu, Liyao Xiang, Hangyu Ye, Dixi Yao, Pengzhi Chu, Baochun Li
- **Year**: 2024 (CVPR)
- **Source**: arXiv:2304.07735
- **Key Contribution**: Proves that vanilla Transformers satisfy permutation equivariance for both inter-token (row) and intra-token (column) permutations, in both forward and backward propagation.
- **Methodology**: Formal proofs of equivariance properties. Row permutation: Enc(P_R Z) = P_R Enc(Z), with identical learned weights W = W_{(R)}. Column permutation: weights become W_{(C)} = P_C^{-1} W P_C — equivalently learned but structurally different.
- **Key Theorems**:
  - Theorem 4.1: Row Permutation Forward Equivariance — Enc(P_R Z) = P_R Enc(Z)
  - Theorem 4.2: Row Permutation Backward Invariance — gradients unchanged under row permutation
  - Column permutation: forward equivariance holds for encoder but NOT for decoder with causal mask
- **Results**: Verified on ViT, BERT, GPT-2. Applications in privacy-enhancing split learning and model authorization.
- **Code Available**: Yes — https://github.com/Doby-Xu/ST
- **Relevance to Our Research**: Provides the theoretical foundation that inter-token permutation (shuffling token positions) preserves the residual stream up to the same permutation. However, our bijective transformation is different: we permute token *identities* (surface forms), not positions. This means the embedding layer maps to different vectors, which then flow through the residual stream differently. The equivariance results tell us what *would* be preserved (position shuffling) vs. what our transformation actually disrupts (embedding vectors).

### Paper 3: The Impossibility of Inverse Permutation Learning in Transformer Models
- **Authors**: Rohan Alur, Chris Hays, Manish Raghavan, Devavrat Shah
- **Year**: 2025
- **Source**: arXiv:2509.24125
- **Key Contribution**: Proves that decoder-only, attention-only transformers of *any depth* cannot learn to invert permutations, due to the causal attention mask preventing backward information flow.
- **Methodology**: Formal impossibility proof (Theorem 1) for disentangled transformer architecture. Complementary existence proofs: (1) removing causal masking enables it (Theorem 2), (2) padding with "scratch tokens" enables it (Theorem 3).
- **Key Results**:
  - Impossibility: No decoder-only transformer can output Y = P^{-1}Y_P to any block of the residual stream
  - With causal-mask-free attention: 2-layer transformer suffices
  - With scratch token padding: 2-layer decoder-only transformer suffices
  - Empirical validation: accuracy goes from random guessing → near-perfect when causal mask removed
- **Code Available**: Yes — https://github.com/johnchrishays/icl
- **Relevance to Our Research**: Directly relevant impossibility result. If we apply a bijective token permutation and ask the model to "undo" it internally, the causal mask fundamentally limits this. This supports the hypothesis that the model may *never fully recover* — the residual stream cannot globally reorganize information that flows only forward. However, with sufficient context (as the Lexinvariant paper shows), partial adaptation may be possible through in-context Bayesian inference.

### Paper 4: Transformer Dynamics — A Neuroscientific Approach to Interpretability
- **Authors**: Jesseba Fernando, Grigori Guitchounts
- **Year**: 2025
- **Source**: arXiv:2502.12131
- **Key Contribution**: Treats the residual stream as a dynamical system evolving across layers. Finds strong continuity of individual units across layers, acceleration and densification of activations, and attractor-like dynamics.
- **Methodology**: Analyze Llama 3.1 8B residual stream activations on WikiText-2. Capture pre-attention and pre-MLP activations at each layer for last token position. Compute correlations, cosine similarity, velocity, mutual information across layers.
- **Datasets Used**: WikiText-2-raw-v1
- **Key Results**:
  - Individual RS units maintain strong correlations across layers despite non-privileged basis
  - RS activations grow denser and accelerate over layers
  - Sharp decrease in mutual information in early layers (fundamental transformation)
  - Individual units trace unstable periodic orbits
  - Low-dimensional dynamics despite d=4096
- **Code Available**: Not found (analysis code not publicly released)
- **Relevance to Our Research**: Provides the analytical framework for studying residual stream dynamics. We can apply the same dynamical systems analysis (correlations, velocity, MI) to compare normal vs. bijectively-transformed inputs. If the bijective transformation disrupts the attractor-like dynamics in early layers, this would explain embedding misalignment effects.

### Paper 5: Eliciting Latent Predictions from Transformers with the Tuned Lens
- **Authors**: Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, Jacob Steinhardt
- **Year**: 2023
- **Source**: arXiv:2303.08112
- **Key Contribution**: Trains affine probes per layer to decode residual stream hidden states into vocabulary distributions, yielding more reliable predictions than the simpler "logit lens."
- **Methodology**: Train affine transformation per layer to map hidden states to logits. Tested on autoregressive LMs up to 20B parameters.
- **Key Insight**: Representations may be rotated, shifted, or stretched from layer to layer — the tuned lens accounts for this. "Rogue dimensions" in early layers cause logit lens to fail.
- **Code Available**: Yes — https://github.com/AlignmentResearch/tuned-lens
- **Relevance to Our Research**: The tuned lens provides a principled way to decode what the residual stream "knows" at each layer. Under bijective token transformation, we can use the tuned lens to track when/whether the model's internal predictions recover the correct (pre-transformation) token identities. If the residual stream adapts to the transformation, we should see the tuned lens predictions converge to the correct mapping at deeper layers.

### Paper 6: Interchangeable Token Embeddings for Extendable Vocabulary and Alpha-Equivalence
- **Authors**: I. Isik, R. G. Cinbis, Ebru Aydin Gol
- **Year**: 2024
- **Source**: arXiv:2410.17161
- **Key Contribution**: Proposes dual-part token embeddings with shared semantic component and randomized distinguishing component to handle interchangeable tokens (like bound variables in logic).
- **Methodology**: Split embedding into shared part (semantic consistency) and random part (token distinguishability). Introduces "alpha-covariance" metric for robustness to token renaming.
- **Code Available**: Yes — https://github.com/necrashter/interchangeable-token-embeddings
- **Relevance to Our Research**: Directly addresses the problem of tokens that should be semantically equivalent but have different surface forms. Their dual embedding strategy could inspire approaches to make models more robust to bijective token transformations.

### Paper 7: Residual Stream Analysis with Multi-Layer SAEs
- **Authors**: Tim Lawson, Lucy Farnik, Conor Houghton, Laurence Aitchison
- **Year**: 2024
- **Source**: arXiv:2405.10263
- **Key Contribution**: Trains a single SAE across all layers of the residual stream, finding that individual latents are often active at a single layer per token but different layers across tokens.
- **Methodology**: Multi-layer SAE (MLSAE) trained on residual stream activations from all transformer layers jointly.
- **Code Available**: Yes — https://github.com/tim-lawson/mlsae
- **Relevance to Our Research**: MLSAEs could reveal how bijective token transformations affect the distribution of feature activations across layers. If transformation causes features to activate at different layers, this indicates disrupted information flow.

### Paper 8: Transformers Represent Belief State Geometry in Their Residual Stream
- **Authors**: A. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Gietelink Oldenziel, P. Riechers
- **Year**: 2024
- **Source**: arXiv:2405.15943
- **Key Contribution**: Shows that belief states (from the theory of optimal prediction) are linearly represented in the residual stream, even with fractal geometry. Belief states contain information beyond local next-token prediction.
- **Relevance to Our Research**: If belief states are linearly encoded in the residual stream, a bijective token transformation would need to preserve or reconstruct these linear encodings. The fractal geometry of belief states may make this particularly challenging under permutation.

### Paper 9: Alice — An Interpretable Neural Architecture for Substitution Ciphers
- **Authors**: (arXiv:2509.07282, 2025)
- **Key Contribution**: Introduces Alice-Bijective model for cryptogram solving with bijective decoding. Single forward pass over ciphertext to decrypt.
- **Relevance**: Specialized architecture for bijective mapping problems; contrast with general-purpose transformers.

### Paper 10: Can Transformers Break Encryption Schemes via In-Context Learning?
- **Authors**: (arXiv:2508.10235, 2025)
- **Key Contribution**: Shows transformers can decipher substitution ciphers with high accuracy given (ciphertext, plaintext) pairs, demonstrating strong inductive bias toward recognizing bijective symbolic relationships.
- **Relevance**: Demonstrates that pre-trained transformers have inherent capacity to learn bijective mappings in-context, supporting the hypothesis that residual streams may adapt to such transformations.

## Common Methodologies

- **Residual stream caching and analysis**: TransformerLens hooks + ActivationCache (used across mech. interp. papers)
- **Logit lens / Tuned lens**: Decode residual stream to vocabulary distributions layer-by-layer
- **Activation patching**: Replace activations from clean/corrupt runs to identify causal components
- **Sparse autoencoders (SAEs)**: Decompose residual stream into interpretable features
- **Random Gaussian embeddings**: Replace fixed token embeddings to achieve lexinvariance (Huang et al.)
- **Permutation matrices**: Apply row (inter-token) or column (intra-token) permutations (Xu et al.)

## Standard Baselines

- **Standard language model**: Same architecture with normal (non-permuted) token embeddings
- **Logit lens**: Simple unembedding projection of residual stream (baseline for tuned lens)
- **Random baseline**: Random permutation with no adaptation
- **Character-level models**: Smaller vocabulary makes permutation effects more tractable

## Evaluation Metrics

- **Perplexity**: Standard language modeling metric; measures how well the model predicts next tokens
- **Cosine similarity**: Between residual streams of normal vs. transformed inputs
- **L1/L2 distance**: Between predicted distributions of normal vs. transformed models
- **Cipher key recovery accuracy**: Probing residual stream to read out inferred permutation
- **Mutual information**: Between layers of residual stream (from Transformer Dynamics)
- **Velocity**: Rate of change of residual stream across layers
- **Tuned lens KL divergence**: Divergence of layer-by-layer predictions from final prediction

## Datasets in the Literature

- **The Pile**: Used in Lexinvariant LM (primary), Gemma Scope. Large, diverse, 22 sources.
- **WikiText-2**: Used in Transformer Dynamics. Small, good for prototyping.
- **OpenWebText**: Used in GPT-2 no-LayerNorm paper.
- **BIG-bench**: Used for evaluating semi-lexinvariant models.
- **CIFAR-10/ImageNet**: Used in Permutation Equivariance paper (vision tasks).

## Gaps and Opportunities

1. **No direct study of pre-trained model residual streams under bijective token transformation**: The Lexinvariant paper trains from scratch; nobody has analyzed what happens to a pre-trained model's residual stream when tokens are permuted.

2. **Missing connection between permutation equivariance and embedding-level permutation**: Xu et al. study position/feature permutations, but not the case where the *embedding lookup table* itself is permuted (which changes what vectors enter the residual stream).

3. **No characterization of adaptation dynamics**: How quickly (if ever) does the residual stream "recover" under in-context exposure to permuted text? The Lexinvariant paper shows convergence rates but doesn't analyze the residual stream directly.

4. **Interaction with causal mask**: The impossibility result (Alur et al.) suggests decoder-only models fundamentally cannot invert permutations, but the Lexinvariant result shows they can *adapt* given enough context. The residual stream dynamics during this adaptation are unexplored.

5. **Layer-by-layer analysis of permutation adaptation**: Using tuned lens to track when the model "figures out" the mapping at different layers would be novel.

## Recommendations for Our Experiment

Based on the literature review:

### Recommended datasets
- **WikiText-2** for prototyping (small, used in Transformer Dynamics)
- **The Pile subsets** for broader evaluation (used in Lexinvariant LMs)

### Recommended models
- **GPT-2 small** (via TransformerLens) — well-studied, 12 layers, manageable size
- **Pythia models** (via TransformerLens) — suite of models at different scales on The Pile

### Recommended baselines
- Normal (non-permuted) input as control
- Random permutation with no adaptation period
- Partial permutation (only some tokens permuted) as gradient

### Recommended metrics
- Cosine similarity between normal and permuted residual streams at each layer
- Tuned lens / logit lens predictions at each layer under permutation
- Perplexity under permutation as function of context length
- Cipher key recovery accuracy via probing (following Lexinvariant methodology)

### Recommended tools
- **TransformerLens** for residual stream access and caching
- **Tuned lens** for layer-by-layer prediction decoding
- **Permutation equivariance utilities** from Xu et al. for applying transformations

### Methodological considerations
- Start with character-level or small vocabulary to make permutation effects tractable
- Use the dynamical systems framework from Transformer Dynamics (correlations, velocity, MI)
- Compare "train-time" bijective transformation (Lexinvariant approach) vs. "test-time" transformation (applying permutation to pre-trained model)
- The impossibility result suggests analyzing the residual stream at the *output* positions (after all context) rather than early positions
