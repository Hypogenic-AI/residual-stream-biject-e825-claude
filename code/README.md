# Cloned Repositories

## Repo 1: TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Mechanistic interpretability library for GPT-style language models. Provides direct access to residual stream activations at every layer via hooks and caching.
- **Location**: `code/TransformerLens/`
- **Key files**:
  - `transformer_lens/HookedTransformer.py` - Main model class with hooks
  - `transformer_lens/ActivationCache.py` - Cache for residual stream access (`decompose_resid()`, `accumulated_resid()`)
  - `transformer_lens/hook_points.py` - Hook infrastructure for intercepting activations
  - `transformer_lens/components/` - Embeddings, attention, MLP blocks
- **Installation**: `pip install transformer-lens`
- **Relevance**: Core infrastructure for accessing and analyzing residual streams under bijective token transformations. Use `model.run_with_cache()` to capture all activations before/after permutation.

## Repo 2: Tuned Lens
- **URL**: https://github.com/AlignmentResearch/tuned-lens
- **Purpose**: Trained affine probes to decode residual stream representations into vocabulary predictions at each layer. Refinement of the "logit lens" method.
- **Location**: `code/tuned-lens/`
- **Key files**:
  - `tuned_lens/nn/lenses.py` - `LogitLens` and `TunedLens` classes
  - `tuned_lens/nn/unembed.py` - Hidden state to logits conversion
  - `tuned_lens/scripts/train_loop.py` - Training tuned lens probes
- **Installation**: `pip install tuned-lens`
- **Relevance**: Allows decoding what information the residual stream encodes at each layer. Can compare decoded predictions under normal vs. bijectively-transformed inputs to understand how the transformation affects semantic content.

## Repo 3: Inverse Permutation Learning
- **URL**: https://github.com/johnchrishays/icl
- **Purpose**: Studies impossibility of inverse permutation learning in decoder-only transformers. Shows causal attention mask prevents learning to invert permutations, but scratch tokens can help.
- **Location**: `code/inverse-permutation-learning/`
- **Key files**:
  - `single_parent.py` - Main experiment script (JAX-based)
  - `catformer.py` - Custom transformer implementation
  - `problems.py` - Task definitions
- **Installation**: JAX + CUDA environment (see `environment.yml`)
- **Relevance**: Provides theoretical grounding for why decoder-only transformers may struggle with bijective token transformations. The impossibility result (Theorem 1) is directly relevant to our hypothesis about embedding misalignment.

## Repo 4: Permutation Equivariance
- **URL**: https://github.com/Doby-Xu/ST
- **Purpose**: Proves and demonstrates that transformers satisfy permutation equivariance for both inter-token (row) and intra-token (column) permutations.
- **Location**: `code/permutation-equivariance/`
- **Key files**:
  - `ShowCase/ViT/encrypt_ViT.py` - Model encryption via permutation
  - `ShowCase/Bert_GPT2/` - BERT/GPT-2 permutation experiments
  - `utilsenc.py` - Permutation utility functions
- **Installation**: `pip install torch torchvision timm einops transformers datasets`
- **Relevance**: Demonstrates that row permutation (inter-token) preserves weights exactly, while column permutation (intra-token, analogous to feature-dimension shuffling) yields equivalently learned weights W_{(C)} = P_C^{-1} W P_C. Directly relevant to understanding how the residual stream transforms under bijective mappings.
