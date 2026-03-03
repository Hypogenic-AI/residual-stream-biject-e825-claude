# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "The Residual Stream After Bijective Token Transformation," including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 23

| Title | Authors | Year | File | Key Relevance |
|-------|---------|------|------|---------------|
| Lexinvariant Language Models | Huang et al. | 2023 | `papers/2305.16349_lexinvariant_language_models.pdf` | Core: LMs without fixed embeddings, bijective invariance |
| Permutation Equivariance of Transformers | Xu et al. | 2024 | `papers/2304.07735_permutation_equivariance_transformers.pdf` | Core: Formal equivariance proofs for row/column permutations |
| Impossibility of Inverse Permutation Learning | Alur et al. | 2025 | `papers/2509.24125_impossibility_inverse_permutation.pdf` | Core: Decoder-only transformers cannot invert permutations |
| Eliciting Latent Predictions with the Tuned Lens | Belrose et al. | 2023 | `papers/2303.08112_tuned_lens.pdf` | Methodology: Layer-by-layer residual stream decoding |
| Transformer Dynamics | Fernando & Guitchounts | 2025 | `papers/2502.12131_transformer_dynamics.pdf` | Methodology: Dynamical systems view of residual stream |
| Interchangeable Token Embeddings | Isik et al. | 2024 | `papers/2410.17161_interchangeable_token_embeddings.pdf` | Related: Dual-part embeddings for interchangeable tokens |
| Probing Embedding Space via Minimal Perturbations | (2025) | 2025 | `papers/2506.18011_probing_embedding_space_perturbations.pdf` | Related: How perturbations propagate across layers |
| Higher Embedding Dimension for Sorting | (2025) | 2025 | `papers/2510.18315_higher_embedding_dimension_sorting.pdf` | Related: Embedding dim affects world model quality |
| Alice: Bijective Substitution Cipher Solver | (2025) | 2025 | `papers/2509.07282_alice_substitution_ciphers.pdf` | Related: Bijective decoding architecture |
| Can Transformers Break Encryption via ICL? | (2025) | 2025 | `papers/2508.10235_transformers_break_encryption.pdf` | Related: ICL for bijective mappings |
| A Practical Review of Mechanistic Interpretability | (2024) | 2024 | `papers/practical_review_mechanistic_interpretability.pdf` | Background: Comprehensive MI survey |
| Residual Stream Analysis with Multi-Layer SAEs | Lawson et al. | 2024 | `papers/residual_stream_analysis_multilayer_sae.pdf` | Methodology: Multi-layer SAE for RS |
| Belief State Geometry in Residual Stream | Shai et al. | 2024 | `papers/belief_state_geometry_residual_stream.pdf` | Related: Linear belief states in RS |
| Patchscopes: Inspecting Hidden Representations | Ghandeharioun et al. | 2024 | `papers/patchscopes_hidden_representations.pdf` | Methodology: Unified inspection framework |
| Removing GPT-2 LayerNorm by Fine-Tuning | Heimersheim | 2024 | `papers/gpt2_layernorm_removal.pdf` | Related: Simplified model for MI |
| Gemma Scope: Open Sparse Autoencoders | Lieberum et al. | 2024 | `papers/gemma_scope_sparse_autoencoders.pdf` | Methodology: SAEs on residual stream |
| Information Flow Routes | Ferrando & Voita | 2024 | `papers/information_flow_routes.pdf` | Methodology: Attribution-based circuit analysis |
| How to Use Activation Patching | Heimersheim & Nanda | 2024 | `papers/activation_patching_how_to.pdf` | Methodology: Best practices for patching |
| LogitLens4LLMs | (2025) | 2025 | `papers/logitlens4llms.pdf` | Methodology: Extended logit lens for modern LLMs |
| A Mathematical Framework for Transformer Circuits | Elhage et al. | 2021 | `papers/mathematical_framework_transformer_circuits.pdf` | Foundation: Circuit-level understanding |
| Word Order Does Matter in Pre-trained LMs | Sinha et al. | 2021 | `papers/word_order_does_matter_pretrained_lms.pdf` | Related: Effect of word order shuffling |
| The Reversal Curse | Berglund et al. | 2024 | `papers/reversal_curse.pdf` | Related: Directional knowledge failures |
| Scrambled Text NLU | (2020) | 2020 | `papers/scrambled_text_nlu.pdf` | Related: Model robustness to text scrambling |

See `papers/README.md` for detailed descriptions.

## Datasets
Total datasets downloaded: 1 (+ 1 recommended for streaming)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WikiText-2-raw-v1 | HuggingFace | 36.7K train / 3.7K val / 4.4K test | LM / RS analysis | `datasets/wikitext-2-raw-v1/` | Primary prototyping dataset |
| The Pile (subset) | HuggingFace (streaming) | Configurable | LM / RS analysis | Download on demand | Used in Lexinvariant LMs paper |

See `datasets/README.md` for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Residual stream access & hooks | `code/TransformerLens/` | Core infrastructure |
| Tuned Lens | github.com/AlignmentResearch/tuned-lens | Layer-by-layer RS decoding | `code/tuned-lens/` | Trained affine probes |
| Inverse Permutation Learning | github.com/johnchrishays/icl | Impossibility results + experiments | `code/inverse-permutation-learning/` | JAX-based |
| Permutation Equivariance | github.com/Doby-Xu/ST | Permutation equivariance proofs + apps | `code/permutation-equivariance/` | PyTorch |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with diligent mode for three query sets: residual stream representations, bijective token mapping, and token embedding permutation
2. Web search for specific topics: logit/tuned lens, lexinvariant LMs, TransformerLens, substitution ciphers
3. Citation following from key papers (especially Lexinvariant LMs and Permutation Equivariance)
4. Searched for code repositories linked in papers and on GitHub

### Selection Criteria
- **Direct relevance**: Papers studying bijective/permutation transformations of tokens (Lexinvariant, Permutation Equivariance, Impossibility)
- **Methodological relevance**: Tools and techniques for analyzing residual streams (TransformerLens, Tuned Lens, Transformer Dynamics, SAEs)
- **Theoretical foundation**: Papers providing framework for understanding transformers (Mathematical Framework, Circuits)
- **Related phenomena**: Papers studying robustness to input perturbations (word order, scrambling, reversal curse)

### Challenges Encountered
- Lexinvariant Language Models code is not publicly available; will need to reimplement the random Gaussian embedding approach
- The Pile is very large (800GB); using streaming API for subset selection
- Inverse Permutation Learning code is JAX-based while TransformerLens is PyTorch; may need bridging
- No existing work directly studies pre-trained model residual streams under bijective token identity transformation

### Gaps and Workarounds
- **No lexinvariant LM code**: Reimplement using TransformerLens hooks to replace embeddings with random Gaussians
- **No pre-existing bijective transformation analysis pipeline**: Build from scratch using TransformerLens caching + tuned lens decoding
- **Limited compute**: Focus on GPT-2 small (124M params) and WikiText-2 for efficiency

## Recommendations for Experiment Design

Based on gathered resources:

1. **Primary dataset**: WikiText-2-raw-v1 for prototyping; Pile subsets for broader evaluation
2. **Primary model**: GPT-2 small via TransformerLens (well-studied, hooks available, manageable size)
3. **Baseline methods**:
   - Normal (unpermuted) input
   - Random permutation of token IDs
   - Partial permutation (varying fraction of vocabulary permuted)
4. **Evaluation approach**:
   - Cosine similarity of residual streams (normal vs. permuted) at each layer
   - Tuned lens predictions under permutation
   - Perplexity as function of context length under permutation
   - Dynamical systems metrics (velocity, correlation, MI) from Transformer Dynamics
5. **Code to adapt/reuse**:
   - TransformerLens for all residual stream access
   - Tuned lens for layer-by-layer decoding
   - Permutation utilities from Xu et al. for applying transformations
6. **Key experimental questions**:
   - Does the residual stream progressively "decode" the permutation across layers?
   - Is there a critical layer where the model "recognizes" the transformation?
   - How does vocabulary size affect adaptation (char-level vs. subword)?
   - Does the impossibility result manifest as a measurable ceiling on recovery?
