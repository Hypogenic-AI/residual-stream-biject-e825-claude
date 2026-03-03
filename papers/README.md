# Downloaded Papers

## Core Papers (Directly Relevant)

1. **Lexinvariant Language Models** (`2305.16349_lexinvariant_language_models.pdf`)
   - Authors: Huang, Zelikman, Chen, Wu, Valiant, Liang
   - Year: 2023 (NeurIPS Spotlight)
   - arXiv: 2305.16349
   - Why relevant: Demonstrates LMs can function without fixed token embeddings via random Gaussian vectors. Shows convergence of lexinvariant LMs to standard LMs. Proves bijective token invariance is achievable.

2. **Permutation Equivariance of Transformers** (`2304.07735_permutation_equivariance_transformers.pdf`)
   - Authors: Xu, Xiang, Ye, Yao, Chu, Li
   - Year: 2024 (CVPR)
   - arXiv: 2304.07735
   - Why relevant: Formal proofs that transformers are equivariant to inter-token and intra-token permutations. Establishes theoretical foundation for understanding how permutations affect the residual stream.

3. **The Impossibility of Inverse Permutation Learning** (`2509.24125_impossibility_inverse_permutation.pdf`)
   - Authors: Alur, Hays, Raghavan, Shah
   - Year: 2025
   - arXiv: 2509.24125
   - Why relevant: Proves decoder-only transformers cannot learn to invert permutations. Key negative result suggesting fundamental limits on recovery from bijective transformations.

## Methodology Papers (Tools and Techniques)

4. **Eliciting Latent Predictions with the Tuned Lens** (`2303.08112_tuned_lens.pdf`)
   - Authors: Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman, Steinhardt
   - Year: 2023
   - arXiv: 2303.08112
   - Why relevant: Layer-by-layer decoding of residual stream. Essential tool for analyzing what residual streams encode under transformation.

5. **Transformer Dynamics** (`2502.12131_transformer_dynamics.pdf`)
   - Authors: Fernando, Guitchounts
   - Year: 2025
   - arXiv: 2502.12131
   - Why relevant: Dynamical systems framework for residual stream analysis. Provides metrics (velocity, correlations, MI) we can apply to transformed inputs.

6. **Practical Review of Mechanistic Interpretability** (`practical_review_mechanistic_interpretability.pdf`)
   - Authors: Various
   - Year: 2024
   - arXiv: 2407.02646
   - Why relevant: Comprehensive survey of MI techniques applicable to our analysis.

7. **Residual Stream Analysis with Multi-Layer SAEs** (`residual_stream_analysis_multilayer_sae.pdf`)
   - Authors: Lawson, Farnik, Houghton, Aitchison
   - Year: 2024
   - Why relevant: Multi-layer sparse autoencoders for joint residual stream analysis across layers.

8. **Patchscopes: Inspecting Hidden Representations** (`patchscopes_hidden_representations.pdf`)
   - Authors: Ghandeharioun, Caciularu, Pearce, Dixon, Geva
   - Year: 2024
   - Why relevant: Unified framework for inspecting hidden representations using the model itself.

9. **Information Flow Routes** (`information_flow_routes.pdf`)
   - Authors: Ferrando, Voita
   - Year: 2024
   - Why relevant: Attribution-based analysis of information flow through transformer layers.

10. **How to Use Activation Patching** (`activation_patching_how_to.pdf`)
    - Authors: Heimersheim, Nanda
    - Year: 2024
    - Why relevant: Best practices for causal interventions on the residual stream.

11. **LogitLens4LLMs** (`logitlens4llms.pdf`)
    - Authors: Various
    - Year: 2025
    - Why relevant: Extended logit lens for modern architectures.

12. **Gemma Scope: Open SAEs** (`gemma_scope_sparse_autoencoders.pdf`)
    - Authors: Lieberum et al.
    - Year: 2024
    - Why relevant: Open SAEs trained on all layers/sub-layers for residual stream decomposition.

13. **GPT-2 LayerNorm Removal** (`gpt2_layernorm_removal.pdf`)
    - Authors: Heimersheim
    - Year: 2024
    - Why relevant: Simplified model for mechanistic interpretability without LayerNorm nonlinearity.

14. **Belief State Geometry in Residual Stream** (`belief_state_geometry_residual_stream.pdf`)
    - Authors: Shai, Marzen, Teixeira, Oldenziel, Riechers
    - Year: 2024
    - Why relevant: Linear belief state representations in residual stream.

15. **A Mathematical Framework for Transformer Circuits** (`mathematical_framework_transformer_circuits.pdf`)
    - Authors: Elhage et al.
    - Year: 2021
    - Why relevant: Foundational circuit-level understanding of transformers.

## Related Papers (Context and Background)

16. **Interchangeable Token Embeddings** (`2410.17161_interchangeable_token_embeddings.pdf`)
    - Authors: Isik, Cinbis, Gol
    - Year: 2024
    - Why relevant: Dual-part embeddings for handling interchangeable (permutable) tokens.

17. **Alice: Substitution Cipher Solver** (`2509.07282_alice_substitution_ciphers.pdf`)
    - Year: 2025
    - Why relevant: Bijective decoding architecture for cipher solving.

18. **Can Transformers Break Encryption via ICL?** (`2508.10235_transformers_break_encryption.pdf`)
    - Year: 2025
    - Why relevant: In-context learning of bijective character mappings.

19. **Probing Embedding Space via Minimal Perturbations** (`2506.18011_probing_embedding_space_perturbations.pdf`)
    - Year: 2025
    - Why relevant: How minimal token perturbations affect embedding space across layers.

20. **Higher Embedding Dimension for Sorting** (`2510.18315_higher_embedding_dimension_sorting.pdf`)
    - Year: 2025
    - Why relevant: How embedding dimension affects internal world model quality.

21. **Word Order Does Matter in Pre-trained LMs** (`word_order_does_matter_pretrained_lms.pdf`)
    - Authors: Sinha et al.
    - Year: 2021
    - Why relevant: Studies effect of word order shuffling on pre-trained models.

22. **The Reversal Curse** (`reversal_curse.pdf`)
    - Authors: Berglund et al.
    - Year: 2024
    - Why relevant: Directional failures in LMs — related to information flow constraints.

23. **Scrambled Text NLU** (`scrambled_text_nlu.pdf`)
    - Year: 2020
    - Why relevant: Robustness of NLU models to scrambled input text.
