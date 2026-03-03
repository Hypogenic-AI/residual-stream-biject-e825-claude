# The Residual Stream After Bijective Token Transformation

Investigating how a pre-trained transformer's internal representations respond when token identities are bijectively permuted at inference time. Does the residual stream eventually "look normal, modulo the shuffle"? Or does embedding misalignment prevent recovery?

## Key Findings

- **The model never recovers.** A pre-trained GPT-2 Small cannot adapt to a bijective token shuffle at inference time. Perplexity increases 2,315x and logit lens top-1 prediction match is <1% at all layers.
- **U-shaped similarity curve.** Cosine similarity between normal and permuted residual streams drops from 0.59 (embedding) to 0.47 (layer 6-7), then rises to 0.85 (layer 11). The late-layer rise is structural (output formatting), not semantic recovery.
- **More context hurts, not helps.** Contrary to the hypothesis, similarity *decreases* with more tokens (Spearman rho = -0.98, p < 0.001). The model accumulates wrong representations, not adaptation.
- **Attention patterns are robust.** Attention similarity remains high (0.78-0.86) through layers 0-9 despite completely wrong token identities, showing positional structure is more robust than semantic content.
- **Norm correlation collapse at final layer.** Activation magnitude patterns are preserved (r > 0.999) through layers 2-10, then collapse at layer 11 (r = 0.089), revealing a phase transition at the output layer.

## Project Structure

```
├── REPORT.md                    # Full research report with all results
├── planning.md                  # Experimental design and motivation
├── src/
│   ├── experiment.py            # Main experiments (5 experiments)
│   └── supplementary_analysis.py # Additional analyses (5 analyses)
├── results/
│   ├── metrics.json             # Main experiment results
│   ├── supplementary_metrics.json # Supplementary results
│   └── plots/                   # All visualizations
│       ├── summary_figure.png
│       ├── layer_cosine_similarity.png
│       ├── logit_lens_analysis.png
│       ├── context_length_heatmap.png
│       ├── partial_permutation.png
│       ├── attention_similarity.png
│       ├── l2_distance_layers.png
│       └── norm_correlation_layers.png
├── datasets/
│   └── wikitext-2-raw-v1/       # WikiText-2 dataset
├── papers/                      # 23 related papers (PDFs)
├── code/                        # Reference implementations
├── literature_review.md         # Literature review
└── resources.md                 # Resource catalog
```

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch transformer-lens 'transformers<4.50' datasets numpy scipy matplotlib seaborn tqdm

# Run experiments (requires GPU, ~5 minutes on RTX A6000)
USER=researcher CUDA_VISIBLE_DEVICES=0 python src/experiment.py
USER=researcher CUDA_VISIBLE_DEVICES=0 python src/supplementary_analysis.py
```

**Requirements:** Python 3.10+, NVIDIA GPU with 8+ GB VRAM, ~2 GB disk for model weights.

## Model & Data

- **Model:** GPT-2 Small (124M params, 12 layers, 768-dim) via TransformerLens
- **Data:** WikiText-2-raw-v1 validation set (1,646 filtered samples)
- **Permutation:** Fixed random bijection of full 50,257-token vocabulary (seed=42)

See [REPORT.md](REPORT.md) for the full analysis.
