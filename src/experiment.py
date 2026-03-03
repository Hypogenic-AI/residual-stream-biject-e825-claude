"""
Experiment: The Residual Stream After Bijective Token Transformation

Investigates how a pre-trained transformer's residual stream responds when
token IDs are bijectively permuted at inference time. Measures layer-by-layer
similarity, logit lens predictions, and context-length adaptation effects.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_from_disk
from scipy import stats
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("/workspaces/residual-stream-biject-e825-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Device: {DEVICE}")


# ─── Load Model ──────────────────────────────────────────────────────────────

print("\n=== Loading GPT-2 Small via TransformerLens ===")
import transformer_lens

model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()

N_LAYERS = model.cfg.n_layers  # 12
D_MODEL = model.cfg.d_model    # 768
VOCAB_SIZE = model.cfg.d_vocab  # 50257

print(f"Model: GPT-2 Small")
print(f"Layers: {N_LAYERS}, d_model: {D_MODEL}, vocab: {VOCAB_SIZE}")


# ─── Load Dataset ────────────────────────────────────────────────────────────

print("\n=== Loading WikiText-2 ===")
dataset = load_from_disk("/workspaces/residual-stream-biject-e825-claude/datasets/wikitext-2-raw-v1")
# Get validation set texts, filter out empty lines
val_texts = [t for t in dataset["validation"]["text"] if len(t.strip()) > 100]
print(f"Validation samples (>100 chars): {len(val_texts)}")


# ─── Create Bijective Permutation ────────────────────────────────────────────

print("\n=== Creating Bijective Token Permutation ===")
rng = np.random.RandomState(SEED)
perm = rng.permutation(VOCAB_SIZE)  # perm[original_id] = new_id
inv_perm = np.argsort(perm)         # inv_perm[new_id] = original_id

# Verify bijectivity
assert len(set(perm)) == VOCAB_SIZE, "Permutation is not bijective"
assert np.all(inv_perm[perm] == np.arange(VOCAB_SIZE)), "Inverse check failed"

# Count fixed points (tokens that map to themselves)
fixed_points = np.sum(perm == np.arange(VOCAB_SIZE))
print(f"Permutation created: {VOCAB_SIZE} tokens shuffled")
print(f"Fixed points: {fixed_points} ({100*fixed_points/VOCAB_SIZE:.2f}%)")

perm_tensor = torch.tensor(perm, dtype=torch.long, device=DEVICE)
inv_perm_tensor = torch.tensor(inv_perm, dtype=torch.long, device=DEVICE)


# ─── Helper Functions ────────────────────────────────────────────────────────

def tokenize_and_permute(text, max_len=512):
    """Tokenize text and create both normal and permuted token sequences."""
    tokens = model.to_tokens(text, prepend_bos=True)  # [1, seq_len]
    tokens = tokens[:, :max_len]
    permuted_tokens = perm_tensor[tokens]  # Apply bijection: f(token_id)
    return tokens, permuted_tokens


def get_residual_streams(tokens):
    """Run forward pass and return residual stream at all layers.
    Returns dict with keys 'embed', 'layer_0', ..., 'layer_11' (after each block).
    Shape of each: [batch, seq_len, d_model]
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: "resid" in name)

    streams = {}
    # After embedding (before any transformer block)
    streams["embed"] = cache["blocks.0.hook_resid_pre"].cpu()
    # After each transformer block
    for i in range(N_LAYERS):
        streams[f"layer_{i}"] = cache[f"blocks.{i}.hook_resid_post"].cpu()
    return streams


def cosine_sim_per_position(stream_a, stream_b):
    """Compute cosine similarity between two residual streams at each token position.
    stream_a, stream_b: [1, seq_len, d_model]
    Returns: [seq_len] array of cosine similarities.
    """
    a = stream_a.squeeze(0).float()  # [seq_len, d_model]
    b = stream_b.squeeze(0).float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return cos.numpy()


def logit_lens(residual_stream):
    """Apply the unembedding matrix to get logit predictions at each layer.
    residual_stream: [1, seq_len, d_model]
    Returns: logits [seq_len, vocab_size]
    """
    with torch.no_grad():
        # Apply LayerNorm then unembed
        normed = model.ln_final(residual_stream.to(DEVICE))
        logits = model.unembed(normed)  # [1, seq_len, vocab_size]
    return logits.squeeze(0).cpu()


# ─── Experiment 1: Layer-wise Residual Stream Similarity ─────────────────────

print("\n" + "="*70)
print("EXPERIMENT 1: Layer-wise Residual Stream Similarity")
print("="*70)

N_SAMPLES = 60
MAX_LEN = 256

# Storage for results
all_cosine_sims = {key: [] for key in ["embed"] + [f"layer_{i}" for i in range(N_LAYERS)]}
all_l2_dists = {key: [] for key in ["embed"] + [f"layer_{i}" for i in range(N_LAYERS)]}

# Also track per-position similarities for context length analysis
position_sims = {key: [] for key in ["embed"] + [f"layer_{i}" for i in range(N_LAYERS)]}

processed = 0
for idx, text in enumerate(tqdm(val_texts[:N_SAMPLES*2], desc="Exp 1: Residual stream similarity")):
    tokens, permuted_tokens = tokenize_and_permute(text, max_len=MAX_LEN)
    if tokens.shape[1] < 50:
        continue

    streams_normal = get_residual_streams(tokens)
    streams_permuted = get_residual_streams(permuted_tokens)

    for key in all_cosine_sims:
        cos_sims = cosine_sim_per_position(streams_normal[key], streams_permuted[key])
        all_cosine_sims[key].append(np.mean(cos_sims))
        all_l2_dists[key].append(
            torch.norm(streams_normal[key].float() - streams_permuted[key].float(), dim=-1).mean().item()
        )
        position_sims[key].append(cos_sims)

    processed += 1
    if processed >= N_SAMPLES:
        break

print(f"\nProcessed {processed} samples")

# Compute statistics
layer_names = ["embed"] + [f"layer_{i}" for i in range(N_LAYERS)]
cos_means = [np.mean(all_cosine_sims[k]) for k in layer_names]
cos_stds = [np.std(all_cosine_sims[k]) / np.sqrt(len(all_cosine_sims[k])) for k in layer_names]
l2_means = [np.mean(all_l2_dists[k]) for k in layer_names]
l2_stds = [np.std(all_l2_dists[k]) / np.sqrt(len(all_l2_dists[k])) for k in layer_names]

print("\nLayer-wise Cosine Similarity (normal vs. permuted):")
print(f"{'Layer':<12} {'Mean Cos Sim':>12} {'± SE':>10} {'Mean L2 Dist':>12} {'± SE':>10}")
print("-" * 58)
for i, key in enumerate(layer_names):
    print(f"{key:<12} {cos_means[i]:>12.4f} {cos_stds[i]:>10.4f} {l2_means[i]:>12.2f} {l2_stds[i]:>10.2f}")


# ─── Experiment 2: Logit Lens Analysis ───────────────────────────────────────

print("\n" + "="*70)
print("EXPERIMENT 2: Logit Lens Analysis")
print("="*70)

N_SAMPLES_LOGIT = 40

# Track: at each layer, what fraction of top-1 predictions match between normal and permuted?
top1_match_rates = {key: [] for key in layer_names}
# Track: after applying inverse permutation to permuted logits, what fraction match?
top1_adjusted_match_rates = {key: [] for key in layer_names}
# Track: KL divergence between logit distributions
kl_divs = {key: [] for key in layer_names}
kl_divs_adjusted = {key: [] for key in layer_names}

processed = 0
for idx, text in enumerate(tqdm(val_texts[:N_SAMPLES_LOGIT*2], desc="Exp 2: Logit lens")):
    tokens, permuted_tokens = tokenize_and_permute(text, max_len=MAX_LEN)
    if tokens.shape[1] < 50:
        continue

    streams_normal = get_residual_streams(tokens)
    streams_permuted = get_residual_streams(permuted_tokens)

    for key in layer_names:
        logits_normal = logit_lens(streams_normal[key])   # [seq_len, vocab]
        logits_permuted = logit_lens(streams_permuted[key])  # [seq_len, vocab]

        # Top-1 match (raw)
        top1_normal = logits_normal.argmax(dim=-1)
        top1_permuted = logits_permuted.argmax(dim=-1)
        raw_match = (top1_normal == top1_permuted).float().mean().item()
        top1_match_rates[key].append(raw_match)

        # Apply inverse permutation to permuted logits:
        # If the model learned to predict perm(correct_token), then
        # inv_perm applied to the logits should recover the correct distribution
        logits_permuted_adjusted = logits_permuted[:, inv_perm_tensor.cpu()]
        top1_adjusted = logits_permuted_adjusted.argmax(dim=-1)
        adjusted_match = (top1_normal == top1_adjusted).float().mean().item()
        top1_adjusted_match_rates[key].append(adjusted_match)

        # KL divergence (raw)
        probs_normal = torch.softmax(logits_normal, dim=-1)
        probs_permuted = torch.softmax(logits_permuted, dim=-1)
        probs_permuted_adjusted = torch.softmax(logits_permuted_adjusted, dim=-1)

        # Clamp for numerical stability
        eps = 1e-10
        kl_raw = (probs_normal * (torch.log(probs_normal + eps) - torch.log(probs_permuted + eps))).sum(dim=-1).mean().item()
        kl_adj = (probs_normal * (torch.log(probs_normal + eps) - torch.log(probs_permuted_adjusted + eps))).sum(dim=-1).mean().item()
        kl_divs[key].append(kl_raw)
        kl_divs_adjusted[key].append(kl_adj)

    processed += 1
    if processed >= N_SAMPLES_LOGIT:
        break

print(f"\nProcessed {processed} samples")

print("\nLogit Lens: Top-1 Match Rates and KL Divergences")
print(f"{'Layer':<12} {'Raw Match':>10} {'Adj Match':>10} {'KL Raw':>10} {'KL Adj':>10}")
print("-" * 54)
for key in layer_names:
    raw = np.mean(top1_match_rates[key])
    adj = np.mean(top1_adjusted_match_rates[key])
    kl_r = np.mean(kl_divs[key])
    kl_a = np.mean(kl_divs_adjusted[key])
    print(f"{key:<12} {raw:>10.4f} {adj:>10.4f} {kl_r:>10.2f} {kl_a:>10.2f}")


# ─── Experiment 3: Context Length Effect ─────────────────────────────────────

print("\n" + "="*70)
print("EXPERIMENT 3: Context Length Effect on Residual Stream Similarity")
print("="*70)

# For each layer, compute how cosine similarity changes with token position
# Use position_sims collected in Experiment 1

# Define position bins
max_positions = min(MAX_LEN, max(len(s) for s in position_sims["embed"]))
n_bins = 8
bin_edges = np.linspace(0, max_positions, n_bins + 1, dtype=int)

context_results = {}
for key in layer_names:
    bin_means = []
    bin_ses = []
    for b in range(n_bins):
        start, end = bin_edges[b], bin_edges[b + 1]
        vals = []
        for sims in position_sims[key]:
            segment = sims[start:end]
            if len(segment) > 0:
                vals.append(np.mean(segment))
        if vals:
            bin_means.append(np.mean(vals))
            bin_ses.append(np.std(vals) / np.sqrt(len(vals)))
        else:
            bin_means.append(np.nan)
            bin_ses.append(np.nan)
    context_results[key] = (bin_means, bin_ses)

print("\nCosine Similarity by Token Position Bin:")
bin_labels = [f"{bin_edges[b]}-{bin_edges[b+1]}" for b in range(n_bins)]
print(f"{'Layer':<12}", "  ".join(f"{l:>8}" for l in bin_labels))
print("-" * (12 + 10 * n_bins))
for key in ["embed", "layer_0", "layer_3", "layer_6", "layer_9", "layer_11"]:
    means, _ = context_results[key]
    print(f"{key:<12}", "  ".join(f"{m:>8.4f}" for m in means))


# ─── Experiment 4: Permutation-Adjusted Residual Stream Similarity ───────────

print("\n" + "="*70)
print("EXPERIMENT 4: Permutation-Adjusted Residual Stream Similarity")
print("="*70)
print("Testing whether residual stream looks 'normal modulo the shuffle'")

# The idea: if the model's residual stream adapts to the permutation,
# then the *embedding-level* representation should be related by the permutation.
# We can test this by checking if the permuted residual stream, when projected
# back to vocabulary space (logit lens), gives predictions that are related
# to the normal predictions via the permutation.
#
# Additionally, we test if the residual stream vectors themselves have a
# relationship beyond random, even if not identical.

# Compute correlation between embedding vectors: if token A maps to token B under perm,
# does the residual stream at position where A was input (in normal) relate to
# where B was input (in permuted)?

# For a cleaner test: compare the *norms* and *directions* of residual streams
print("\nComparing residual stream norms (normal vs permuted):")
norm_correlations = {key: [] for key in layer_names}

processed = 0
for idx, text in enumerate(tqdm(val_texts[:50], desc="Exp 4: Perm-adjusted analysis")):
    tokens, permuted_tokens = tokenize_and_permute(text, max_len=MAX_LEN)
    if tokens.shape[1] < 50:
        continue

    streams_normal = get_residual_streams(tokens)
    streams_permuted = get_residual_streams(permuted_tokens)

    for key in layer_names:
        norms_n = torch.norm(streams_normal[key].float(), dim=-1).squeeze().numpy()
        norms_p = torch.norm(streams_permuted[key].float(), dim=-1).squeeze().numpy()
        corr, _ = stats.pearsonr(norms_n, norms_p)
        norm_correlations[key].append(corr)

    processed += 1
    if processed >= 40:
        break

print(f"\nNorm correlation (Pearson r) across layers:")
for key in layer_names:
    r = np.mean(norm_correlations[key])
    se = np.std(norm_correlations[key]) / np.sqrt(len(norm_correlations[key]))
    print(f"  {key:<12}: r = {r:.4f} ± {se:.4f}")


# ─── Experiment 5: Random Embedding Baseline ─────────────────────────────────

print("\n" + "="*70)
print("EXPERIMENT 5: Random Embedding Baseline Comparison")
print("="*70)
print("Comparing: normal vs permuted vs random-embedding inputs")

N_SAMPLES_RAND = 30
random_cos_sims = {key: [] for key in layer_names}

for idx, text in enumerate(tqdm(val_texts[:N_SAMPLES_RAND*2], desc="Exp 5: Random embedding baseline")):
    tokens, permuted_tokens = tokenize_and_permute(text, max_len=MAX_LEN)
    if tokens.shape[1] < 50:
        continue

    # Create random token IDs (not bijective, just random)
    random_tokens = torch.randint(0, VOCAB_SIZE, tokens.shape, device=DEVICE)

    streams_normal = get_residual_streams(tokens)
    streams_random = get_residual_streams(random_tokens)

    for key in layer_names:
        cos_sims = cosine_sim_per_position(streams_normal[key], streams_random[key])
        random_cos_sims[key].append(np.mean(cos_sims))

    if len(random_cos_sims["embed"]) >= N_SAMPLES_RAND:
        break

print("\nComparison: Normal vs Permuted vs Random")
print(f"{'Layer':<12} {'vs Permuted':>12} {'vs Random':>12} {'Difference':>12}")
print("-" * 50)
for key in layer_names:
    perm_sim = np.mean(all_cosine_sims[key])
    rand_sim = np.mean(random_cos_sims[key])
    diff = perm_sim - rand_sim
    print(f"{key:<12} {perm_sim:>12.4f} {rand_sim:>12.4f} {diff:>12.4f}")


# ─── Statistical Tests ───────────────────────────────────────────────────────

print("\n" + "="*70)
print("STATISTICAL TESTS")
print("="*70)

# Test 1: Is permuted similarity significantly different from random at each layer?
print("\nWilcoxon signed-rank test: Permuted vs Random similarity")
print(f"{'Layer':<12} {'Perm Mean':>10} {'Rand Mean':>10} {'W-stat':>10} {'p-value':>12} {'Sig?':>6}")
print("-" * 62)

min_len = min(len(all_cosine_sims["embed"]), len(random_cos_sims["embed"]))
stat_results = {}
for key in layer_names:
    perm_vals = np.array(all_cosine_sims[key][:min_len])
    rand_vals = np.array(random_cos_sims[key][:min_len])
    try:
        w_stat, p_val = stats.wilcoxon(perm_vals, rand_vals)
    except ValueError:
        w_stat, p_val = np.nan, np.nan
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    stat_results[key] = {"w_stat": float(w_stat) if not np.isnan(w_stat) else None,
                          "p_value": float(p_val) if not np.isnan(p_val) else None,
                          "significant": sig}
    print(f"{key:<12} {np.mean(perm_vals):>10.4f} {np.mean(rand_vals):>10.4f} {w_stat:>10.1f} {p_val:>12.2e} {sig:>6}")

# Test 2: Does similarity increase with layer depth? (Spearman correlation)
print("\nSpearman correlation: Layer index vs cosine similarity")
layer_indices = list(range(len(layer_names)))
perm_means = [np.mean(all_cosine_sims[k]) for k in layer_names]
rho, p = stats.spearmanr(layer_indices, perm_means)
print(f"  Permuted: rho = {rho:.4f}, p = {p:.4e}")

rand_means = [np.mean(random_cos_sims[k]) for k in layer_names]
rho_r, p_r = stats.spearmanr(layer_indices, rand_means)
print(f"  Random:   rho = {rho_r:.4f}, p = {p_r:.4e}")

# Test 3: Context length effect - is there a trend within each layer?
print("\nSpearman correlation: Token position vs cosine similarity (per layer)")
for key in ["embed", "layer_0", "layer_5", "layer_11"]:
    means, _ = context_results[key]
    bin_centers = [(bin_edges[b] + bin_edges[b+1]) / 2 for b in range(n_bins)]
    valid = [(c, m) for c, m in zip(bin_centers, means) if not np.isnan(m)]
    if len(valid) >= 3:
        cs, ms = zip(*valid)
        r, p = stats.spearmanr(cs, ms)
        print(f"  {key:<12}: rho = {r:.4f}, p = {p:.4e}")


# ─── Save Results ────────────────────────────────────────────────────────────

print("\n=== Saving Results ===")

results = {
    "config": {
        "seed": SEED,
        "model": "gpt2",
        "n_layers": N_LAYERS,
        "d_model": D_MODEL,
        "vocab_size": VOCAB_SIZE,
        "max_len": MAX_LEN,
        "n_samples_exp1": processed,
        "fixed_points_in_perm": int(fixed_points),
        "device": DEVICE,
    },
    "experiment_1_cosine_similarity": {
        key: {
            "mean": float(np.mean(all_cosine_sims[key])),
            "std": float(np.std(all_cosine_sims[key])),
            "se": float(np.std(all_cosine_sims[key]) / np.sqrt(len(all_cosine_sims[key]))),
            "values": [float(v) for v in all_cosine_sims[key]],
        }
        for key in layer_names
    },
    "experiment_1_l2_distance": {
        key: {
            "mean": float(np.mean(all_l2_dists[key])),
            "std": float(np.std(all_l2_dists[key])),
            "se": float(np.std(all_l2_dists[key]) / np.sqrt(len(all_l2_dists[key]))),
        }
        for key in layer_names
    },
    "experiment_2_logit_lens": {
        key: {
            "top1_match_raw": float(np.mean(top1_match_rates[key])),
            "top1_match_adjusted": float(np.mean(top1_adjusted_match_rates[key])),
            "kl_div_raw": float(np.mean(kl_divs[key])),
            "kl_div_adjusted": float(np.mean(kl_divs_adjusted[key])),
        }
        for key in layer_names
    },
    "experiment_3_context_length": {
        key: {
            "bin_means": [float(m) for m in context_results[key][0]],
            "bin_ses": [float(s) for s in context_results[key][1]],
            "bin_edges": [int(e) for e in bin_edges],
        }
        for key in layer_names
    },
    "experiment_4_norm_correlation": {
        key: {
            "mean_r": float(np.mean(norm_correlations[key])),
            "se_r": float(np.std(norm_correlations[key]) / np.sqrt(len(norm_correlations[key]))),
        }
        for key in layer_names
    },
    "experiment_5_random_baseline": {
        key: {
            "mean_cos_sim": float(np.mean(random_cos_sims[key])),
            "std_cos_sim": float(np.std(random_cos_sims[key])),
        }
        for key in layer_names
    },
    "statistical_tests": {
        "permuted_vs_random": stat_results,
        "layer_trend_permuted": {"rho": float(rho), "p": float(p)},
        "layer_trend_random": {"rho": float(rho_r), "p": float(p_r)},
    },
}

with open(RESULTS_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {RESULTS_DIR / 'metrics.json'}")


# ─── Generate Visualizations ─────────────────────────────────────────────────

print("\n=== Generating Visualizations ===")
sns.set_theme(style="whitegrid", font_scale=1.2)

# Plot 1: Layer-wise cosine similarity (normal vs permuted vs random)
fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(layer_names))
labels = ["Emb"] + [f"L{i}" for i in range(N_LAYERS)]

perm_means = [np.mean(all_cosine_sims[k]) for k in layer_names]
perm_ses = [np.std(all_cosine_sims[k]) / np.sqrt(len(all_cosine_sims[k])) for k in layer_names]
rand_means_plot = [np.mean(random_cos_sims[k]) for k in layer_names]
rand_ses_plot = [np.std(random_cos_sims[k]) / np.sqrt(len(random_cos_sims[k])) for k in layer_names]

ax.errorbar(x, perm_means, yerr=perm_ses, marker='o', capsize=4, linewidth=2, label='Bijective Permutation')
ax.errorbar(x, rand_means_plot, yerr=rand_ses_plot, marker='s', capsize=4, linewidth=2, label='Random Tokens', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Residual Stream Position")
ax.set_ylabel("Cosine Similarity (vs Normal)")
ax.set_title("Residual Stream Similarity: Normal vs Permuted/Random Inputs")
ax.legend()
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "layer_cosine_similarity.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: layer_cosine_similarity.png")

# Plot 2: Logit lens top-1 match rates
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

raw_matches = [np.mean(top1_match_rates[k]) for k in layer_names]
adj_matches = [np.mean(top1_adjusted_match_rates[k]) for k in layer_names]

ax1.plot(x, raw_matches, marker='o', linewidth=2, label='Raw Match')
ax1.plot(x, adj_matches, marker='s', linewidth=2, label='Perm-Adjusted Match')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel("Residual Stream Position")
ax1.set_ylabel("Top-1 Match Rate")
ax1.set_title("Logit Lens: Top-1 Prediction Match (Normal vs Permuted)")
ax1.legend()

kl_raw_vals = [np.mean(kl_divs[k]) for k in layer_names]
kl_adj_vals = [np.mean(kl_divs_adjusted[k]) for k in layer_names]

ax2.plot(x, kl_raw_vals, marker='o', linewidth=2, label='Raw KL Divergence')
ax2.plot(x, kl_adj_vals, marker='s', linewidth=2, label='Perm-Adjusted KL')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_xlabel("Residual Stream Position")
ax2.set_ylabel("KL Divergence")
ax2.set_title("Logit Lens: KL Divergence (Normal vs Permuted)")
ax2.legend()
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "logit_lens_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: logit_lens_analysis.png")

# Plot 3: Context length effect heatmap
fig, ax = plt.subplots(figsize=(14, 8))
heatmap_data = []
heatmap_labels = []
for key in layer_names:
    means, _ = context_results[key]
    heatmap_data.append(means)
    heatmap_labels.append(key.replace("layer_", "L").replace("embed", "Emb"))

heatmap_array = np.array(heatmap_data)
sns.heatmap(heatmap_array, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
            xticklabels=bin_labels, yticklabels=heatmap_labels,
            vmin=-0.1, vmax=max(0.3, np.nanmax(heatmap_array)))
ax.set_xlabel("Token Position Bin")
ax.set_ylabel("Layer")
ax.set_title("Cosine Similarity (Normal vs Permuted) by Position and Layer")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "context_length_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: context_length_heatmap.png")

# Plot 4: L2 distance across layers
fig, ax = plt.subplots(figsize=(12, 6))
l2_means_plot = [np.mean(all_l2_dists[k]) for k in layer_names]
l2_ses_plot = [np.std(all_l2_dists[k]) / np.sqrt(len(all_l2_dists[k])) for k in layer_names]
ax.errorbar(x, l2_means_plot, yerr=l2_ses_plot, marker='o', capsize=4, linewidth=2, color='tab:red')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Residual Stream Position")
ax.set_ylabel("L2 Distance")
ax.set_title("Residual Stream L2 Distance: Normal vs Permuted Inputs")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "l2_distance_layers.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: l2_distance_layers.png")

# Plot 5: Norm correlation across layers
fig, ax = plt.subplots(figsize=(12, 6))
norm_r_means = [np.mean(norm_correlations[k]) for k in layer_names]
norm_r_ses = [np.std(norm_correlations[k]) / np.sqrt(len(norm_correlations[k])) for k in layer_names]
ax.errorbar(x, norm_r_means, yerr=norm_r_ses, marker='o', capsize=4, linewidth=2, color='tab:green')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Residual Stream Position")
ax.set_ylabel("Pearson r (Norm Correlation)")
ax.set_title("Residual Stream Norm Correlation: Normal vs Permuted")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "norm_correlation_layers.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: norm_correlation_layers.png")

# Plot 6: Combined summary figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Cosine similarity
ax = axes[0, 0]
ax.errorbar(x, perm_means, yerr=perm_ses, marker='o', capsize=3, linewidth=2, label='Bijective Perm')
ax.errorbar(x, rand_means_plot, yerr=rand_ses_plot, marker='s', capsize=3, linewidth=2, label='Random', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Cosine Similarity")
ax.set_title("A. Residual Stream Similarity vs Normal")
ax.legend(fontsize=9)

# Panel B: Logit lens
ax = axes[0, 1]
ax.plot(x, raw_matches, marker='o', linewidth=2, label='Raw')
ax.plot(x, adj_matches, marker='s', linewidth=2, label='Perm-Adjusted')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Top-1 Match Rate")
ax.set_title("B. Logit Lens Prediction Match")
ax.legend(fontsize=9)

# Panel C: L2 distance
ax = axes[1, 0]
ax.errorbar(x, l2_means_plot, yerr=l2_ses_plot, marker='o', capsize=3, linewidth=2, color='tab:red')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("L2 Distance")
ax.set_title("C. Residual Stream L2 Distance (Normal vs Perm)")

# Panel D: Norm correlation
ax = axes[1, 1]
ax.errorbar(x, norm_r_means, yerr=norm_r_ses, marker='o', capsize=3, linewidth=2, color='tab:green')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Pearson r")
ax.set_title("D. Norm Correlation (Normal vs Perm)")
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

plt.suptitle("The Residual Stream After Bijective Token Transformation\n(GPT-2 Small, WikiText-2)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "summary_figure.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: summary_figure.png")


print("\n" + "="*70)
print("ALL EXPERIMENTS COMPLETE")
print("="*70)
print(f"Results: {RESULTS_DIR / 'metrics.json'}")
print(f"Plots:   {PLOTS_DIR}/")
