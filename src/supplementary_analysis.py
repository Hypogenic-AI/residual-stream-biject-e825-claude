"""
Supplementary Analysis: Deeper investigation of residual stream behavior.

Focuses on:
1. Why does cosine similarity show a U-shape (drops mid-layers, recovers at L10-11)?
2. Why does norm correlation collapse at layer 11?
3. What does the model's final output actually look like under permutation?
4. Per-token analysis: do some token types adapt better?
"""

import json
import os
import sys
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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda:0"
RESULTS_DIR = Path("/workspaces/residual-stream-biject-e825-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"

# Load model
import transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()
N_LAYERS = model.cfg.n_layers
D_MODEL = model.cfg.d_model
VOCAB_SIZE = model.cfg.d_vocab

# Load data
dataset = load_from_disk("/workspaces/residual-stream-biject-e825-claude/datasets/wikitext-2-raw-v1")
val_texts = [t for t in dataset["validation"]["text"] if len(t.strip()) > 100]

# Create permutation
rng = np.random.RandomState(SEED)
perm = rng.permutation(VOCAB_SIZE)
inv_perm = np.argsort(perm)
perm_tensor = torch.tensor(perm, dtype=torch.long, device=DEVICE)
inv_perm_tensor = torch.tensor(inv_perm, dtype=torch.long, device=DEVICE)


# ─── Analysis 1: Attention Pattern Comparison ────────────────────────────────

print("=" * 70)
print("ANALYSIS 1: Attention Pattern Similarity (Normal vs Permuted)")
print("=" * 70)

N_SAMPLES = 30
MAX_LEN = 256
attn_similarities = {f"layer_{i}": [] for i in range(N_LAYERS)}

processed = 0
for text in tqdm(val_texts[:N_SAMPLES * 2], desc="Attention analysis"):
    tokens = model.to_tokens(text, prepend_bos=True)[:, :MAX_LEN]
    if tokens.shape[1] < 50:
        continue
    permuted_tokens = perm_tensor[tokens]

    with torch.no_grad():
        _, cache_n = model.run_with_cache(tokens, names_filter=lambda n: "attn.hook_pattern" in n)
        _, cache_p = model.run_with_cache(permuted_tokens, names_filter=lambda n: "attn.hook_pattern" in n)

    for i in range(N_LAYERS):
        key = f"blocks.{i}.attn.hook_pattern"
        # Attention patterns: [batch, heads, seq, seq]
        attn_n = cache_n[key].squeeze(0).cpu().float()  # [heads, seq, seq]
        attn_p = cache_p[key].squeeze(0).cpu().float()

        # Compute mean cosine sim across all head-position pairs
        # Flatten heads and query positions
        a = attn_n.reshape(-1, attn_n.shape[-1])  # [heads*seq, seq]
        b = attn_p.reshape(-1, attn_p.shape[-1])
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
        attn_similarities[f"layer_{i}"].append(cos)

    processed += 1
    if processed >= N_SAMPLES:
        break

print(f"\nAttention Pattern Cosine Similarity (Normal vs Permuted):")
for i in range(N_LAYERS):
    key = f"layer_{i}"
    m = np.mean(attn_similarities[key])
    se = np.std(attn_similarities[key]) / np.sqrt(len(attn_similarities[key]))
    print(f"  Layer {i:2d}: {m:.4f} ± {se:.4f}")

# Plot attention similarity
fig, ax = plt.subplots(figsize=(10, 5))
means = [np.mean(attn_similarities[f"layer_{i}"]) for i in range(N_LAYERS)]
ses = [np.std(attn_similarities[f"layer_{i}"]) / np.sqrt(len(attn_similarities[f"layer_{i}"])) for i in range(N_LAYERS)]
ax.errorbar(range(N_LAYERS), means, yerr=ses, marker='o', capsize=4, linewidth=2)
ax.set_xlabel("Layer")
ax.set_ylabel("Attention Pattern Cosine Similarity")
ax.set_title("Attention Pattern Similarity: Normal vs Permuted Inputs")
ax.set_xticks(range(N_LAYERS))
plt.tight_layout()
plt.savefig(PLOTS_DIR / "attention_similarity.png", dpi=150)
plt.close()
print("Saved: attention_similarity.png")


# ─── Analysis 2: Perplexity Under Permutation ────────────────────────────────

print("\n" + "=" * 70)
print("ANALYSIS 2: Perplexity Under Permutation")
print("=" * 70)

N_SAMPLES_PPL = 40

losses_normal = []
losses_permuted = []

processed = 0
for text in tqdm(val_texts[:N_SAMPLES_PPL * 2], desc="Perplexity comparison"):
    tokens = model.to_tokens(text, prepend_bos=True)[:, :MAX_LEN]
    if tokens.shape[1] < 50:
        continue
    permuted_tokens = perm_tensor[tokens]

    with torch.no_grad():
        logits_n = model(tokens)  # [1, seq, vocab]
        logits_p = model(permuted_tokens)

        # Compute cross-entropy loss (next token prediction)
        # Shift: logits[:-1] should predict tokens[1:]
        loss_n = torch.nn.functional.cross_entropy(
            logits_n[0, :-1].float(), tokens[0, 1:], reduction='mean'
        ).item()

        # For permuted: the "correct" next token in the permuted space is perm[original_next_token]
        # But the model doesn't know the permutation, so we measure:
        # (a) Loss predicting the permuted tokens (model's local coherence)
        loss_p_local = torch.nn.functional.cross_entropy(
            logits_p[0, :-1].float(), permuted_tokens[0, 1:], reduction='mean'
        ).item()

        losses_normal.append(loss_n)
        losses_permuted.append(loss_p_local)

    processed += 1
    if processed >= N_SAMPLES_PPL:
        break

ppl_normal = np.exp(np.mean(losses_normal))
ppl_permuted = np.exp(np.mean(losses_permuted))
print(f"\nPerplexity (normal input, predicting normal): {ppl_normal:.1f}")
print(f"Perplexity (permuted input, predicting permuted): {ppl_permuted:.1f}")
print(f"Ratio: {ppl_permuted / ppl_normal:.1f}x")


# ─── Analysis 3: Layer 11 Investigation ──────────────────────────────────────

print("\n" + "=" * 70)
print("ANALYSIS 3: Layer 11 Norm Collapse Investigation")
print("=" * 70)

# The norm correlation collapses at layer 11. Let's investigate why.
# Hypothesis: The final layer dramatically reorganizes to predict the next token,
# and the norm pattern (which positions have large activations) changes fundamentally
# when the token identities are wrong.

N_SAMPLES_L11 = 20

# Compare the distribution of norms at layer 10 vs layer 11
norms_l10_n = []
norms_l10_p = []
norms_l11_n = []
norms_l11_p = []

processed = 0
for text in tqdm(val_texts[:N_SAMPLES_L11 * 2], desc="Layer 11 investigation"):
    tokens = model.to_tokens(text, prepend_bos=True)[:, :MAX_LEN]
    if tokens.shape[1] < 50:
        continue
    permuted_tokens = perm_tensor[tokens]

    with torch.no_grad():
        _, cache_n = model.run_with_cache(tokens, names_filter=lambda n: "resid_post" in n and ("10" in n or "11" in n))
        _, cache_p = model.run_with_cache(permuted_tokens, names_filter=lambda n: "resid_post" in n and ("10" in n or "11" in n))

    norms_l10_n.append(torch.norm(cache_n["blocks.10.hook_resid_post"].float(), dim=-1).squeeze().cpu().numpy())
    norms_l10_p.append(torch.norm(cache_p["blocks.10.hook_resid_post"].float(), dim=-1).squeeze().cpu().numpy())
    norms_l11_n.append(torch.norm(cache_n["blocks.11.hook_resid_post"].float(), dim=-1).squeeze().cpu().numpy())
    norms_l11_p.append(torch.norm(cache_p["blocks.11.hook_resid_post"].float(), dim=-1).squeeze().cpu().numpy())

    processed += 1
    if processed >= N_SAMPLES_L11:
        break

# Compare mean norms
mean_norm_l10_n = np.mean([np.mean(n) for n in norms_l10_n])
mean_norm_l10_p = np.mean([np.mean(n) for n in norms_l10_p])
mean_norm_l11_n = np.mean([np.mean(n) for n in norms_l11_n])
mean_norm_l11_p = np.mean([np.mean(n) for n in norms_l11_p])

print(f"\nMean Norm at Layer 10 (normal): {mean_norm_l10_n:.1f}")
print(f"Mean Norm at Layer 10 (permuted): {mean_norm_l10_p:.1f}")
print(f"Mean Norm at Layer 11 (normal): {mean_norm_l11_n:.1f}")
print(f"Mean Norm at Layer 11 (permuted): {mean_norm_l11_p:.1f}")
print(f"\nRatio L11/L10 (normal): {mean_norm_l11_n / mean_norm_l10_n:.2f}")
print(f"Ratio L11/L10 (permuted): {mean_norm_l11_p / mean_norm_l10_p:.2f}")

# Compare coefficient of variation (spread of norms)
cv_l11_n = np.mean([np.std(n) / np.mean(n) for n in norms_l11_n])
cv_l11_p = np.mean([np.std(n) / np.mean(n) for n in norms_l11_p])
print(f"\nCoefficient of variation at L11 (normal): {cv_l11_n:.4f}")
print(f"Coefficient of variation at L11 (permuted): {cv_l11_p:.4f}")


# ─── Analysis 4: Partial Permutation Gradient ────────────────────────────────

print("\n" + "=" * 70)
print("ANALYSIS 4: Partial Permutation - Varying Fraction of Shuffled Tokens")
print("=" * 70)

fractions = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_SAMPLES_PARTIAL = 20

partial_results = {f: {f"layer_{i}": [] for i in range(N_LAYERS)} for f in fractions}
partial_results_embed = {f: [] for f in fractions}

for frac in fractions:
    # Create partial permutation: only permute `frac` fraction of vocabulary
    partial_perm = np.arange(VOCAB_SIZE)
    n_to_perm = int(VOCAB_SIZE * frac)
    if n_to_perm > 1:
        # Select random subset of tokens to permute
        perm_indices = rng.choice(VOCAB_SIZE, n_to_perm, replace=False)
        shuffled = rng.permutation(perm_indices)
        partial_perm[perm_indices] = shuffled
    partial_perm_t = torch.tensor(partial_perm, dtype=torch.long, device=DEVICE)

    processed = 0
    for text in val_texts[:N_SAMPLES_PARTIAL * 2]:
        tokens = model.to_tokens(text, prepend_bos=True)[:, :MAX_LEN]
        if tokens.shape[1] < 50:
            continue
        partial_tokens = partial_perm_t[tokens]

        with torch.no_grad():
            _, cache_n = model.run_with_cache(tokens, names_filter=lambda n: "resid_post" in n)
            _, cache_p = model.run_with_cache(partial_tokens, names_filter=lambda n: "resid_post" in n)

        for i in range(N_LAYERS):
            key = f"blocks.{i}.hook_resid_post"
            a = cache_n[key].squeeze(0).cpu().float()
            b = cache_p[key].squeeze(0).cpu().float()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
            partial_results[frac][f"layer_{i}"].append(cos)

        processed += 1
        if processed >= N_SAMPLES_PARTIAL:
            break

    layer_means = [np.mean(partial_results[frac][f"layer_{i}"]) for i in range(N_LAYERS)]
    print(f"  Frac={frac:.1f}: L0={layer_means[0]:.3f}, L5={layer_means[5]:.3f}, L11={layer_means[11]:.3f}")

# Plot partial permutation results
fig, ax = plt.subplots(figsize=(12, 7))
for frac in fractions:
    means = [np.mean(partial_results[frac][f"layer_{i}"]) for i in range(N_LAYERS)]
    ax.plot(range(N_LAYERS), means, marker='o', linewidth=2, label=f'{int(frac*100)}% permuted')
ax.set_xlabel("Layer")
ax.set_ylabel("Cosine Similarity (vs Normal)")
ax.set_title("Effect of Partial Permutation on Residual Stream Similarity")
ax.set_xticks(range(N_LAYERS))
ax.legend(title="Fraction Permuted")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "partial_permutation.png", dpi=150)
plt.close()
print("Saved: partial_permutation.png")


# ─── Analysis 5: Embedding Space Similarity Structure ─────────────────────────

print("\n" + "=" * 70)
print("ANALYSIS 5: Does the Permutation Preserve Embedding Neighborhoods?")
print("=" * 70)

# Check if the permutation accidentally maps similar tokens to similar tokens
# (which would explain some residual stream similarity)
W_E = model.W_E.cpu().float()  # [vocab, d_model]

# Sample 1000 tokens and check if their neighbors are preserved
sample_tokens = torch.randint(0, VOCAB_SIZE, (1000,))
original_embeddings = W_E[sample_tokens]
permuted_embeddings = W_E[perm_tensor[sample_tokens].cpu()]

# Compute cosine similarity between original and permuted embeddings
cos_emb = torch.nn.functional.cosine_similarity(original_embeddings, permuted_embeddings, dim=-1)
print(f"Mean cosine sim between E(t) and E(f(t)): {cos_emb.detach().mean():.4f} ± {cos_emb.detach().std():.4f}")
print(f"  (This measures whether the permutation maps tokens to 'nearby' tokens in embedding space)")

# Compare to random pairs
random_pairs = torch.randint(0, VOCAB_SIZE, (1000,))
random_embeddings = W_E[random_pairs]
cos_rand = torch.nn.functional.cosine_similarity(original_embeddings, random_embeddings, dim=-1)
print(f"Mean cosine sim between E(t) and E(random): {cos_rand.detach().mean():.4f} ± {cos_rand.detach().std():.4f}")

# The permutation should NOT preserve neighborhoods (it's random)
t_stat, p_val = stats.ttest_ind(cos_emb.detach().numpy(), cos_rand.detach().numpy())
print(f"t-test (permuted vs random embedding similarity): t={t_stat:.2f}, p={p_val:.4f}")


# ─── Save Supplementary Results ──────────────────────────────────────────────

supp_results = {
    "attention_similarity": {
        f"layer_{i}": {
            "mean": float(np.mean(attn_similarities[f"layer_{i}"])),
            "se": float(np.std(attn_similarities[f"layer_{i}"]) / np.sqrt(len(attn_similarities[f"layer_{i}"])))
        } for i in range(N_LAYERS)
    },
    "perplexity": {
        "normal": float(ppl_normal),
        "permuted": float(ppl_permuted),
        "ratio": float(ppl_permuted / ppl_normal),
    },
    "layer_11_norms": {
        "mean_norm_l10_normal": float(mean_norm_l10_n),
        "mean_norm_l10_permuted": float(mean_norm_l10_p),
        "mean_norm_l11_normal": float(mean_norm_l11_n),
        "mean_norm_l11_permuted": float(mean_norm_l11_p),
    },
    "partial_permutation": {
        str(frac): {
            f"layer_{i}": float(np.mean(partial_results[frac][f"layer_{i}"]))
            for i in range(N_LAYERS)
        } for frac in fractions
    },
    "embedding_neighborhood": {
        "perm_cos_sim_mean": float(cos_emb.detach().mean()),
        "random_cos_sim_mean": float(cos_rand.detach().mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
    }
}

with open(RESULTS_DIR / "supplementary_metrics.json", "w") as f:
    json.dump(supp_results, f, indent=2)

print("\n" + "=" * 70)
print("ALL SUPPLEMENTARY ANALYSES COMPLETE")
print("=" * 70)
