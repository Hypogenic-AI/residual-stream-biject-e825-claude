# Datasets

This directory contains datasets for the research project "The Residual Stream After Bijective Token Transformation."
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: WikiText-2-raw-v1

### Overview
- **Source**: HuggingFace `wikitext/wikitext-2-raw-v1`
- **Size**: train (36,718), validation (3,760), test (4,358 examples)
- **Format**: HuggingFace Dataset (text field)
- **Task**: Language modeling / residual stream analysis
- **License**: Creative Commons Attribution-ShareAlike License

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset.save_to_disk("datasets/wikitext-2-raw-v1")
```

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikitext-2-raw-v1")
```

### Why This Dataset
- Used in the Transformer Dynamics paper (Fernando & Guitchounts, 2025) for residual stream analysis
- Small enough for quick prototyping and iteration
- Standard benchmark for language modeling experiments
- Contains diverse Wikipedia text suitable for studying how models process natural language

### Notes
- Filter empty lines before use: many entries are empty strings or section headers
- For residual stream analysis, filter to sequences of 100-500 characters as in the Transformer Dynamics paper
- Can be used with GPT-2 tokenizer for compatibility with TransformerLens

## Dataset 2: The Pile (via EleutherAI)

### Overview
- **Source**: HuggingFace `EleutherAI/pile-deduped-pythia-random-sampled` or `monology/pile-uncopyrighted`
- **Size**: Very large (800GB full); we use small subsets
- **Format**: JSONL with text field
- **Task**: Language modeling with diverse text sources
- **License**: Various (depends on subset)

### Download Instructions

**Using HuggingFace (streaming for subset):**
```python
from datasets import load_dataset
# Stream and take first N examples
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
subset = list(dataset.take(10000))
```

### Why This Dataset
- Used in the Lexinvariant Language Models paper (Huang et al., 2023) as primary training data
- Contains 22 diverse high-quality text sources (Wikipedia, GitHub, StackExchange, etc.)
- Code subset is especially relevant: the Lexinvariant paper showed faster convergence on structured text
- Diverse text types allow testing bijective transformation effects across domains

### Notes
- Full dataset is very large; only download small subsets for experimentation
- The GitHub subset is particularly useful for testing structured text
- Consider streaming API for memory efficiency

## Recommended Usage for This Research

For experiments on residual streams after bijective token transformation:
1. **WikiText-2** for quick prototyping and comparison with Transformer Dynamics results
2. **Pile subsets** for broader evaluation across text domains
3. Focus on sequences of 128-512 tokens for manageable computation
4. Use GPT-2 tokenizer (50257 vocab) for TransformerLens compatibility
