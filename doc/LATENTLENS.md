# LatentLens

[LatentLens](https://github.com/McGill-NLP/latentlens) interprets what hidden representations in LLMs and VLMs encode by finding their nearest neighbors in a bank of **contextual text embeddings**. Unlike Logit Lens (which projects to vocabulary space), LatentLens compares against embeddings *in context*, yielding more interpretable results. It works with any HuggingFace model and requires no training.

Paper: [Krojer et al., 2026](https://arxiv.org/abs/2602.00462) -- *LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs*

## Setup

```bash
bash env_setup.sh lens       # one-time: creates .venv-lens/
source login_setup.sh lens   # daily login
```

## Use in This Repository

The LatentLens experiments run in the `lens` environment (`uv run python -m experiments.<name> ...`), with shared helpers in `vlm_icl/lens.py`:

| Experiment | Purpose |
|---|---|
| `experiments.lens_visualize` | Decode text from query-image visual tokens per layer |
| `experiments.lens_ablate_tokens` | Identify distractor tokens, then compare baseline / ablate / 8-shot |
| `experiments.lens_divert` | Attention diversion (ICL minus 0-shot attention-density deltas) across layers |
| `experiments.lens_rewrite` | Compare decoded tokens at distractor positions under baseline vs 8-shot |

`--latentlens_index` takes a Hub repo id from the table below or a local index directory.

## Pre-built Indices

Pre-built contextual embedding indices are available on HuggingFace Hub:

| Model | HuggingFace Repo | Layers | Size |
|---|---|---|---|
| Qwen2.5-VL-7B | `McGill-NLP/contextual_embeddings-qwen2.5-vl-7b` | 1,2,4,8,16,24,26,27 | 28 GB |
| Qwen3-VL-8B | `McGill-NLP/contextual_embeddings-qwen3-vl-8b` | 1,2,4,8,16,24,34,35 | 32 GB |
| Llama-3.1-8B | `McGill-NLP/contextual_embeddings-llama3.1-8b` | 1,2,4,8,16,24,30,31 | 32 GB |
| Qwen2.5-7B | `McGill-NLP/contextual_embeddings-qwen2.5-7b` | 1,2,4,8,16,24,26,27 | 28 GB |
| Gemma-2-9B | `McGill-NLP/contextual_embeddings-gemma2-9b` | 1,2,4,8,16,24,40,41 | 15 GB |
| Molmo2-8B | `McGill-NLP/contextual_embeddings-molmo2-8b` | 1,2,4,8,16,24,34,35 | 32 GB |

Load only specific layers to save memory:
```python
index = latentlens.ContextualIndex.from_pretrained(
    "McGill-NLP/contextual_embeddings-qwen2.5-vl-7b", layers=[1, 27]
)
```

## Quick Start: LLM

```python
import latentlens

model, tokenizer = latentlens.load_model("Qwen/Qwen2.5-7B")
index = latentlens.ContextualIndex.from_pretrained(
    "McGill-NLP/contextual_embeddings-qwen2.5-7b"
)

inputs = tokenizer("a photo of a dog", return_tensors="pt").to("cuda")
hidden_states = latentlens.get_hidden_states(model, inputs["input_ids"])
# hidden_states[0] = input embeddings, [1..N] = transformer layer outputs

results = index.search(hidden_states[27].squeeze(0), top_k=5)
for i, neighbors in enumerate(results):
    token = tokenizer.decode(inputs["input_ids"][0, i])
    nn = neighbors[0]
    print(f"{token:>10} -> {nn.token_str!r} (sim={nn.similarity:.2f}, layer={nn.contextual_layer})")
```

## Quick Start: VLM (Qwen2.5-VL)

```python
import torch, latentlens
from transformers import AutoProcessor
from PIL import Image

model, tokenizer = latentlens.load_model(
    "Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.float16
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
index = latentlens.ContextualIndex.from_pretrained(
    "McGill-NLP/contextual_embeddings-qwen2.5-vl-7b"
)

image = Image.open("example.jpg")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Describe this image."},
]}]
text_prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    images=[image], text=[text_prompt], return_tensors="pt", padding=True
).to("cuda")

hidden_states = latentlens.get_hidden_states(model, **inputs)
results = index.search(hidden_states[27].squeeze(0), top_k=5)
```

## Building a Custom Index

```python
import latentlens

# From a text file (one sentence per line)
index = latentlens.build_index("Qwen/Qwen2.5-7B", corpus="my_corpus.txt")
index.save("my_index/")

# From a Python list
index = latentlens.build_index(
    "Qwen/Qwen2.5-7B",
    corpus=["a red car on a highway", "a black cat on a sofa"],
    layers=[1, 8, 16, 27],       # specific layers (default: auto_layers())
    max_contexts_per_token=50,    # soft cap per token string
    batch_size=32,
    dtype=torch.bfloat16,
)
```

A bundled `concepts.txt` corpus (~117k sentences from ~23k concepts) is included in the package.

## Core API Reference

| Function / Class | Description |
|---|---|
| `latentlens.load_model(name, dtype=float32)` | Load HuggingFace model + tokenizer |
| `latentlens.get_hidden_states(model, input_ids, ...)` | Forward pass returning all layer hidden states |
| `latentlens.build_index(model_name, corpus, ...)` | Build contextual embedding index |
| `latentlens.auto_layers(num_hidden_layers)` | Auto-select layers: `[1,2,4,8,16,24,n-2,n-1]` |
| `ContextualIndex.from_pretrained(repo_id, layers=None)` | Load index from HuggingFace Hub |
| `ContextualIndex.from_directory(path)` | Load index from local directory |
| `index.search(query, top_k=5, layers=None)` | Search for nearest neighbors |
| `index.save(path)` | Save index to disk |

Each search result is a `Neighbor` dataclass:
```python
Neighbor(
    token_str="dog",          # decoded token string
    similarity=0.92,          # cosine similarity
    caption="a brown dog",    # source sentence from corpus
    position=2,               # position in source sentence
    token_id=5765,            # vocabulary token ID
    contextual_layer=27,      # which layer this embedding came from
)
```

## Tips

- Use `bfloat16` over `float16` to avoid NaN/Inf issues with large activations.
- Load only the layers you need (`layers=[1, 27]`) to reduce memory usage.
- The index can be moved to GPU with `index.to("cuda")` for faster search.
- Any HuggingFace causal LM works; if not in `SUPPORTED_MODELS`, `auto_layers()` picks sensible defaults.
