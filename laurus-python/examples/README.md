# laurus-python Examples

This directory contains runnable examples for the `laurus` Python bindings.

## Prerequisites

- Rust toolchain (`rustup` — <https://rustup.rs>)
- Python 3.10+
- `maturin` build tool

```bash
pip install maturin
```

## Setup

All examples must be run from the `laurus-python/` directory after building
the native extension with `maturin develop`.

```bash
cd laurus-python

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install maturin
```

## Examples

### Basic examples (no extra dependencies)

Build once, then run any of the examples below:

```bash
maturin develop
```

| Example | Description |
| :--- | :--- |
| [quickstart.py](quickstart.py) | Minimal full-text search: index, search, update |
| [lexical_search.py](lexical_search.py) | All lexical query types: Term, Phrase, Fuzzy, Wildcard, NumericRange, Geo, Boolean, Span |
| [synonym_graph_filter.py](synonym_graph_filter.py) | Synonym expansion in the analysis pipeline |

```bash
python examples/quickstart.py
python examples/lexical_search.py
python examples/synonym_graph_filter.py
```

---

### Vector search — built-in embedder

Uses laurus's built-in `CandleBert` embedder (via [Candle](https://github.com/huggingface/candle)).
Text is embedded automatically by the Rust engine at index and query time — **no external
embedding library is needed**.

Build with the `embeddings-candle` feature:

```bash
maturin develop --features embeddings-candle
```

| Example | Description |
| :--- | :--- |
| [vector_search.py](vector_search.py) | Semantic similarity search with laurus's built-in BERT embedder |
| [hybrid_search.py](hybrid_search.py) | Hybrid lexical + vector search with RRF and WeightedSum fusion |

```bash
python examples/vector_search.py
python examples/hybrid_search.py
```

> **Note:** The first run downloads the model weights from Hugging Face Hub
> (`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB). Subsequent runs use
> the local cache.

---

### Vector search — external embedder

Uses [sentence-transformers](https://www.sbert.net/) to produce embeddings on
the Python side, then passes pre-computed vectors to laurus via `VectorQuery`.
Falls back to random vectors (no semantic meaning) if `sentence-transformers`
is not installed.

```bash
maturin develop
pip install sentence-transformers   # optional but recommended
```

| Example | Description |
| :--- | :--- |
| [external_embedder.py](external_embedder.py) | Vector and hybrid search with a user-managed Python embedder |

```bash
python examples/external_embedder.py
```

---

### OpenAI embeddings

Produces embeddings via the OpenAI API and passes them to laurus as
pre-computed vectors. Requires an OpenAI API key.

```bash
maturin develop
pip install openai
export OPENAI_API_KEY=your-api-key-here
```

| Example | Description |
| :--- | :--- |
| [search_with_openai.py](search_with_openai.py) | Vector search using OpenAI `text-embedding-3-small` |

```bash
python examples/search_with_openai.py
```

---

### Multimodal search

Searches across text and image data using CLIP embeddings produced on the
Python side. Falls back to random vectors if `torch`/`transformers` are not
installed.

```bash
maturin develop
pip install torch transformers Pillow   # optional but recommended
```

| Example | Description |
| :--- | :--- |
| [multimodal_search.py](multimodal_search.py) | Text-to-image and image-to-image search with CLIP |

```bash
python examples/multimodal_search.py
```

---

## Choosing an embedding approach

| Approach | Example | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Built-in embedder** | `vector_search.py`, `hybrid_search.py` | No Python embedding library needed; simpler code | Requires `embeddings-candle` feature at build time |
| **External embedder** | `external_embedder.py` | Full control over the model; any Python library | You manage embedding at index and query time |
| **OpenAI API** | `search_with_openai.py` | High-quality cloud embeddings | Requires API key and network access |
| **CLIP (multimodal)** | `multimodal_search.py` | Text + image search | Heavy dependencies (`torch`) |
