# Basic Hybrid Search Sample

A single-page application demonstrating Japanese full-text, vector,
and hybrid search in the browser using laurus-wasm. Data persists
in OPFS across page reloads.

See the [examples README](../README.md) for the full list of samples
and the shared build instructions. The shortest path:

```bash
cd laurus-wasm
wasm-pack build --target web --dev
./scripts/postbuild.sh

# Make the UniDic zip available at examples/dict/lindera-unidic.zip,
# then serve from any HTTP server:
python3 -m http.server 8080

# Open http://localhost:8080/examples/basic/ in your browser.
```

## What this sample demonstrates

- An OPFS-persistent search index with `title` and `body` fields
  (data survives page reloads)
- Seeding 8 sample documents on first visit; skipping the seed when
  existing data is loaded from OPFS
- Real 384-dim semantic embeddings produced by Transformers.js
  (`paraphrase-multilingual-MiniLM-L12-v2`) via the callback embedder
- A search box that speaks the unified query DSL:
  - Lexical: `rust`, `title:wasm`, `"memory safety"`
  - Vector: `embedding:"how to make code faster"`,
    `embedding:python`
  - Hybrid: `rust embedding:"systems programming"`
- Adding new documents interactively
- Showing search results with relevance scores
- Logging all operations in the console panel

## Layout

This sample shares the dictionary loader, embedder, log helper, and
theme stylesheet with the other samples through `examples/shared/`.
The UniDic zip (~52 MB) is fetched from
`examples/dict/lindera-unidic.zip` (one level up from this sample)
so multiple samples can share the same cached dictionary.
