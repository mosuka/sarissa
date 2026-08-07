# Examples

This directory contains example scripts for indexing and searching datasets with laurus.

## Sample list

| Sample | Dataset | Highlights |
| --- | --- | --- |
| [movies](movies/) | Meilisearch movies (~32,000 records) | English lexical full-text search + CLIP multimodal vector search |
| [aozora](aozora/) | Aozora Bunko (public-domain Japanese literature) | Japanese morphological full-text search via Lindera + IPADIC, per-field analyzers |

## Common prerequisites

- A Rust toolchain (`cargo build` must work at the repository root)

See each sample's own section below for its specific prerequisites (dataset
acquisition, extra tools, build features, etc.).

## Movies

Index and search ~32,000 movies from the Meilisearch movies dataset.

### Prerequisites (Movies-specific)

- [jq](https://jqlang.org/) — used to parse the JSON dataset
- [curl](https://curl.se/) — used to download poster images
- [python3](https://www.python.org/) — used for binary-to-JSON conversion
- The `embeddings-multimodal` feature must be enabled at build time
- Dataset: clone [meilisearch/datasets](https://github.com/meilisearch/datasets) next to the laurus project directory

  ```bash
  cd ..
  git clone https://github.com/meilisearch/datasets.git
  ```

  Expected directory layout:

  ```text
  parent/
  ├── datasets/       # meilisearch/datasets clone
  │   └── datasets/
  │       └── movies/
  │           └── movies.json
  └── laurus/         # this project
      └── examples/
  ```

### Run

```bash
# 1. Create the index
bash examples/movies/scripts/create_index.sh

# 2. Index all movies
bash examples/movies/scripts/index_movies.sh

# 3. Run example searches
bash examples/movies/scripts/search_movies.sh
```

See [examples/movies/schema.toml](movies/schema.toml) for the schema definition.

## Aozora Bunko

Index and search public-domain Japanese literature from [Aozora Bunko](https://www.aozora.gr.jp/).
Demonstrates morphological analysis via Lindera + IPADIC, per-field
analyzers, quoted-phrase vs. unquoted-OR query semantics, and semantic
vector / hybrid search via a Candle BERT text embedder.

### Prerequisites (Aozora-specific)

- [python3](https://www.python.org/) — used to parse the work list CSV, decode body text (CP932), and strip ruby markup
- [curl](https://curl.se/), [unzip](https://linux.die.net/man/1/unzip) — used to download and extract the dictionary and body-text archives
- Network access on first run (downloads the ~15 MB Lindera IPADIC dictionary, the ~470 MB text embedding model, and work body text; all are cached afterward)
- The `embeddings-candle` build feature must be enabled (all scripts in this example already pass it); no dataset needs to be cloned separately

### Run

```bash
# 1. Create the index (includes fetching the IPADIC dictionary)
bash examples/aozora/scripts/create_index.sh

# 2. Index works (default: 1,000)
bash examples/aozora/scripts/index_aozora.sh

# 3. Run example searches
bash examples/aozora/scripts/search_aozora.sh
```

> Aozora Bunko is run by volunteers. `--limit 0` (all works) downloads
> ~17,000 works — please use the default limit unless you have a specific
> reason not to.

See [examples/aozora/schema.toml](aozora/schema.toml) for the schema definition.
