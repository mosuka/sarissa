# Feature Flags

The `laurus` crate ships with no default features. Enable embedding support as needed.

## Available Flags

| Feature | Description | Key Dependencies |
| :--- | :--- | :--- |
| `embeddings-candle` | Local BERT embeddings via Hugging Face Candle | candle-core, candle-nn, candle-transformers, hf-hub, tokenizers |
| `embeddings-openai` | OpenAI API embeddings | reqwest |
| `embeddings-multimodal` | CLIP multimodal embeddings (text + image) | image, embeddings-candle |
| `embeddings-all` | All embedding features combined | All of the above |

## What Each Flag Enables

### `embeddings-candle`

Enables `CandleBertEmbedder` for running BERT models locally on the CPU. Models are downloaded from Hugging Face Hub on first use.

```toml
[dependencies]
laurus = { version = "0.9", features = ["embeddings-candle"] }
```

### `embeddings-openai`

Enables `OpenAIEmbedder` for calling the OpenAI Embeddings API. Requires an `OPENAI_API_KEY` environment variable at runtime.

```toml
[dependencies]
laurus = { version = "0.9", features = ["embeddings-openai"] }
```

### `embeddings-multimodal`

Enables `CandleClipEmbedder` for CLIP-based text and image embeddings. Implies `embeddings-candle`.

```toml
[dependencies]
laurus = { version = "0.9", features = ["embeddings-multimodal"] }
```

### `embeddings-all`

Convenience flag that enables all embedding features.

```toml
[dependencies]
laurus = { version = "0.9", features = ["embeddings-all"] }
```

## Feature Flag Impact on Binary Size

Enabling embedding features adds dependencies that increase compile time and binary size:

| Configuration | Approximate Impact |
| :--- | :--- |
| No features (lexical only) | Baseline |
| `embeddings-candle` | + Candle ML framework |
| `embeddings-openai` | + reqwest HTTP client |
| `embeddings-multimodal` | + image processing + Candle |
| `embeddings-all` | All of the above |

If you only need lexical (keyword) search, you can use Laurus with no features enabled for the smallest binary and fastest compile time.
