# Feature Flags

`laurus` クレートはデフォルトでは Feature が無効の状態で提供されます。必要に応じて Embedding サポートを有効にしてください。

## 利用可能な Feature

| Feature | 説明 | 主な依存クレート |
| :--- | :--- | :--- |
| `embeddings-candle` | Hugging Face Candle によるローカル BERT Embedding | candle-core, candle-nn, candle-transformers, hf-hub, tokenizers |
| `embeddings-openai` | OpenAI API Embedding | reqwest |
| `embeddings-multimodal` | CLIP マルチモーダル Embedding（テキスト + 画像） | image, embeddings-candle |
| `embeddings-all` | すべての Embedding Feature を統合 | 上記すべて |

## 各 Feature の詳細

### `embeddings-candle`

`CandleBertEmbedder` を有効にし、CPU 上でローカルに BERT モデルを実行できるようにします。モデルは初回使用時に Hugging Face Hub からダウンロードされます。

```toml
[dependencies]
laurus = { version = "0.9", features = ["embeddings-candle"] }
```

### `embeddings-openai`

`OpenAIEmbedder` を有効にし、OpenAI Embeddings API を呼び出せるようにします。実行時に `OPENAI_API_KEY` 環境変数が必要です。

```toml
[dependencies]
laurus = { version = "0.9", features = ["embeddings-openai"] }
```

### `embeddings-multimodal`

`CandleClipEmbedder` を有効にし、CLIP ベースのテキストおよび画像 Embedding を使用できるようにします。`embeddings-candle` を暗黙的に有効にします。

```toml
[dependencies]
laurus = { version = "0.9", features = ["embeddings-multimodal"] }
```

### `embeddings-all`

すべての Embedding Feature を有効にする便利な Feature です。

```toml
[dependencies]
laurus = { version = "0.9", features = ["embeddings-all"] }
```

## Feature Flag がバイナリサイズに与える影響

Embedding Feature を有効にすると、コンパイル時間とバイナリサイズが増加する依存クレートが追加されます。

| 構成 | おおよその影響 |
| :--- | :--- |
| Feature なし（Lexical のみ） | ベースライン |
| `embeddings-candle` | + Candle ML フレームワーク |
| `embeddings-openai` | + reqwest HTTP クライアント |
| `embeddings-multimodal` | + 画像処理 + Candle |
| `embeddings-all` | 上記すべて |

Lexical（キーワード）検索のみが必要な場合は、Feature を有効にせずに Laurus を使用することで、最小のバイナリサイズと最速のコンパイル時間を実現できます。
