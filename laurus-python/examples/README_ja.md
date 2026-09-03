# laurus-python サンプル集

このディレクトリには `laurus` Python バインディングの実行可能なサンプルが含まれています。

## 前提条件

- Rust ツールチェーン（`rustup` — <https://rustup.rs>）
- Python 3.10 以上
- `maturin` ビルドツール

```bash
pip install maturin
```

## セットアップ

すべてのサンプルは `laurus-python/` ディレクトリで `maturin develop` を実行した後に動かせます。

```bash
cd laurus-python

# 仮想環境の作成と有効化（推奨）
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install maturin
```

## サンプル一覧

### 基本サンプル（追加依存なし）

一度ビルドすれば、以下のサンプルをすべて実行できます。

```bash
maturin develop
```

| サンプル | 説明 |
| :--- | :--- |
| [quickstart.py](quickstart.py) | 最小構成の全文検索: インデックス・検索・更新 |
| [lexical_search.py](lexical_search.py) | 全 Lexical クエリタイプ: Term、Phrase、Fuzzy、Wildcard、NumericRange、Geo、Boolean、Span |
| [synonym_graph_filter.py](synonym_graph_filter.py) | 解析パイプラインでの同義語展開 |

```bash
python examples/quickstart.py
python examples/lexical_search.py
python examples/synonym_graph_filter.py
```

---

### ベクトル検索 — 組み込み Embedder

laurus 内蔵の `CandleBert` Embedder（[Candle](https://github.com/huggingface/candle) 経由）を使用します。
テキストのベクトル化はインデックス時・検索時に Rust エンジンが自動で行います。
**外部の埋め込みライブラリは不要です。**

`embeddings-candle` フィーチャーフラグを付けてビルドします。

```bash
maturin develop --features embeddings-candle
```

| サンプル | 説明 |
| :--- | :--- |
| [vector_search.py](vector_search.py) | 組み込み BERT Embedder によるセマンティック類似度検索 |
| [hybrid_search.py](hybrid_search.py) | RRF・WeightedSum フュージョンによる Lexical + Vector ハイブリッド検索 |

```bash
python examples/vector_search.py
python examples/hybrid_search.py
```

> **注意:** 初回実行時に Hugging Face Hub からモデルをダウンロードします
> （`sentence-transformers/all-MiniLM-L6-v2`、約 90 MB）。
> 2回目以降はローカルキャッシュが使われます。

---

### ベクトル検索 — 外部 Embedder

Python 側で [sentence-transformers](https://www.sbert.net/) を使ってベクトルを生成し、
`VectorQuery` で laurus に渡す方式です。
`sentence-transformers` がインストールされていない場合はランダムベクトルにフォールバックします
（意味的な類似度は保証されません）。

```bash
maturin develop
pip install sentence-transformers   # 省略可（フォールバックあり）
```

| サンプル | 説明 |
| :--- | :--- |
| [external_embedder.py](external_embedder.py) | Python 管理の Embedder によるベクトル検索とハイブリッド検索 |

```bash
python examples/external_embedder.py
```

---

### OpenAI Embeddings

OpenAI API でベクトルを生成し、事前計算済みベクトルとして laurus に渡します。
OpenAI API キーが必要です。

```bash
maturin develop
pip install openai
export OPENAI_API_KEY=your-api-key-here
```

| サンプル | 説明 |
| :--- | :--- |
| [search_with_openai.py](search_with_openai.py) | OpenAI `text-embedding-3-small` によるベクトル検索 |

```bash
python examples/search_with_openai.py
```

---

### マルチモーダル検索

Python 側で生成した CLIP 埋め込みを使い、テキストと画像をまたいで検索します。
`torch` / `transformers` がインストールされていない場合はランダムベクトルにフォールバックします。

```bash
maturin develop
pip install torch transformers Pillow   # 省略可（フォールバックあり）
```

| サンプル | 説明 |
| :--- | :--- |
| [multimodal_search.py](multimodal_search.py) | CLIP を使ったテキスト→画像・画像→画像検索 |

```bash
python examples/multimodal_search.py
```

---

## 埋め込み方式の比較

| 方式 | サンプル | メリット | デメリット |
| :--- | :--- | :--- | :--- |
| **組み込み Embedder** | `vector_search.py`, `hybrid_search.py` | Python 側の埋め込みライブラリ不要・コードがシンプル | ビルド時に `embeddings-candle` フィーチャーフラグが必要 |
| **外部 Embedder** | `external_embedder.py` | モデルを自由に選択可能・任意の Python ライブラリが使える | インデックス時・検索時の埋め込みを自前で管理する必要がある |
| **OpenAI API** | `search_with_openai.py` | 高品質なクラウド埋め込み | API キーとネットワーク接続が必要 |
| **CLIP（マルチモーダル）** | `multimodal_search.py` | テキストと画像をまたいで検索可能 | `torch` 等の重い依存がある |
