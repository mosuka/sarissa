# サンプル

`laurus/examples/` ディレクトリには、ライブラリのさまざまな機能を示す実行可能なサンプルが含まれています。

## サンプルの実行

```bash
# Feature Flags なしでサンプルを実行
cargo run --example <name>

# Feature Flags を指定してサンプルを実行
cargo run --example <name> --features <flag>
```

## 利用可能なサンプル

### quickstart

基本的なワークフローを示す最小限のサンプルです: Storage の作成、Schema の定義、Engine の構築、ドキュメントのインデックス、検索を行います。

```bash
cargo run --example quickstart
```

デモ内容: インメモリストレージ、`TextOption`、`TermQuery`、`LexicalSearchQuery`

### lexical_search

すべての Lexical クエリ型を示す包括的なサンプルです。Builder API と QueryParser DSL の両方を使用します。

```bash
cargo run --example lexical_search
```

デモ内容: `TermQuery`、`PhraseQuery`、`FuzzyQuery`、`WildcardQuery`、`NumericRangeQuery`、`GeoDistanceQuery` / `GeoBoundingBoxQuery`、`BooleanQuery`、`SpanQuery`

### vector_search

モックエンベッダを使用した Vector 検索のサンプルです。フィルタ付き Vector 検索や DSL 構文も含みます。

```bash
cargo run --example vector_search
```

デモ内容: `PerFieldEmbedder`、`VectorSearchRequestBuilder`、フィルタ付き検索、DSL 構文（`field:"query"`）

### hybrid_search

異なる融合アルゴリズムを用いた Lexical 検索と Vector 検索の統合サンプルです。

```bash
cargo run --example hybrid_search
```

デモ内容: Lexical のみ、Vector のみ、ハイブリッド検索。`RRF` と `WeightedSum` の両方の融合アルゴリズム。Builder API と DSL。

### geo3d_search

Earth-Centered Earth-Fixed (ECEF) 座標系を用いた 3D 地理検索のサンプルです。6 つのランドマーク（東京タワー、東京スカイツリー、富士山頂、自由の女神像、シドニー・オペラハウス、ISS サンプル点）を `wgs84_to_ecef` で変換してインデックスします。

```bash
cargo run --example geo3d_search
```

デモ内容: `Geo3dDistanceQuery`（球）、`Geo3dBoundingBoxQuery`（3D AABB）、`Geo3dNearestQuery`（k-NN）、および `wgs84_to_ecef` 変換ユーティリティ。バウンディングボックスクエリは ISS サンプルを意図的に除外し、高度が第三の軸として機能することを示します。

### search_with_candle

Hugging Face Candle を使用した実際の BERT エンベディングによる Vector 検索です。初回実行時にモデルが自動的にダウンロードされます（約 80 MB）。

```bash
cargo run --example search_with_candle --features embeddings-candle
```

**必要条件:** `embeddings-candle` Feature Flag

デモ内容: `CandleBertEmbedder`（`sentence-transformers/all-MiniLM-L6-v2`、384 次元）

### search_with_openai

OpenAI Embeddings API を使用した Vector 検索です。

```bash
export OPENAI_API_KEY=your-api-key
cargo run --example search_with_openai --features embeddings-openai
```

**必要条件:** `embeddings-openai` Feature Flag、`OPENAI_API_KEY` 環境変数

デモ内容: `OpenAIEmbedder`（`text-embedding-3-small`、1536 次元）

### multimodal_search

CLIP モデルを使用したマルチモーダル（テキスト + 画像）検索です。

```bash
cargo run --example multimodal_search --features embeddings-multimodal
```

**必要条件:** `embeddings-multimodal` Feature Flag

デモ内容: `CandleClipEmbedder`、ファイルシステムからの画像インデックス、テキスト→画像クエリおよび画像→画像クエリ

### synonym_graph_filter

解析時のトークン展開のための `SynonymGraphFilter` のデモです。

```bash
cargo run --example synonym_graph_filter
```

デモ内容: シノニム辞書の作成、シノニムによるトークン展開、ブーストの適用、トークンの position および position_length 属性

## ヘルパーモジュール: common.rs

`common.rs` ファイルは、サンプルで使用される共通ユーティリティを提供します:

- `memory_storage()` -- インメモリストレージインスタンスの作成
- `per_field_analyzer()` -- 特定のフィールドに `KeywordAnalyzer` を設定した `PerFieldAnalyzer` の作成
- `MockEmbedder` -- 実際のモデルなしで Vector 検索をテストするためのモック `Embedder` 実装
