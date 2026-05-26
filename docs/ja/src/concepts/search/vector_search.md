# Vector 検索

Vector 検索は、意味的類似性によってドキュメントを検索します。キーワードのマッチングではなく、ベクトル空間におけるクエリの意味とドキュメントエンベディングを比較します。

## 基本的な使い方

### Builder API

```rust
use laurus::SearchRequestBuilder;
use laurus::vector::VectorSearchRequestBuilder;

let request = SearchRequestBuilder::new()
    .vector_query(
        VectorSearchRequestBuilder::new()
            .add_text("embedding", "systems programming language")
            .limit(10)
            .build()
    )
    .build();

let results = engine.search(request).await?;
```

`add_text()` メソッドはテキストをクエリペイロードとして格納します。検索時に、エンジンが設定されたエンベッダーを使用してテキストをエンベディングし、ベクトルインデックスを検索します。

### Query DSL

```rust
use laurus::vector::VectorQueryParser;

let parser = VectorQueryParser::new(embedder.clone())
    .with_default_field("embedding");

let request = parser.parse(r#"embedding:"systems programming""#).await?;
```

## VectorSearchRequestBuilder

Builder API により、きめ細かな制御が可能です。

```rust
use laurus::vector::VectorSearchRequestBuilder;
use laurus::vector::store::request::QueryVector;

let request = VectorSearchRequestBuilder::new()
    // Text query (will be embedded at search time)
    .add_text("text_vec", "machine learning")

    // Or use a pre-computed vector directly
    .add_vector("embedding", vec![0.1, 0.2, 0.3, /* ... */])

    // Search parameters
    .limit(20)

    .build();
```

### メソッド

| メソッド | 説明 |
| :--- | :--- |
| `add_text(field, text)` | 特定のフィールドに対するテキストクエリを追加（検索時にエンベディング） |
| `add_vector(field, vector)` | 特定のフィールドに対する事前計算済みクエリベクトルを追加 |
| `add_vector_with_weight(field, vector, weight)` | 明示的なウェイトを持つ事前計算済みベクトルを追加 |
| `add_payload(field, payload)` | エンベディング対象の汎用 `DataValue` ペイロードを追加 |
| `add_bytes(field, bytes, mime)` | バイナリペイロードを追加（例: マルチモーダル用の画像バイト） |
| `field(name)` | 検索を特定のフィールドに制限 |
| `fields(names)` | 検索を複数のフィールドに制限 |
| `limit(n)` | 結果の最大件数（デフォルト: 10） |
| `score_mode(VectorScoreMode)` | スコア結合モード（`WeightedSum`、`MaxSim`、`LateInteraction`） |
| `min_score(f32)` | 最小スコア閾値（デフォルト: 0.0） |
| `overfetch(f32)` | 結果品質向上のためのオーバーフェッチ係数（デフォルト: 1.0） |
| `build()` | `VectorSearchRequest` を構築 |

## マルチフィールド Vector 検索

単一のリクエストで複数のベクトルフィールドを横断して検索できます。

```rust
let request = VectorSearchRequestBuilder::new()
    .add_text("text_vec", "cute kitten")
    .add_text("image_vec", "fluffy cat")
    .build();
```

各クエリ句はベクトルを生成し、対応するフィールドに対して検索されます。結果は設定されたスコアモードで結合されます。

### スコアモード

| モード | 説明 |
| :--- | :--- |
| `WeightedSum`（デフォルト） | すべてのクエリ句にわたる（類似度 * ウェイト）の合計 |
| `MaxSim` | クエリ句間の最大類似度スコア |
| `LateInteraction` | ColBERT スタイルの Late Interaction スコアリング |

### マルチベクトル検索の並列実行

複数のクエリベクトル（ColBERT スタイルの late interaction、マルチベクトル
MaxSim、ensemble reranker など）を含むリクエストに対して、Laurus は
[rayon][rayon-crate] を使ってクエリごとの類似度検索を並列に実行します。

動作:

- ネイティブビルド（`native` feature がデフォルトで有効）では、
  `query_vectors.len()` が内部の `MULTI_QUERY_PARALLEL_THRESHOLD`（現状 `4`）
  に達した時点で、HNSW / Flat / IVF のクエリごとの検索が rayon のグローバル
  スレッドプール上で並列実行されます。これを下回るとシリアルループのほうが
  速くなります（rayon のディスパッチオーバーヘッド ~1-2 µs が、単一クエリの
  50-200 µs を上回りやすいため）。
- `wasm32` ターゲットでは rayon が使用できないため、常にシリアルパスを
  通ります。
- 集約（スコアモードによるマージ）と最終ソートはクエリ並列フェーズの後に
  シリアル実行されます。スコアが同点の場合は `doc_id` の昇順でタイブレーク
  するため、rayon のワークスチール順序に依存しない決定的な結果になります。

外部 API（`VectorStore::search`、gRPC の `Search`、REST の
`POST /v1/search`、各言語バインディング）は一切変更されません。並列実行は
完全に内部最適化で、laurus をアップグレードするだけで有効になります。
speedup はホストの利用可能コア数に応じてスケールアップします（4 物理コア /
8 スレッドの HT 有効ラップトップ CPU では、B = 64 クエリ時にスループットが
およそ 2× 向上します。これは物理コア数と HT 共有による上限に近い値です）。

[rayon-crate]: https://docs.rs/rayon

### ウェイト

DSL では `^` ブースト構文を使用するか、`QueryVector` の `weight` で各フィールドの寄与度を調整します。

```text
text_vec:"cute kitten"^1.0 image_vec:"fluffy cat"^0.5
```

これは、テキストの類似度が画像の類似度の 2 倍の重みを持つことを意味します。

## フィルター付き Vector 検索

Lexical フィルターを適用して Vector 検索の結果を絞り込むことができます。

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::vector::VectorSearchRequestBuilder;

// Vector search with a category filter
let request = SearchRequestBuilder::new()
    .vector_query(
        VectorSearchRequestBuilder::new()
            .add_text("embedding", "machine learning")
            .build()
    )
    .filter_query(Box::new(TermQuery::new("category", "tutorial")))
    .limit(10)
    .build();

let results = engine.search(request).await?;
```

フィルタークエリはまず Lexical インデックス上で実行されて許可されるドキュメント ID のセットを特定し、その後 Vector 検索がそれらの ID に制限されます。

### 数値範囲によるフィルター

```rust
use laurus::lexical::NumericRangeQuery;
use laurus::lexical::core::field::NumericType;

let request = SearchRequestBuilder::new()
    .vector_query(
        VectorSearchRequestBuilder::new()
            .add_text("embedding", "type systems")
            .build()
    )
    .filter_query(Box::new(NumericRangeQuery::new(
        "year", NumericType::Integer,
        Some(2020.0), Some(2024.0), true, true
    )))
    .limit(10)
    .build();
```

## 距離メトリクス（Distance Metrics）

距離メトリクスはスキーマでフィールドごとに設定されます（[Vector インデキシング](../indexing/vector_indexing.md) を参照）。

| メトリクス | 説明 | 小さい値 = より類似 |
| :--- | :--- | :--- |
| **Cosine** | 1 - コサイン類似度 | はい |
| **Euclidean** | L2 距離 | はい |
| **Manhattan** | L1 距離 | はい |
| **DotProduct** | 負の内積 | はい |
| **Angular** | 角度距離 | はい |

## コード例: 完全な Vector 検索

```rust
use std::sync::Arc;
use laurus::{Document, Engine, Schema, SearchRequestBuilder, PerFieldEmbedder};
use laurus::lexical::TextOption;
use laurus::vector::HnswOption;
use laurus::vector::VectorSearchRequestBuilder;
use laurus::storage::memory::MemoryStorage;

#[tokio::main]
async fn main() -> laurus::Result<()> {
    let storage = Arc::new(MemoryStorage::new(Default::default()));

    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_hnsw_field("text_vec", HnswOption {
            dimension: 384,
            ..Default::default()
        })
        .build();

    // Set up per-field embedder
    let embedder = Arc::new(my_embedder);
    let pfe = PerFieldEmbedder::new(embedder.clone());
    pfe.add_embedder("text_vec", embedder.clone());

    let engine = Engine::builder(storage, schema)
        .embedder(Arc::new(pfe))
        .build()
        .await?;

    // Index documents (text in vector field is auto-embedded)
    engine.add_document("doc-1", Document::builder()
        .add_text("title", "Rust Programming")
        .add_text("text_vec", "Rust is a systems programming language.")
        .build()
    ).await?;
    engine.commit().await?;

    // Search by semantic similarity
    let results = engine.search(
        SearchRequestBuilder::new()
            .vector_query(
                VectorSearchRequestBuilder::new()
                    .add_text("text_vec", "systems language")
                    .build()
            )
            .limit(5)
            .build()
    ).await?;

    for r in &results {
        println!("{}: score={:.4}", r.id, r.score);
    }

    Ok(())
}
```

## 次のステップ

- キーワード検索と組み合わせる: [ハイブリッド検索](hybrid_search.md)
- Vector クエリの DSL 構文: [Query DSL](../query_dsl.md)
