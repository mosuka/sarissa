# Vector インデキシング

Vector インデキシングは、類似性ベースの検索を支える仕組みです。ドキュメントのベクトルフィールドがインデキシングされると、Laurus はエンベディングベクトルを専用のインデックス構造に格納し、高速な近似最近傍探索（Approximate Nearest Neighbor, ANN）を可能にします。

## Vector インデキシングの仕組み

```mermaid
sequenceDiagram
    participant Doc as Document
    participant Embedder
    participant Normalize as Normalizer
    participant Index as Vector Index

    Doc->>Embedder: "Rust is a systems language"
    Embedder-->>Normalize: [0.12, -0.45, 0.78, ...]
    Normalize->>Normalize: L2 normalize
    Normalize-->>Index: [0.14, -0.52, 0.90, ...]
    Index->>Index: Insert into index structure
```

### ステップごとの流れ

1. **エンベディング（Embed）**: テキスト（または画像）が、設定されたエンベッダーによってベクトルに変換される
2. **正規化（Normalize）**: ベクトルが L2 正規化される（コサイン類似度のため）
3. **インデキシング（Index）**: ベクトルが設定されたインデックス構造（Flat、HNSW、または IVF）に挿入される
4. **コミット（Commit）**: `commit()` の呼び出し時に、インデックスが永続ストレージにフラッシュされる

## インデックスタイプ

Laurus は 3 種類のベクトルインデックスタイプをサポートしており、それぞれ異なるパフォーマンス特性を持ちます。

### 比較

| 特性 | Flat | HNSW | IVF |
| :--- | :--- | :--- | :--- |
| **精度** | 100%（厳密） | 約 95-99%（近似） | 約 90-98%（近似） |
| **検索速度** | O(n) 線形スキャン | O(log n) グラフ走査 | O(n/k) クラスタスキャン |
| **メモリ使用量** | 低 | 高（グラフエッジ） | 中程度（セントロイド） |
| **インデックス構築時間** | 高速 | 中程度 | 低速（クラスタリング） |
| **最適な用途** | 1 万ベクトル未満 | 1 万 - 1,000 万ベクトル | 100 万ベクトル以上 |

### Flat インデックス

最もシンプルなインデックスです。クエリベクトルを格納されたすべてのベクトルと比較します（総当たり）。

```rust
use laurus::vector::FlatOption;
use laurus::vector::core::distance::DistanceMetric;

let opt = FlatOption {
    dimension: 384,
    distance: DistanceMetric::Cosine,
    ..Default::default()
};
```

- **利点**: 100% の再現率（厳密な結果）、シンプル、低メモリ
- **欠点**: 大規模データセットでは低速（線形スキャン）
- **使用場面**: ベクトル数が約 1 万未満の場合、または厳密な結果が必要な場合

### HNSW インデックス

**Hierarchical Navigable Small World** グラフ。デフォルトで最も一般的に使用されるインデックスタイプです。

```mermaid
graph TB
    subgraph "Layer 2 (sparse)"
        A2["A"] --- C2["C"]
    end

    subgraph "Layer 1 (medium)"
        A1["A"] --- B1["B"]
        A1 --- C1["C"]
        B1 --- D1["D"]
        C1 --- D1
    end

    subgraph "Layer 0 (dense - all vectors)"
        A0["A"] --- B0["B"]
        A0 --- C0["C"]
        B0 --- D0["D"]
        B0 --- E0["E"]
        C0 --- D0
        C0 --- F0["F"]
        D0 --- E0
        E0 --- F0
    end

    A2 -.->|"entry point"| A1
    A1 -.-> A0
    C2 -.-> C1
    C1 -.-> C0
    B1 -.-> B0
    D1 -.-> D0
```

HNSW アルゴリズムは、上位の疎なレイヤーから下位の密なレイヤーへと検索し、各レベルで検索空間を絞り込みます。

```rust
use laurus::vector::HnswOption;
use laurus::vector::core::distance::DistanceMetric;

let opt = HnswOption {
    dimension: 384,
    distance: DistanceMetric::Cosine,
    m: 16,                  // max connections per node per layer
    ef_construction: 200,   // search width during index building
    ..Default::default()
};
```

#### HNSW パラメータ

| パラメータ | デフォルト | 説明 | 影響 |
| :--- | :--- | :--- | :--- |
| `m` | 16 | レイヤーごとのノードあたりの最大双方向接続数 | 大きいほど再現率が向上するが、メモリ消費が増加 |
| `ef_construction` | 200 | インデックス構築時の探索幅 | 大きいほど再現率が向上するが、構築が低速に |
| `dimension` | 128 | ベクトルの次元数 | エンベッダーの出力と一致させる必要あり |
| `distance` | Cosine | 距離メトリクス | 下記の距離メトリクスを参照 |

**チューニングのヒント:**

- 再現率を向上させるには `m` を増やす（例: 32 または 64）。ただしメモリ消費が増加する
- インデックス品質を向上させるには `ef_construction` を増やす（例: 400）。ただし構築時間が増加する
- 検索時には、検索リクエストで設定する `ef_search` パラメータが探索幅を制御する

### IVF インデックス

**Inverted File Index**。ベクトルをクラスタに分割し、関連するクラスタのみを検索します。

```mermaid
graph TB
    Q["Query Vector"]
    Q --> C1["Cluster 1\n(centroid)"]
    Q --> C2["Cluster 2\n(centroid)"]

    C1 --> V1["vec_3"]
    C1 --> V2["vec_7"]
    C1 --> V3["vec_12"]

    C2 --> V4["vec_1"]
    C2 --> V5["vec_9"]
    C2 --> V6["vec_15"]

    style C1 fill:#f9f,stroke:#333
    style C2 fill:#f9f,stroke:#333
```

```rust
use laurus::vector::IvfOption;
use laurus::vector::core::distance::DistanceMetric;

let opt = IvfOption {
    dimension: 384,
    distance: DistanceMetric::Cosine,
    n_clusters: 100,   // number of clusters
    n_probe: 10,       // clusters to search at query time
    ..Default::default()
};
```

#### IVF パラメータ

| パラメータ | デフォルト | 説明 | 影響 |
| :--- | :--- | :--- | :--- |
| `n_clusters` | 100 | ボロノイセル（Voronoi Cell）の数 | クラスタ数が多いほど検索は高速になるが、再現率は低下 |
| `n_probe` | 1 | クエリ時に検索するクラスタ数 | 大きいほど再現率が向上するが、検索が低速に |
| `dimension` | （必須） | ベクトルの次元数 | エンベッダーの出力と一致させる必要あり |
| `distance` | Cosine | 距離メトリクス | 下記の距離メトリクスを参照 |

**チューニングのヒント:**

- `n_clusters` はベクトル数 `n` に対して `sqrt(n)` 程度に設定する
- 再現率と速度のバランスを取るため、`n_probe` を `n_clusters` の 5-20% に設定する
- IVF はトレーニングフェーズが必要なため、初回のインデキシングが遅くなる場合がある

## 距離メトリクス（Distance Metrics）

| メトリクス | 説明 | 値の範囲 | 最適な用途 |
| :--- | :--- | :--- | :--- |
| `Cosine` | 1 - コサイン類似度 | [0, 2] | テキストエンベディング（最も一般的） |
| `Euclidean` | L2 距離 | [0, +inf) | 空間データ |
| `Manhattan` | L1 距離 | [0, +inf) | 特徴ベクトル |
| `DotProduct` | 負の内積 | (-inf, +inf) | 事前正規化済みベクトル |
| `Angular` | 角度距離 | [0, pi] | 方向の類似性 |

```rust
use laurus::vector::core::distance::DistanceMetric;

let metric = DistanceMetric::Cosine;      // Default for text
let metric = DistanceMetric::Euclidean;    // For spatial data
let metric = DistanceMetric::Manhattan;    // L1 distance
let metric = DistanceMetric::DotProduct;   // For pre-normalized vectors
let metric = DistanceMetric::Angular;      // Angular distance
```

> **注意:** コサイン類似度の場合、ベクトルはインデキシング前に自動的に L2 正規化されます。距離が小さいほど、より類似していることを示します。

## 量子化（Quantization）

ベクトルはディスク上で **8 ビットスカラー量子化された整数** として保存
されます（Issue #481 Stage 1）。以前の 32 ビット浮動小数点形式と比較
して **約 4 倍小さく**、recall 損失は実用上ほぼ無視できる範囲（f32
ground truth に対して Recall@10 ≥ 0.95 — recall テスト
`laurus/tests/vector_recall_test.rs` 参照）です。

| 方式 | Enum バリアント | 説明 | メモリ削減率 |
| :--- | :--- | :--- | :--- |
| **スカラー 8 ビット** *(デフォルト)* | `Scalar8Bit` | per-segment global affine による `u8` 量子化 | 約 4 倍 |
| **プロダクト量子化** *(予約)* | `ProductQuantization { subvector_count }` | Issue #481 Stage 3 — 現状 `NotImplemented` | 約 16-64 倍 |

```rust
use laurus::vector::HnswOption;
use laurus::vector::core::quantization::QuantizationMethod;

// `quantizer` は `Scalar8Bit` がデフォルト。下記は
// `HnswOption { dimension: 384, ..Default::default() }` と等価。
let opt = HnswOption {
    dimension: 384,
    quantizer: QuantizationMethod::Scalar8Bit,
    ..Default::default()
};
```

> **破壊的変更（Issue #481 Stage 1）:** `quantizer` フィールドはもはや
> `Option<QuantizationMethod>` ではなく必須となり、デフォルトは
> `Scalar8Bit` です。f32 のままディスクに保存する形式は廃止されまし
> た。Stage 1 より前に作成した既存 vector index は意図的に読み取り
> 不可能で、ソースデータからの再構築が必要です。

### Scalar8Bit のしくみ

- 各 segment は flush 時に f32 ベクトル群から **global** な
  `(offset, scale)` ペア 1 組をトレーニング
  （`offset = min`, `scale = (max - min) / 255`）。
- 各 `f32` 要素は `u8 = clamp(round((v - offset) / scale), 0, 255)`
  でエンコード。
- 各ベクトル単位のメタデータ（`sum_q: u32`, `norm_q: f32`）を
  事前計算して int8 ペイロードと併置するため、cosine 検索の hot loop
  は int8 SIMD multiply-accumulate 1 回 + scalar 補正 3 回に縮約され、
  検索時の per-element dequantize は不要。
- segment ファイルは `LVS1` magic + 16 byte header で始まり、reader
  はロード時にフォーマットを判定。

### Two-stage rerank（Issue #481 Stage 2 — 予約）

`VectorIndexQueryParams.rerank_factor: Option<usize>` および gRPC /
JSON gateway 側の `VectorParams.rerank_factor` は次の 2 段階 rerank
フロー用に予約されています:

1. 量子化された HNSW 検索が `top_k * rerank_factor` 候補を返す。
2. それらを完全な f32 ベクトルで再スコアし、最終 `top_k` を返す。

現在 `rerank_factor` を設定すると `LaurusError::NotImplemented` /
`tonic::Status::Unimplemented` が返ります。Stage 2 の実装が入った際に
proto / バインディングのスキーマ変更を伴わずにオプトインできるよう、
API surface を予約してあります。

## セグメントファイル

各ベクトルインデックスタイプは、データを単一のセグメントファイルに格納します。

| インデックスタイプ | ファイル拡張子 | 内容 |
| :--- | :--- | :--- |
| HNSW | `.hnsw` | グラフ構造、ベクトル、メタデータ |
| Flat | `.flat` | 生ベクトルとメタデータ |
| IVF | `.ivf` | クラスタセントロイド、割り当て済みベクトル、メタデータ |

## コード例

```rust
use std::sync::Arc;
use laurus::{Document, Engine, Schema};
use laurus::lexical::TextOption;
use laurus::vector::HnswOption;
use laurus::vector::core::distance::DistanceMetric;
use laurus::storage::memory::MemoryStorage;

#[tokio::main]
async fn main() -> laurus::Result<()> {
    let storage = Arc::new(MemoryStorage::new(Default::default()));
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_hnsw_field("embedding", HnswOption {
            dimension: 384,
            distance: DistanceMetric::Cosine,
            m: 16,
            ef_construction: 200,
            ..Default::default()
        })
        .build();

    // With an embedder, text in vector fields is automatically embedded
    let engine = Engine::builder(storage, schema)
        .embedder(my_embedder)
        .build()
        .await?;

    // Add text to the vector field — it will be embedded automatically
    engine.add_document("doc-1", Document::builder()
        .add_text("title", "Rust Programming")
        .add_text("embedding", "Rust is a systems programming language.")
        .build()
    ).await?;

    engine.commit().await?;

    Ok(())
}
```

## 次のステップ

- ベクトルインデックスを検索する: [Vector 検索](../search/vector_search.md)
- Lexical 検索と組み合わせる: [ハイブリッド検索](../search/hybrid_search.md)
