# スコアリングとランキング

Laurusは、Lexical検索に複数のスコアリングアルゴリズムを提供し、Vector検索には距離ベースの類似度を使用します。このページでは、すべてのスコアリングメカニズムとハイブリッド検索における相互作用について説明します。

## Lexicalスコアリング

### BM25（デフォルト）

BM25はLexical検索のデフォルトスコアリング関数です。単語頻度とドキュメント長の正規化をバランスさせます。

```text
score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
```

各パラメータの意味:

- **tf** -- ドキュメント内の単語頻度（Term Frequency）
- **IDF** -- 逆文書頻度（Inverse Document Frequency）。全ドキュメントにおける単語の希少性
- **k1** -- 単語頻度の飽和パラメータ
- **b** -- ドキュメント長の正規化係数
- **doc_len / avg_doc_len** -- ドキュメント長と平均ドキュメント長の比率

### ScoringConfig

`ScoringConfig` はBM25およびその他のスコアリングパラメータを制御します。

| パラメータ | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `k1` | `f32` | 1.2 | 単語頻度の飽和。値が大きいほど単語頻度の重みが増します。 |
| `b` | `f32` | 0.75 | フィールド長の正規化。0.0 = 正規化なし、1.0 = 完全な正規化。 |
| `tf_idf_boost` | `f32` | 1.0 | TF-IDFのグローバルブースト係数 |
| `enable_field_norm` | `bool` | true | フィールド長の正規化を有効にする |
| `field_boosts` | `HashMap<String, f32>` | empty | フィールドごとのスコア乗数 |
| `enable_coord` | `bool` | true | クエリ調整係数（matched_terms / total_query_terms）を有効にする |

### 代替スコアリング関数

| 関数 | 説明 |
| :--- | :--- |
| `BM25ScoringFunction` | 設定可能なk1とbを持つBM25（デフォルト） |
| `TfIdfScoringFunction` | フィールド長正規化付きの対数正規化TF-IDF |
| `VectorSpaceScoringFunction` | ドキュメントの単語ベクトル空間におけるコサイン類似度 |
| `CustomScoringFunction` | カスタムスコアリングロジック用のユーザー定義クロージャ |

### ScoringRegistry

`ScoringRegistry` はスコアリングアルゴリズムの中央レジストリを提供します。

```rust
// 事前登録済みアルゴリズム:
// - "bm25"          -> BM25ScoringFunction
// - "tf_idf"        -> TfIdfScoringFunction
// - "vector_space"  -> VectorSpaceScoringFunction
```

### BM25Plan（事前計算済みクエリ）

同じクエリに対して多数のドキュメントをスコアリングする場合、`BM25Plan` はクエリビルド時に単語ごとの IDF、フィールド長正規化の不変項、`k1 + 1` を一度だけ算出し、各ドキュメントを軽量な数値ループでスコアリングします。スコアリングメソッドは 2 つあります。

- `BM25Plan::score(&doc_stats)` — `BM25ScoringFunction::score` のドロップイン高速版。`DocumentStats` の `HashMap<String, u64>` 経由で単語頻度を引きます。
- `BM25Plan::score_packed(&term_freqs, doc_length)` — `query_terms` での位置でインデックスされた `&[u64]` をパック済み単語頻度として受け取ります。ドキュメントごとの文字列キー `HashMap` 引きを完全に回避できるため、呼び出し側でドキュメントごとに頻度を 1 度だけパックできる場合に使ってください。

```rust
use laurus::lexical::search::scoring::bm25::{BM25Plan, ScoringConfig};

let plan = BM25Plan::new(&query_terms, &collection_stats, &ScoringConfig::default());

// HashMap パス（ドロップイン高速版）:
let score = plan.score(&doc_stats);

// Packed パス（ドキュメントごとの HashMap 引きを回避）:
let term_freqs: Vec<u64> = query_terms
    .iter()
    .map(|t| *doc_stats.term_frequencies.get(t).unwrap_or(&0))
    .collect();
let score = plan.score_packed(&term_freqs, doc_stats.doc_length);
```

`BM25ScoringFunction::score` は 1 ドキュメントずつスコアリングする呼び出し側のために維持されています。

### フィールドブースト

フィールドブーストは、特定のフィールドからのスコア寄与に乗数を適用します。一部のフィールドが他よりも重要な場合に有用です。

```rust
use std::collections::HashMap;

let mut field_boosts = HashMap::new();
field_boosts.insert("title".to_string(), 2.0);  // titleのマッチはスコア2倍
field_boosts.insert("body".to_string(), 1.0);   // bodyのマッチはスコア1倍
```

### 調整係数（Coordination Factor）

`enable_coord` が true の場合、`AdvancedScorer` は調整係数を適用します。

```text
coord = matched_query_terms / total_query_terms
```

これはより多くのクエリ単語にマッチするドキュメントに報酬を与えます。例えば、クエリが3つの単語を含み、ドキュメントがそのうち2つにマッチする場合、調整係数は 2/3 = 0.667 になります。

## Vectorスコアリング

Vector検索は距離ベースの類似度で結果をランク付けします。

```text
similarity = 1 / (1 + distance)
```

距離は設定された距離メトリクスを使用して計算されます。

| メトリクス | 説明 | 最適な用途 |
| :--- | :--- | :--- |
| `Cosine` | 1 - コサイン類似度 | テキストEmbedding（最も一般的） |
| `Euclidean` | L2距離 | 空間データ |
| `Manhattan` | L1距離 | 特徴ベクトル |
| `DotProduct` | 符号反転した内積 | 事前正規化されたベクトル |
| `Angular` | 角度距離 | 方向の類似度 |

## ハイブリッド検索のスコア正規化

LexicalとVectorの結果を結合する場合、スコアを比較可能にする必要があります。

### RRF（Reciprocal Rank Fusion）

RRFは生のスコアの代わりにランクを使用することで、スコア正規化を完全に回避します。

```text
rrf_score = sum(1 / (k + rank))
```

`k` パラメータ（デフォルト: 60）はスムージングを制御します。値が大きいほど上位ランクの結果の重みが小さくなります。

### WeightedSum

WeightedSumは、各検索タイプのスコアをmin-max正規化で独立に正規化した後、結合します。

```text
norm_score = (score - min_score) / (max_score - min_score)
final_score = (norm_lexical * lexical_weight) + (norm_vector * vector_weight)
```

両方の重みは [0.0, 1.0] にクランプされます。
