# スキーマフォーマットリファレンス

スキーマファイルはインデックスの構造を定義します。どのフィールドが存在するか、その型、およびインデックスの方法を指定します。Laurus はスキーマファイルに TOML 形式を使用します。

## 概要

スキーマは 3 つのトップレベル要素で構成されます:

```toml
# スキーマに宣言されていないフィールドの扱い。省略時は "dynamic"。
dynamic_field_policy = "dynamic"

# クエリでフィールドが指定されていない場合にデフォルトで検索するフィールド。
default_fields = ["title", "body"]

# フィールド定義。各フィールドには名前と型付き設定があります。
[fields.<field_name>.<FieldType>]
# ... 型固有のオプション
```

- **`dynamic_field_policy`** — スキーマに**宣言されていない**フィールドがドキュメントに含まれる場合の挙動を制御します。値は `"strict"` / `"dynamic"` / `"ignore"`。デフォルトは `"dynamic"`。詳細および「`dynamic` では情報損失が起きうる」という警告は [動的スキーマ](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。
- **`default_fields`** — [Query DSL](../concepts/query_dsl.md) でデフォルトの検索対象として使用されるフィールド名のリストです。Lexical フィールド（Text、Integer、Float など）のみデフォルトフィールドに指定できます。このキーはオプションで、デフォルトは空のリストです。
- **`fields`** — フィールド名とその型付き設定のマップです。各フィールドにはフィールド型を1つだけ指定する必要があります。

## フィールド命名規則

- フィールド名は任意の文字列です（例: `title`、`body_vec`、`created_at`）。
- **アンダースコア（`_`）で始まるフィールド名はエンジンの予約領域**です。例外として `_id`（自動管理）のみ許可されます。それ以外の `_` プレフィックス名を宣言しようとするとエラーになります。
- フィールド名はスキーマ内で一意である必要があります。

## フィールド型

フィールドは **Lexical**（キーワード/全文検索用）と **Vector**（類似検索用）の2つのカテゴリに分類されます。1つのフィールドが両方を兼ねることはできません。

### Lexical フィールド

#### Text

全文検索可能なフィールドです。テキストは解析パイプライン（トークン化、正規化、ステミングなど）によって処理されます。

```toml
[fields.title.Text]
indexed = true       # このフィールドを検索用にインデックスするかどうか
stored = true        # 取得用に元の値を保存するかどうか
term_vectors = false # タームの位置を保存するかどうか（フレーズクエリ、ハイライト用）
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | このフィールドの検索を有効にする |
| `stored` | `bool` | `true` | 結果に返せるよう元の値を保存する |
| `term_vectors` | `bool` | `true` | フレーズクエリ、ハイライト、More-Like-This 用にタームの位置を保存する |

#### Integer

64ビット符号付き整数フィールド。範囲クエリと完全一致をサポートします。

```toml
[fields.year.Integer]
indexed = true
stored = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | 範囲クエリおよび完全一致クエリを有効にする |
| `stored` | `bool` | `true` | 元の値を保存する |

#### Float

64ビット浮動小数点フィールド。範囲クエリをサポートします。

```toml
[fields.rating.Float]
indexed = true
stored = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | 範囲クエリを有効にする |
| `stored` | `bool` | `true` | 元の値を保存する |

#### Boolean

ブーリアンフィールド（`true` / `false`）。

```toml
[fields.published.Boolean]
indexed = true
stored = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | ブーリアン値によるフィルタリングを有効にする |
| `stored` | `bool` | `true` | 元の値を保存する |

#### DateTime

UTC タイムスタンプフィールド。範囲クエリをサポートします。

```toml
[fields.created_at.DateTime]
indexed = true
stored = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | 日時の範囲クエリを有効にする |
| `stored` | `bool` | `true` | 元の値を保存する |

#### Geo

地理座標フィールド（緯度/経度）。半径クエリおよびバウンディングボックスクエリをサポートします。

```toml
[fields.location.Geo]
indexed = true
stored = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Geo クエリ（半径、バウンディングボックス）を有効にする |
| `stored` | `bool` | `true` | 元の値を保存する |

#### Bytes

生バイナリデータフィールド。インデックスされず、保存のみです。

```toml
[fields.thumbnail.Bytes]
stored = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `stored` | `bool` | `true` | バイナリデータを保存する |

### Vector フィールド

Vector フィールドは近似最近傍探索（ANN: Approximate Nearest Neighbor）用にインデックスされます。`dimension`（各ベクトルの長さ）と `distance` メトリクスの指定が必要です。

#### Hnsw

HNSW（Hierarchical Navigable Small World）グラフインデックス。ほとんどのユースケースに最適で、速度と再現率（Recall）のバランスに優れています。

```toml
[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
m = 16
ef_construction = 200
base_weight = 1.0
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | `128` | ベクトルの次元数（Embedding モデルの出力と一致させる必要あり） |
| `distance` | `string` | `"Cosine"` | 距離メトリクス（[距離メトリクス](#距離メトリクス)を参照） |
| `m` | `integer` | `16` | ノードあたりの最大双方向接続数。大きいほど再現率が向上するがメモリ使用量が増加 |
| `ef_construction` | `integer` | `200` | インデックス構築時の探索幅。大きいほど品質が向上するが構築が遅くなる |
| `base_weight` | `float` | `1.0` | ハイブリッド検索のスコア融合における重み |
| `quantizer` | `object` | *なし* | オプションの量子化方式（[量子化](#量子化)を参照） |

**チューニングガイドライン:**

- `m`: 12〜48 が一般的です。高次元ベクトルには大きい値を使用してください。
- `ef_construction`: 100〜500。大きい値ほどグラフの品質が向上しますが、構築時間が増加します。
- `dimension`: Embedding モデルの出力次元と正確に一致させる必要があります（例: `all-MiniLM-L6-v2` は 384、`BERT-base` は 768、`text-embedding-3-small` は 1536）。

#### Flat

ブルートフォース線形スキャンインデックス。近似を行わず正確な結果を返します。小規模データセット（10,000 ベクトル未満）に最適です。

```toml
[fields.embedding.Flat]
dimension = 384
distance = "Cosine"
base_weight = 1.0
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | `128` | ベクトルの次元数 |
| `distance` | `string` | `"Cosine"` | 距離メトリクス（[距離メトリクス](#距離メトリクス)を参照） |
| `base_weight` | `float` | `1.0` | ハイブリッド検索のスコア融合における重み |
| `quantizer` | `object` | *なし* | オプションの量子化方式（[量子化](#量子化)を参照） |

#### Ivf

IVF（Inverted File Index）。ベクトルをクラスタリングし、クラスタのサブセットのみを検索します。大規模データセットに適しています。

```toml
[fields.embedding.Ivf]
dimension = 384
distance = "Cosine"
n_clusters = 100
n_probe = 1
base_weight = 1.0
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | *（必須）* | ベクトルの次元数 |
| `distance` | `string` | `"Cosine"` | 距離メトリクス（[距離メトリクス](#距離メトリクス)を参照） |
| `n_clusters` | `integer` | `100` | クラスタ数。多いほど細かい分割が可能 |
| `n_probe` | `integer` | `1` | クエリ時に検索するクラスタ数。大きいほど再現率が向上するが遅くなる |
| `base_weight` | `float` | `1.0` | ハイブリッド検索のスコア融合における重み |
| `quantizer` | `object` | *なし* | オプションの量子化方式（[量子化](#量子化)を参照） |

> **注意:** Hnsw および Flat とは異なり、Ivf の `dimension` フィールドは**必須**であり、デフォルト値はありません。

**チューニングガイドライン:**

- `n_clusters`: 一般的な経験則は `sqrt(N)`（N はベクトルの総数）です。
- `n_probe`: 1 から始めて、再現率が許容範囲になるまで増やしてください。一般的な範囲は 1〜20 です。

## 距離メトリクス

Vector フィールドの `distance` オプションは以下の値を受け付けます:

| 値 | 説明 | 使用場面 |
| :--- | :--- | :--- |
| `"Cosine"` | コサイン距離（1 - コサイン類似度）。デフォルト。 | 正規化されたテキスト/画像 Embedding |
| `"Euclidean"` | L2（ユークリッド）距離 | 空間データ、正規化されていないベクトル |
| `"Manhattan"` | L1（マンハッタン）距離 | スパースな特徴ベクトル |
| `"DotProduct"` | 内積（大きいほど類似度が高い） | 大きさが重要な正規化済みベクトル |
| `"Angular"` | 角度距離 | コサインに似ているが角度に基づく |

ほとんどの Embedding モデル（BERT、Sentence Transformers、OpenAI など）では `"Cosine"` が適切な選択です。

## 量子化

Vector フィールドはオプションで量子化（Quantization）をサポートしており、精度を若干犠牲にしてメモリ使用量を削減できます。`quantizer` オプションを TOML テーブルとして指定します。

### なし（デフォルト）

量子化なし — 32ビット浮動小数点のフル精度。

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"
# quantizer を省略（量子化なし）
```

### Scalar 8-bit

各 float32 コンポーネントを uint8 に圧縮します（約4倍のメモリ削減）。

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"
quantizer = "Scalar8Bit"
```

### Product Quantization

ベクトルをサブベクトルに分割し、それぞれを独立に量子化します。

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"

[fields.embedding.Hnsw.quantizer.ProductQuantization]
subvector_count = 48
```

| オプション | 型 | 説明 |
| :--- | :--- | :--- |
| `subvector_count` | `integer` | サブベクトルの数。`dimension` を均等に割り切れる必要があります。 |

## 完全な例

### 全文検索のみ

Lexical 検索のみのシンプルなブログ記事インデックス:

```toml
default_fields = ["title", "body"]

[fields.title.Text]
indexed = true
stored = true
term_vectors = false

[fields.body.Text]
indexed = true
stored = true
term_vectors = false

[fields.category.Text]
indexed = true
stored = true
term_vectors = false

[fields.published_at.DateTime]
indexed = true
stored = true
```

### Vector 検索のみ

セマンティック類似検索用の Vector のみのインデックス:

```toml
[fields.embedding.Hnsw]
dimension = 768
distance = "Cosine"
m = 16
ef_construction = 200
```

### ハイブリッド検索（Lexical + Vector）

Lexical 検索と Vector 検索を組み合わせた両方の長所を活かす検索:

```toml
default_fields = ["title", "body"]

[fields.title.Text]
indexed = true
stored = true
term_vectors = false

[fields.body.Text]
indexed = true
stored = true
term_vectors = true

[fields.category.Text]
indexed = true
stored = true
term_vectors = false

[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
m = 16
ef_construction = 200
```

> **ヒント:** 1つのフィールドが Lexical と Vector の両方を兼ねることはできません。別々のフィールド（例: テキスト用の `body`、Embedding 用の `body_vec`）を使用し、どちらも同じソースコンテンツにマッピングしてください。

### E コマースの商品インデックス

複数のフィールド型を組み合わせたより複雑なスキーマ:

```toml
default_fields = ["name", "description"]

[fields.name.Text]
indexed = true
stored = true
term_vectors = false

[fields.description.Text]
indexed = true
stored = true
term_vectors = true

[fields.price.Float]
indexed = true
stored = true

[fields.in_stock.Boolean]
indexed = true
stored = true

[fields.created_at.DateTime]
indexed = true
stored = true

[fields.location.Geo]
indexed = true
stored = true

[fields.description_vec.Hnsw]
dimension = 384
distance = "Cosine"
```

## スキーマの生成

CLI を使用して対話的にスキーマ TOML ファイルを生成できます:

```bash
laurus create schema
laurus create schema --output my_schema.toml
```

詳細は [`create schema`](commands.md#create-schema) を参照してください。

## スキーマの使用

スキーマファイルが用意できたら、そこからインデックスを作成します:

```bash
laurus create index --schema schema.toml
```

または Rust でプログラム的に読み込みます:

```rust
use laurus::Schema;

let toml_str = std::fs::read_to_string("schema.toml")?;
let schema: Schema = toml::from_str(&toml_str)?;
```
