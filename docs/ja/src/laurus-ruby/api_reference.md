# API リファレンス

## Index

Laurus 検索エンジンをラップするメインクラスです。

```ruby
Laurus::Index.new(path: nil, schema: nil)
```

### コンストラクタ

| パラメータ | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `path:` | `String \| nil` | `nil` | 永続ストレージのディレクトリパス。`nil` の場合はインメモリインデックスを作成します。 |
| `schema:` | `Schema \| nil` | `nil` | スキーマ定義。省略時は空のスキーマが使用されます。 |

### メソッド

| メソッド | 説明 |
| :--- | :--- |
| `put_document(id, doc)` | ドキュメントをアップサート（upsert）します。同じ ID の既存バージョンをすべて置換します。 |
| `add_document(id, doc)` | 既存バージョンを削除せずにドキュメントチャンクを追記します。 |
| `get_documents(id) -> Array<Hash>` | 指定 ID の全保存バージョンを返します。 |
| `delete_documents(id)` | 指定 ID の全バージョンを削除します。 |
| `commit` | バッファリングされた書き込みをフラッシュし、すべての保留中の変更を検索可能にします。 |
| `search(query, limit: 10, offset: 0) -> Array<SearchResult>` | 検索クエリを実行します。 |
| `stats -> Hash` | インデックス統計（`"document_count"`、`"vector_fields"`）を返します。 |

### `search` の query 引数

`query` パラメータは以下のいずれかを受け付けます：

- **DSL 文字列**（例: `"title:hello"`、`"content:\"memory safety\""`)
- **Lexical クエリオブジェクト**（`TermQuery`、`PhraseQuery`、`BooleanQuery` など）
- **Vector クエリオブジェクト**（`VectorQuery`、`VectorTextQuery`）
- **`SearchRequest`**（完全な制御が必要な場合）

---

## Schema

`Index` のフィールドとインデックスタイプを定義します。

```ruby
Laurus::Schema.new
```

### フィールドメソッド

| メソッド | 説明 |
| :--- | :--- |
| `add_text_field(name, stored: true, indexed: true, term_vectors: false, analyzer: nil)` | 全文フィールド（転置インデックス、BM25）。 |
| `add_integer_field(name, stored: true, indexed: true, multi_valued: false)` | 64 ビット整数フィールド。`multi_valued: true` で整数配列を受け付け（範囲クエリは "any match"）。 |
| `add_float_field(name, stored: true, indexed: true, multi_valued: false)` | 64 ビット浮動小数点フィールド。`multi_valued: true` で浮動小数点配列を受け付け（範囲クエリは "any match"）。 |
| `add_boolean_field(name, stored: true, indexed: true)` | ブールフィールド。 |
| `add_bytes_field(name, stored: true)` | 生バイトフィールド。 |
| `add_geo_field(name, stored: true, indexed: true)` | 地理座標フィールド（緯度/経度）。 |
| `add_datetime_field(name, stored: true, indexed: true)` | UTC 日時フィールド。 |
| `add_hnsw_field(name, dimension, distance: "cosine", m: 16, ef_construction: 200, embedder: nil)` | HNSW 近似最近傍ベクトルフィールド。 |
| `add_flat_field(name, dimension, distance: "cosine", embedder: nil)` | Flat（総当たり）ベクトルフィールド。 |
| `add_ivf_field(name, dimension, distance: "cosine", n_clusters: 100, n_probe: 1, embedder: nil)` | IVF 近似最近傍ベクトルフィールド。 |

### その他のメソッド

| メソッド | 説明 |
| :--- | :--- |
| `add_embedder(name, config)` | 名前付きエンベダー定義を登録します。`config` は `"type"` キーを持つ Hash です（下記参照）。 |
| `set_default_fields(fields)` | クエリでフィールドが指定されていない場合に使用するデフォルトフィールドを設定します。`fields` は文字列の配列です。 |
| `set_dynamic_field_policy(policy)` | 未宣言フィールドの扱いを設定します。`policy` は `"strict"` / `"dynamic"`（デフォルト）/ `"ignore"`。詳細は下記を参照。 |
| `dynamic_field_policy -> String` | 現在のポリシーを小文字の文字列で返します。 |
| `field_names -> Array<String>` | このスキーマに定義されたフィールド名のリストを返します。 |

#### Dynamic field policy（動的フィールドポリシー）

ドキュメントに含まれるがスキーマに宣言されていないフィールドの扱いを制御します:

- `"strict"` — ドキュメントを拒否
- `"dynamic"`（デフォルト）— 各未宣言フィールドの型を推論してスキーマに追加。**警告**: integer フィールドに入ってきた float 値は静かに切り捨てられます（`3.14` → `3`）。厳密さが必要なら `"strict"` を使用してください
- `"ignore"` — 未宣言フィールドを静かに破棄

詳細な挙動マトリクスは [スキーマとフィールド](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。

### エンベダータイプ

| `"type"` | 必須キー | Feature Flag |
| :--- | :--- | :--- |
| `"precomputed"` | -- | （常に利用可能） |
| `"candle_bert"` | `"model"` | `embeddings-candle` |
| `"candle_clip"` | `"model"` | `embeddings-multimodal` |
| `"openai"` | `"model"` | `embeddings-openai` |

### 距離メトリクス

| 値 | 説明 |
| :--- | :--- |
| `"cosine"` | コサイン類似度（デフォルト） |
| `"euclidean"` | ユークリッド距離 |
| `"dot_product"` | 内積 |
| `"manhattan"` | マンハッタン距離 |
| `"angular"` | 角度距離 |

---

## クエリクラス

### TermQuery

```ruby
Laurus::TermQuery.new(field, term)
```

指定フィールドに完全一致する語句を含むドキュメントを検索します。

### PhraseQuery

```ruby
Laurus::PhraseQuery.new(field, terms)
```

指定した語句が順序どおりに含まれるドキュメントを検索します。`terms` は文字列の配列です。

### FuzzyQuery

```ruby
Laurus::FuzzyQuery.new(field, term, max_edits: 2)
```

編集距離が `max_edits` 以内の近似一致を検索します。

### WildcardQuery

```ruby
Laurus::WildcardQuery.new(field, pattern)
```

ワイルドカードパターン検索。`*` は任意の文字列、`?` は任意の1文字に一致します。

### NumericRangeQuery

```ruby
Laurus::NumericRangeQuery.new(field, min: nil, max: nil)
```

`[min, max]` の範囲内の数値を検索します。開いた境界には `nil` を指定します。型（整数または浮動小数点）は `min`/`max` の Ruby 型から推論されます。

### GeoQuery

```ruby
# 半径検索
Laurus::GeoQuery.within_radius(field, lat, lon, distance_km)

# バウンディングボックス検索
Laurus::GeoQuery.within_bounding_box(field, min_lat, min_lon, max_lat, max_lon)
```

`within_radius` は指定した地点から `distance_km` 以内の座標を持つドキュメントを返します。`within_bounding_box` は指定したバウンディングボックス内のドキュメントを返します。

### BooleanQuery

```ruby
bq = Laurus::BooleanQuery.new
bq.must(query)
bq.should(query)
bq.must_not(query)
```

複合ブールクエリ。`must` 節はすべて一致する必要があります。`should` 節は少なくとも1つ一致する必要があります。`must_not` 節は一致してはなりません。

### SpanQuery

```ruby
# 単一語句
Laurus::SpanQuery.term(field, term)

# Near: slop 位置以内の語句
Laurus::SpanQuery.near(field, terms, slop: 0, ordered: true)

# ネストされた SpanQuery 句を使った Near
Laurus::SpanQuery.near_spans(field, clauses, slop: 0, ordered: true)

# Containing: big スパンが little スパンを含む
Laurus::SpanQuery.containing(field, big, little)

# Within: 最大距離での include スパンと exclude スパン
Laurus::SpanQuery.within(field, include_span, exclude_span, distance)
```

位置・近接スパンクエリ。`near` は語句文字列の配列を受け取り、`near_spans` はネスト式のために `SpanQuery` オブジェクトの配列を受け取ります。

### VectorQuery

```ruby
Laurus::VectorQuery.new(field, vector)
```

事前計算済みエンベディングベクトルを使った近似最近傍検索を行います。`vector` は Float の配列です。

### VectorTextQuery

```ruby
Laurus::VectorTextQuery.new(field, text)
```

クエリ時に `text` をエンベディングに変換してベクトル検索を行います。インデックスにエンベダーの設定が必要です。

---

## SearchRequest

高度な制御が必要な場合の完全なリクエストクラスです。

```ruby
Laurus::SearchRequest.new(
  query: nil,
  lexical_query: nil,
  vector_query: nil,
  filter_query: nil,
  fusion: nil,
  limit: 10,
  offset: 0,
)
```

| パラメータ | 説明 |
| :--- | :--- |
| `query:` | DSL 文字列または単一クエリオブジェクト。`lexical_query:` / `vector_query:` と排他的。 |
| `lexical_query:` | 明示的なハイブリッド検索の Lexical コンポーネント。 |
| `vector_query:` | 明示的なハイブリッド検索の Vector コンポーネント。 |
| `filter_query:` | スコアリング後に適用する Lexical フィルター。 |
| `fusion:` | フュージョンアルゴリズム（`RRF` または `WeightedSum`）。両コンポーネント指定時のデフォルトは `RRF(k: 60)`。 |
| `limit:` | 最大結果件数（デフォルト 10）。 |
| `offset:` | ページネーションオフセット（デフォルト 0）。 |

---

## SearchResult

`Index#search` が返すクラスです。

```ruby
result.id        # => String   -- 外部ドキュメント識別子
result.score     # => Float    -- 関連性スコア
result.document  # => Hash|nil -- 取得されたフィールド値。削除済みの場合は nil
```

---

## フュージョンアルゴリズム

### RRF

```ruby
Laurus::RRF.new(k: 60.0)
```

逆順位フュージョン（Reciprocal Rank Fusion）。Lexical と Vector の結果リストを順位位置によってマージします。`k` は平滑化定数で、値が大きいほど上位ランクの影響が小さくなります。

### WeightedSum

```ruby
Laurus::WeightedSum.new(lexical_weight: 0.5, vector_weight: 0.5)
```

両スコアリストをそれぞれ正規化した後、`lexical_weight * lexical_score + vector_weight * vector_score` として結合します。

---

## テキスト解析

### SynonymDictionary

```ruby
dict = Laurus::SynonymDictionary.new
dict.add_synonym_group(["fast", "quick", "rapid"])
```

同義語グループの辞書です。グループ内のすべての語句は互いの同義語として扱われます。

### WhitespaceTokenizer

```ruby
tokenizer = Laurus::WhitespaceTokenizer.new
tokens = tokenizer.tokenize("hello world")
```

空白で分割してテキストをトークン化し、`Token` オブジェクトの配列を返します。

### SynonymGraphFilter

```ruby
filter = Laurus::SynonymGraphFilter.new(dictionary, keep_original: true, boost: 1.0)
expanded = filter.apply(tokens)
```

`SynonymDictionary` の同義語でトークンを展開するトークンフィルターです。

### Token

```ruby
token.text                # => String  -- トークンテキスト
token.position            # => Integer -- トークンストリーム内の位置
token.start_offset        # => Integer -- 元テキスト内の文字開始オフセット
token.end_offset          # => Integer -- 元テキスト内の文字終了オフセット
token.boost               # => Float   -- スコアブースト係数（1.0 = 調整なし）
token.stopped             # => Boolean -- ストップフィルターによって除去されたかどうか
token.position_increment  # => Integer -- 前のトークンの位置との差分
token.position_length     # => Integer -- このトークンがカバーする位置数
```

---

## フィールド値の型マッピング

Ruby の値は自動的に Laurus の `DataValue` 型に変換されます：

| Ruby 型 | Laurus 型 | 備考 |
| :--- | :--- | :--- |
| `nil` | `Null` | |
| `true` / `false` | `Bool` | |
| `Integer` | `Int64` | |
| `Float` | `Float64` | |
| `String` | `Text` | |
| `Array`（数値） | `Vector` | 要素は `f32` に変換 |
| `Hash`（`"lat"`, `"lon"`） | `Geo` | 2 つの `Float` 値 |
| `Time`（`iso8601` に応答） | `DateTime` | `iso8601` 経由で変換 |
