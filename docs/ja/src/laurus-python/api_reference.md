# API リファレンス

## Index

Laurus 検索エンジンをラップするメインクラスです。

```python
class Index:
    def __init__(self, path: str | None = None, schema: Schema | None = None) -> None: ...
```

### コンストラクタ

| パラメータ | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `path` | `str \| None` | `None` | 永続ストレージのディレクトリパス。`None` の場合はインメモリインデックスを作成します。 |
| `schema` | `Schema \| None` | `None` | スキーマ定義。省略時は空のスキーマが使用されます。 |

### メソッド

| メソッド | 説明 |
| :--- | :--- |
| `put_document(id, doc)` | ドキュメントをアップサート（upsert）します。同じ ID の既存バージョンをすべて置換します。 |
| `add_document(id, doc)` | 既存バージョンを削除せずにドキュメントチャンクを追記します。 |
| `get_documents(id) -> list[dict]` | 指定 ID の全保存バージョンを返します。 |
| `delete_documents(id)` | 指定 ID の全バージョンを削除します。 |
| `commit()` | バッファリングされた書き込みをフラッシュし、すべての保留中の変更を検索可能にします。 |
| `search(query, *, limit=10, offset=0) -> list[SearchResult]` | 検索クエリを実行します。 |
| `stats() -> dict` | インデックス統計（`document_count`、`vector_fields`）を返します。 |

### `search` の query 引数

`query` パラメータは以下のいずれかを受け付けます：

- **DSL 文字列**（例: `"title:hello"`、`"content:\"memory safety\""`)
- **Lexical クエリオブジェクト**（`TermQuery`、`PhraseQuery`、`BooleanQuery` など）
- **Vector クエリオブジェクト**（`VectorQuery`、`VectorTextQuery`）
- **`SearchRequest`**（完全な制御が必要な場合）

---

## Schema

`Index` のフィールドとインデックスタイプを定義します。

```python
class Schema:
    def __init__(self) -> None: ...
```

### フィールドメソッド

| メソッド | 説明 |
| :--- | :--- |
| `add_text_field(name, *, stored=True, indexed=True, term_vectors=False, analyzer=None)` | 全文フィールド（転置インデックス、BM25）。 |
| `add_integer_field(name, *, stored=True, indexed=True, multi_valued=False)` | 64 ビット整数フィールド。`multi_valued=True` で整数配列を受け付け（範囲クエリは "any match"）。 |
| `add_float_field(name, *, stored=True, indexed=True, multi_valued=False)` | 64 ビット浮動小数点フィールド。`multi_valued=True` で浮動小数点配列を受け付け（範囲クエリは "any match"）。 |
| `add_boolean_field(name, *, stored=True, indexed=True)` | ブールフィールド。 |
| `add_bytes_field(name, *, stored=True)` | 生バイトフィールド。 |
| `add_geo_field(name, *, stored=True, indexed=True)` | 地理座標フィールド（緯度/経度）。 |
| `add_geo3d_field(name, *, stored=True, indexed=True)` | 3D ECEF カルテシアン座標フィールド（x, y, z はメートル）。詳細は [Geo3d の概念](../concepts/geo3d.md)。 |
| `add_datetime_field(name, *, stored=True, indexed=True)` | UTC 日時フィールド。 |
| `add_hnsw_field(name, dimension, *, distance="cosine", m=16, ef_construction=200, embedder=None)` | HNSW 近似最近傍ベクトルフィールド。 |
| `add_flat_field(name, dimension, *, distance="cosine", embedder=None)` | Flat（総当たり）ベクトルフィールド。 |
| `add_ivf_field(name, dimension, *, distance="cosine", n_clusters=100, n_probe=1, embedder=None)` | IVF 近似最近傍ベクトルフィールド。 |

### その他のメソッド

| メソッド | 説明 |
| :--- | :--- |
| `add_embedder(name, config)` | 名前付きエンベダー定義を登録します。`config` は `"type"` キーを持つ辞書です（下記参照）。 |
| `set_default_fields(fields)` | デフォルト検索フィールドを設定（文字列のリスト）。 |
| `set_dynamic_field_policy(policy)` | 未宣言フィールドの扱いを設定。`policy` は `"strict"` / `"dynamic"`（デフォルト）/ `"ignore"`。詳細は下記を参照。 |
| `dynamic_field_policy()` | 現在のポリシーを小文字の文字列で返す。 |
| `field_names()` | 全フィールド名を返す。 |

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

```python
TermQuery(field: str, term: str)
```

指定フィールドに完全一致する語句を含むドキュメントを検索します。

### PhraseQuery

```python
PhraseQuery(field: str, terms: list[str])
```

指定した語句が順序どおりに含まれるドキュメントを検索します。

### FuzzyQuery

```python
FuzzyQuery(field: str, term: str, *, max_edits: int = 2)
```

編集距離が `max_edits` 以内の近似一致を検索します。`max_edits` はキーワード専用引数です。

### WildcardQuery

```python
WildcardQuery(field: str, pattern: str)
```

ワイルドカードパターン検索。`*` は任意の文字列、`?` は任意の1文字に一致します。

### NumericRangeQuery

```python
NumericRangeQuery(field: str, *, min: int | float | None = None, max: int | float | None = None)
```

`[min, max]` の範囲内の数値を検索します。開いた境界には `None` を指定する
（または省略する）と開放されます。`min` と `max` はキーワード専用引数です。
数値型（整数または浮動小数点）は `min`/`max` の Python 型から推論されます。

### GeoDistanceQuery

```python
GeoDistanceQuery.within_radius(
    field: str, lat: float, lon: float, distance_km: float,
)
```

地理的距離検索（半径指定）。指定した地点から `distance_km` キロメートル以内の
`(lat, lon)` 座標を持つドキュメントを返します。

### GeoBoundingBoxQuery

```python
GeoBoundingBoxQuery.within_bounding_box(
    field: str,
    min_lat: float, min_lon: float,
    max_lat: float, max_lon: float,
)
```

地理的範囲（バウンディングボックス）検索。軸並行 `[min_lat, max_lat] ×
[min_lon, max_lon]` 内の `(lat, lon)` 座標を持つドキュメントを返します。

### Geo3dDistanceQuery

```python
Geo3dDistanceQuery.within_sphere(
    field: str, x: float, y: float, z: float, radius_m: float,
)
```

3D ECEF 座標フィールドへの球距離検索。中心 `(x, y, z)` から `radius_m` メートル以内
の座標を持つドキュメントを返します。ECEF の理論については
[Geo3d の概念](../concepts/geo3d.md) を参照。

### Geo3dBoundingBoxQuery

```python
Geo3dBoundingBoxQuery.within_box(
    field: str,
    min_x: float, min_y: float, min_z: float,
    max_x: float, max_y: float, max_z: float,
)
```

軸並行 3D 範囲（AABB）検索。`[min_x, max_x] × [min_y, max_y] × [min_z, max_z]` 内
にある ECEF 座標を持つドキュメントを返します。

### Geo3dNearestQuery

```python
Geo3dNearestQuery.k_nearest(
    field: str,
    x: float, y: float, z: float,
    k: int,
    *,
    initial_radius_m: float | None = None,
    max_radius_m: float | None = None,
)
```

3D ECEF 座標フィールドへの k 最近傍検索。`(x, y, z)` から最も近い `k` 件のドキュ
メントを返します。`initial_radius_m` / `max_radius_m` は反復拡張サーチの探索コーン
を調整します。

### BooleanQuery

```python
bq = BooleanQuery()
bq.must(query)
bq.should(query)
bq.must_not(query)
```

複合ブールクエリ。引数なしでコンストラクタを呼び出し、`must` / `should` /
`must_not` メソッドで節を一つずつ追加します。各メソッドは任意のクエリ
オブジェクト（ネストされた `BooleanQuery` も含む）を受け付けます。

`must` 節はすべて一致する必要があり、`must_not` 節は一致してはなりません。
`should` 節はスコアリングに寄与し、`must` 節が無い場合は少なくとも1つが
一致する必要があります。

### SpanQuery

```python
# 単一語句
SpanQuery.term(field: str, term: str)

# Near: slop 位置以内の語句
SpanQuery.near(field: str, terms: list[str], *, slop: int = 0, ordered: bool = True)

# ネストされた SpanQuery 句を使った Near
SpanQuery.near_spans(field: str, clauses: list[SpanQuery], *, slop: int = 0, ordered: bool = True)

# Containing: big スパンが little スパンを含む
SpanQuery.containing(field: str, big: SpanQuery, little: SpanQuery)

# Within: 最大距離での include スパンと exclude スパン
SpanQuery.within(field: str, include: SpanQuery, exclude: SpanQuery, distance: int)
```

位置・近接スパンクエリ。静的ファクトリメソッドで構築します。`near` は語句
文字列のリストを受け取り、`near_spans` はネスト式のために `SpanQuery`
オブジェクトのリストを受け取ります。`slop` と `ordered` はキーワード専用
引数です。

### VectorQuery

```python
VectorQuery(field: str, vector: list[float])
```

事前計算済みエンベディングベクトルを使った近似最近傍検索を行います。

### VectorTextQuery

```python
VectorTextQuery(field: str, text: str)
```

クエリ時に `text` をエンベディングに変換してベクトル検索を行います。インデックスにエンベダーの設定が必要です。

---

## SearchRequest

高度な制御が必要な場合の完全なリクエストクラスです。

```python
class SearchRequest:
    def __init__(
        self,
        *,
        query=None,
        lexical_query=None,
        vector_query=None,
        filter_query=None,
        fusion=None,
        limit: int = 10,
        offset: int = 0,
    ) -> None: ...
```

| パラメータ | 説明 |
| :--- | :--- |
| `query` | DSL 文字列または単一クエリオブジェクト。`lexical_query` / `vector_query` と排他的。 |
| `lexical_query` | 明示的なハイブリッド検索の Lexical コンポーネント。 |
| `vector_query` | 明示的なハイブリッド検索の Vector コンポーネント。 |
| `filter_query` | スコアリング後に適用する Lexical フィルター。 |
| `fusion` | フュージョンアルゴリズム（`RRF` または `WeightedSum`）。両コンポーネント指定時のデフォルトは `RRF(k=60)`。 |
| `limit` | 最大結果件数（デフォルト 10）。 |
| `offset` | ページネーションオフセット（デフォルト 0）。 |

---

## SearchResult

`Index.search()` が返すクラスです。

```python
class SearchResult:
    id: str          # 外部ドキュメント識別子
    score: float     # 関連性スコア
    document: dict | None  # 取得されたフィールド値。stored=False の場合は None
```

---

## フュージョンアルゴリズム

### RRF

```python
RRF(k: float = 60.0)
```

逆順位フュージョン（Reciprocal Rank Fusion）。Lexical と Vector の結果リストを順位位置によってマージします。`k` は平滑化定数で、値が大きいほど上位ランクの影響が小さくなります。

### WeightedSum

```python
WeightedSum(lexical_weight: float = 0.5, vector_weight: float = 0.5)
```

両スコアリストをそれぞれ正規化した後、`lexical_weight * lexical_score + vector_weight * vector_score` として結合します。

---

## テキスト解析

### SynonymDictionary

```python
class SynonymDictionary:
    def __init__(self) -> None: ...
    def add_synonym_group(self, synonyms: list[str]) -> None: ...
```

### WhitespaceTokenizer

```python
class WhitespaceTokenizer:
    def __init__(self) -> None: ...
    def tokenize(self, text: str) -> list[Token]: ...
```

### SynonymGraphFilter

```python
class SynonymGraphFilter:
    def __init__(
        self,
        dictionary: SynonymDictionary,
        keep_original: bool = True,
        boost: float = 1.0,
    ) -> None: ...
    def apply(self, tokens: list[Token]) -> list[Token]: ...
```

### Token

```python
class Token:
    text: str
    position: int
    start_offset: int
    end_offset: int
    boost: float
    stopped: bool
    position_increment: int
    position_length: int
```

---

## フィールド値の型マッピング

Python の値は自動的に Laurus の `DataValue` 型に変換されます：

| Python 型 | Laurus 型 | 備考 |
| :--- | :--- | :--- |
| `None` | `Null` | |
| `bool` | `Bool` | `int` より先にチェック |
| `int` | `Int64` | |
| `float` | `Float64` | |
| `str` | `Text` | |
| `bytes` | `Bytes` | |
| `list[float]` | `Vector` | 要素は `f32` に変換 |
| `(lat, lon)` タプル | `Geo` | 2 つの `float` 値 |
| `datetime.datetime` | `DateTime` | `isoformat()` 経由で変換 |
