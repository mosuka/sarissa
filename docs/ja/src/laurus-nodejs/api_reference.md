# API リファレンス

## Index

主要なエントリポイント。Laurus 検索エンジンをラップします。

```typescript
class Index {
  static create(
    path?: string | null,
    schema?: Schema,
  ): Promise<Index>;
}
```

### ファクトリメソッド

| パラメータ | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `path` | `string \| null` | `null` | 永続化ストレージのディレクトリ。`null` でインメモリ。 |
| `schema` | `Schema` | 空 | スキーマ定義。 |

### メソッド

| メソッド | 説明 |
| :--- | :--- |
| `putDocument(id, doc)` | ドキュメントを上書き保存。 |
| `addDocument(id, doc)` | 既存バージョンを残してチャンクを追記。 |
| `getDocuments(id)` | 指定 ID の全バージョンを取得。 |
| `deleteDocuments(id)` | 指定 ID の全バージョンを削除。 |
| `commit()` | 書き込みをフラッシュし変更を検索可能にする。 |
| `search(query, limit?, offset?)` | DSL 文字列で検索。 |
| `searchTerm(field, term, limit?, offset?)` | 完全一致 Term 検索。 |
| `searchVector(field, vector, limit?, offset?)` | 事前計算ベクトルで検索。 |
| `searchVectorText(field, text, limit?, offset?)` | テキストを自動埋め込みして検索。 |
| `searchWithRequest(request)` | `SearchRequest` で検索。 |
| `stats()` | インデックス統計を返す。 |

ドキュメント操作と検索メソッドはすべて非同期で Promise を返します。
`stats()` は同期メソッドです。

---

## Schema

`Index` のフィールドとインデックス型を定義します。

```typescript
class Schema {
  constructor();
}
```

### フィールドメソッド

| メソッド | 説明 |
| :--- | :--- |
| `addTextField(name, stored?, indexed?, termVectors?, analyzer?)` | 全文検索フィールド（転置インデックス、BM25）。 |
| `addIntegerField(name, stored?, indexed?, multiValued?)` | 64 ビット整数フィールド。`multiValued: true` で整数配列を受け付け（範囲クエリは "any match"）。 |
| `addFloatField(name, stored?, indexed?, multiValued?)` | 64 ビット浮動小数点フィールド。`multiValued: true` で浮動小数点配列を受け付け（範囲クエリは "any match"）。 |
| `addBooleanField(name, stored?, indexed?)` | 真偽値フィールド。 |
| `addBytesField(name, stored?)` | バイナリデータフィールド。 |
| `addGeoField(name, stored?, indexed?)` | 地理座標フィールド。 |
| `addGeo3dField(name, stored?, indexed?)` | 3D ECEF カルテシアン座標フィールド（x, y, z はメートル）。詳細は [Geo3d の概念](../concepts/geo3d.md)。 |
| `addDatetimeField(name, stored?, indexed?)` | UTC 日時フィールド。 |
| `addHnswField(name, dimension, distance?, m?, efConstruction?, embedder?)` | HNSW ベクトルフィールド。 |
| `addFlatField(name, dimension, distance?, embedder?)` | Flat（全探索）ベクトルフィールド。 |
| `addIvfField(name, dimension, distance?, nClusters?, nProbe?, embedder?)` | IVF ベクトルフィールド。 |
| `addEmbedder(name, config)` | 名前付き Embedder を登録。 |
| `setDefaultFields(fields)` | デフォルト検索フィールドを設定。 |
| `setDynamicFieldPolicy(policy)` | 未宣言フィールドの扱いを設定。`policy` は `"strict"` / `"dynamic"`（デフォルト）/ `"ignore"`。詳細は下記を参照。 |
| `dynamicFieldPolicy()` | 現在のポリシーを小文字の文字列で返す。 |
| `fieldNames()` | 全フィールド名を返す。 |

#### Dynamic field policy（動的フィールドポリシー）

ドキュメントに含まれるがスキーマに宣言されていないフィールドの扱いを制御します:

- `"strict"` — ドキュメントを拒否
- `"dynamic"`（デフォルト）— 各未宣言フィールドの型を推論してスキーマに追加。**警告**: integer フィールドに入ってきた float 値は静かに切り捨てられます（`3.14` → `3`）。厳密さが必要なら `"strict"` を使用してください
- `"ignore"` — 未宣言フィールドを静かに破棄

詳細な挙動マトリクスは [スキーマとフィールド](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。

### 距離指標

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

```typescript
new TermQuery(field: string, term: string)
```

指定フィールドで完全一致する Term を含むドキュメントにマッチ。

### PhraseQuery

```typescript
new PhraseQuery(field: string, terms: string[])
```

指定順序で Term を含むドキュメントにマッチ。

### FuzzyQuery

```typescript
new FuzzyQuery(field: string, term: string, maxEdits?: number)
```

最大 `maxEdits` 編集距離までの近似マッチ（デフォルト 2）。

### WildcardQuery

```typescript
new WildcardQuery(field: string, pattern: string)
```

パターンマッチ。`*` は任意の文字列、`?` は任意の1文字。

### NumericRangeQuery

```typescript
new NumericRangeQuery(
  field: string,
  min?: number | null,
  max?: number | null,
  isFloat?: boolean,
)
```

`[min, max]` 範囲の数値にマッチ。`null` で開放端。

### GeoQuery

```typescript
GeoQuery.withinRadius(
  field: string, lat: number, lon: number, distanceKm: number,
): GeoQuery

GeoQuery.withinBoundingBox(
  field: string,
  minLat: number, minLon: number,
  maxLat: number, maxLon: number,
): GeoQuery
```

半径またはバウンディングボックスによる地理検索。

### Geo3dDistanceQuery

```typescript
Geo3dDistanceQuery.withinSphere(
  field: string,
  x: number, y: number, z: number,
  radiusM: number,
): Geo3dDistanceQuery
```

3D ECEF 座標フィールドへの球距離検索。中心から `radiusM` メートル以内の `(x, y, z)`
座標を持つドキュメントを返します。ECEF の理論については
[Geo3d の概念](../concepts/geo3d.md) を参照。

### Geo3dBoundingBoxQuery

```typescript
Geo3dBoundingBoxQuery.withinBox(
  field: string,
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
): Geo3dBoundingBoxQuery
```

軸並行 3D 範囲（AABB）検索。

### Geo3dNearestQuery

```typescript
Geo3dNearestQuery.kNearest(
  field: string,
  x: number, y: number, z: number,
  k: number,
  initialRadiusM?: number,
  maxRadiusM?: number,
): Geo3dNearestQuery
```

3D ECEF 座標フィールドへの k 最近傍検索。`initialRadiusM` / `maxRadiusM` は
反復拡張サーチの探索コーンを調整します（Node.js 専用 — バインディング間の対称化は
[#344] を参照）。

[#344]: https://github.com/mosuka/laurus/issues/344

### BooleanQuery

```typescript
class BooleanQuery {
  constructor();
  mustTerm(field: string, term: string): void;
  shouldTerm(field: string, term: string): void;
  mustNotTerm(field: string, term: string): void;
}
```

MUST / SHOULD / MUST_NOT 句による複合ブーリアンクエリ。

### SpanQuery

```typescript
SpanQuery.term(field: string, term: string): SpanQuery
SpanQuery.near(
  field: string, terms: string[],
  slop?: number, ordered?: boolean,
): SpanQuery
SpanQuery.nearSpans(
  field: string, clauses: SpanQuery[],
  slop?: number, ordered?: boolean,
): SpanQuery
SpanQuery.containing(
  field: string, big: SpanQuery, little: SpanQuery,
): SpanQuery
SpanQuery.within(
  field: string,
  include: SpanQuery, exclude: SpanQuery, distance: number,
): SpanQuery
```

位置・近接ベースのスパンクエリ。

### VectorQuery

```typescript
new VectorQuery(field: string, vector: number[])
```

事前計算済み埋め込みベクトルによる最近傍検索。

### VectorTextQuery

```typescript
new VectorTextQuery(field: string, text: string)
```

クエリ時にテキストを埋め込みに変換して検索。
インデックスに Embedder の設定が必要。

---

## SearchRequest

高度な制御のための全機能検索リクエスト。

```typescript
class SearchRequest {
  constructor(limit?: number, offset?: number);
}
```

### セッターメソッド

| メソッド | 説明 |
| :--- | :--- |
| `setQueryDsl(dsl)` | DSL 文字列クエリを設定。 |
| `setLexicalTermQuery(field, term)` | Term ベースの Lexical クエリを設定。 |
| `setLexicalPhraseQuery(field, terms)` | Phrase ベースの Lexical クエリを設定。 |
| `setVectorQuery(field, vector)` | 事前計算ベクトルクエリを設定。 |
| `setVectorTextQuery(field, text)` | テキストベースのベクトルクエリを設定。 |
| `setFilterQuery(field, term)` | スコアリング後のフィルタを設定。 |
| `setRrfFusion(k?)` | RRF 融合を使用（デフォルト k=60）。 |
| `setWeightedSumFusion(lexicalWeight?, vectorWeight?)` | 加重和融合を使用。 |

---

## SearchResult

検索メソッドが配列として返す結果。

```typescript
interface SearchResult {
  id: string;        // 外部ドキュメント識別子
  score: number;     // 関連度スコア
  document: object | null; // 取得フィールド、または null
}
```

---

## 融合アルゴリズム

### RRF

```typescript
new RRF(k?: number)  // デフォルト 60.0
```

Reciprocal Rank Fusion。ランク位置で Lexical と Vector の
結果リストを統合。

### WeightedSum

```typescript
new WeightedSum(
  lexicalWeight?: number,  // デフォルト 0.5
  vectorWeight?: number,   // デフォルト 0.5
)
```

両スコアリストを個別に正規化し、加重和で結合。

---

## テキスト解析

### SynonymDictionary

```typescript
class SynonymDictionary {
  constructor();
  addSynonymGroup(terms: string[]): void;
}
```

### WhitespaceTokenizer

```typescript
class WhitespaceTokenizer {
  constructor();
  tokenize(text: string): Token[];
}
```

### SynonymGraphFilter

```typescript
class SynonymGraphFilter {
  constructor(
    dictionary: SynonymDictionary,
    keepOriginal?: boolean,  // デフォルト true
    boost?: number,          // デフォルト 1.0
  );
  apply(tokens: Token[]): Token[];
}
```

### Token

```typescript
interface Token {
  text: string;
  position: number;
  startOffset: number;
  endOffset: number;
  boost: number;
  stopped: boolean;
  positionIncrement: number;
  positionLength: number;
}
```

---

## フィールド値の型

JavaScript の値は自動的に Laurus の `DataValue` 型に変換されます:

| JavaScript 型 | Laurus 型 | 備考 |
| :--- | :--- | :--- |
| `null` | `Null` | |
| `boolean` | `Bool` | |
| `number`（整数） | `Int64` | |
| `number`（浮動小数点） | `Float64` | |
| `string` | `Text` | ISO8601 文字列は `DateTime` になる |
| `number[]` | `Vector` | `f32` に変換 |
| `{ lat, lon }` | `Geo` | 2つの `number` 値 |
| `Date` | `DateTime` | タイムスタンプ経由 |
| `Buffer` | `Bytes` | |
