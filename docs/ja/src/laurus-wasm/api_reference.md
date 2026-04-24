# API リファレンス

## Index

検索インデックスの作成・クエリを行うメインエントリポイントです。

### 静的メソッド

#### `Index.create(schema?)`

新しいインメモリ（一時）インデックスを作成します。

- **引数:**
  - `schema` (Schema, 省略可) -- スキーマ定義
- **戻り値:** `Promise<Index>`

#### `Index.open(name, schema?)`

OPFS で永続化されたインデックスを開くか、新規作成します。

- **引数:**
  - `name` (string) -- インデックス名（OPFS サブディレクトリ）
  - `schema` (Schema, 省略可) -- スキーマ定義
- **戻り値:** `Promise<Index>`

### インスタンスメソッド

#### `putDocument(id, document)`

ドキュメントを置換（upsert）します。

- **引数:**
  - `id` (string) -- ドキュメント識別子
  - `document` (object) -- スキーマフィールドに対応するキーバリューペア
- **戻り値:** `Promise<void>`

#### `addDocument(id, document)`

ドキュメントバージョンを追加します（マルチバージョン RAG パターン）。

- **引数・戻り値:** `putDocument` と同じ

#### `getDocuments(id)`

ドキュメントの全バージョンを取得します。

- **引数:** `id` (string)
- **戻り値:** `Promise<object[]>`

#### `deleteDocuments(id)`

ドキュメントの全バージョンを削除します。

- **引数:** `id` (string)
- **戻り値:** `Promise<void>`

#### `commit()`

書き込みをフラッシュし、変更を検索可能にします。
`Index.open()` で作成したインデックスの場合、OPFS にも自動永続化されます。

- **戻り値:** `Promise<void>`

#### `search(query, limit?, offset?)`

DSL 文字列クエリで検索します。

- **引数:**
  - `query` (string) -- クエリ DSL（例: `"title:hello"`）
  - `limit` (number, デフォルト 10)
  - `offset` (number, デフォルト 0)
- **戻り値:** `Promise<SearchResult[]>`

#### `searchTerm(field, term, limit?, offset?)`

完全一致タームで検索します。

- **引数:**
  - `field` (string) -- フィールド名
  - `term` (string) -- 検索ターム
  - `limit`, `offset` (number, 省略可)
- **戻り値:** `Promise<SearchResult[]>`

#### `searchVector(field, vector, limit?, offset?)`

ベクトル類似度で検索します。

- **引数:**
  - `field` (string) -- ベクトルフィールド名
  - `vector` (number[]) -- クエリ埋め込みベクトル
  - `limit`, `offset` (number, 省略可)
- **戻り値:** `Promise<SearchResult[]>`

#### `searchVectorText(field, text, limit?, offset?)`

テキストで検索します（登録された埋め込み器で変換）。

- **引数:**
  - `field` (string) -- ベクトルフィールド名
  - `text` (string) -- 埋め込み対象テキスト
  - `limit`, `offset` (number, 省略可)
- **戻り値:** `Promise<SearchResult[]>`

#### `stats()`

インデックス統計を返します。

- **戻り値:** `{ documentCount: number, vectorFields: { [name]: { count, dimension } } }`

## Schema

インデックスフィールドと埋め込み器を定義するビルダーです。

### コンストラクタ

#### `new Schema()`

空のスキーマを作成します。

### メソッド

#### `addTextField(name, stored?, indexed?, termVectors?, analyzer?)`

全文検索テキストフィールドを追加します。

#### `addIntegerField(name, stored?, indexed?)`

整数フィールドを追加します。

#### `addFloatField(name, stored?, indexed?)`

浮動小数点フィールドを追加します。

#### `addBooleanField(name, stored?, indexed?)`

真偽値フィールドを追加します。

#### `addDateTimeField(name, stored?, indexed?)`

日時フィールドを追加します。

#### `addGeoField(name, stored?, indexed?)`

地理座標フィールドを追加します。

#### `addBytesField(name, stored?)`

バイナリデータフィールドを追加します。

#### `addHnswField(name, dimension, distance?, m?, efConstruction?, embedder?)`

HNSW ベクトルインデックスフィールドを追加します。

- `distance`: `"cosine"`（デフォルト）、`"euclidean"`、`"dot_product"`、`"manhattan"`、`"angular"`

#### `addFlatField(name, dimension, distance?, embedder?)`

全探索ベクトルインデックスフィールドを追加します。

#### `addIvfField(name, dimension, distance?, nClusters?, nProbe?, embedder?)`

IVF ベクトルインデックスフィールドを追加します。

#### `addEmbedder(name, config)`

名前付き埋め込み器を登録します。WASM では `"precomputed"` のみ対応しています。

#### `setDefaultFields(fields)`

デフォルト検索フィールドを設定します。

#### `setDynamicFieldPolicy(policy)`

ドキュメントに含まれるがスキーマに宣言されていないフィールドの扱いを設定します。`policy` は `"strict"` / `"dynamic"`（デフォルト）/ `"ignore"` のいずれか（大文字小文字を無視）。不正な値を渡すと例外をスローします。

- `"strict"` — ドキュメントを拒否
- `"dynamic"` — 各未宣言フィールドの型を推論してスキーマに追加。**警告**: integer フィールドに入ってきた float 値は静かに切り捨てられます（`3.14` → `3`）
- `"ignore"` — 未宣言フィールドを静かに破棄

詳細な挙動マトリクスは [スキーマとフィールド](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。

#### `dynamicFieldPolicy()`

現在のポリシーを小文字の文字列で返します。

#### `fieldNames()`

定義済みフィールド名の配列を返します。

## SearchResult

```typescript
interface SearchResult {
  id: string;
  score: number;
  document: object | null;
}
```
