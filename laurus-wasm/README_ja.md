# laurus-wasm

[Laurus](https://github.com/mosuka/laurus) 検索ライブラリの
WebAssembly バインディング —
ブラウザ上で Lexical 検索、Vector 検索、Hybrid 検索を実行できます。

## 特徴

- **Lexical 検索** — BM25 スコアリングによる Term、Phrase、Fuzzy、
  Wildcard、Geo、Boolean、Span クエリ
- **Vector 検索** — HNSW、Flat、IVF インデックスと複数の距離メトリクス
- **Hybrid 検索** — Lexical 検索と Vector 検索を
  RRF または Weighted Sum フュージョンで組み合わせ
- **CJK 対応** — [Lindera](https://github.com/lindera/lindera) による日本語・中国語・韓国語のトークナイズ
- **OPFS 永続化** — ブラウザの Origin Private File System を使用し、
  ページリロード後もデータを保持
- **JS コールバック Embedder** — JavaScript コールバックで
  任意のエンベディング関数を提供可能（例: Transformers.js）

## クイックスタート

```javascript
import init, { Index, Schema } from "./pkg/laurus_wasm.js";

await init();

// スキーマ定義
const schema = new Schema();
schema.addTextField("title");
schema.addTextField("body");
schema.setDefaultFields(["title", "body"]);

// OPFS 永続化インデックスを作成（ページリロード後もデータ保持）
const index = await Index.open("my-index", schema);

// ドキュメントをインデックス
await index.putDocument("doc1", {
  title: "Rust プログラミング",
  body: "安全性と速度。",
});
await index.putDocument("doc2", {
  title: "Python 入門",
  body: "汎用的な言語。",
});
await index.commit();

// DSL 文字列で検索
const results = await index.search("programming", 5);
for (const r of results) {
  console.log(r.id, r.score, r.document.title);
}
```

## API 概要

### Index

```javascript
// インデックス作成（インメモリまたは OPFS 永続化）
const index = await Index.create(schema);              // インメモリ（揮発性）
const index = await Index.open("my-index", schema);    // OPFS（永続化）

// ドキュメント CRUD
await index.putDocument("id", { field: "value" });     // upsert
await index.addDocument("id", { field: "chunk" });     // 追記 (RAG)
const docs = await index.getDocuments("id");
await index.deleteDocuments("id");
await index.commit();                                  // フラッシュ + OPFS に永続化

// 検索
const results = await index.search("query DSL", limit, offset);
const results = await index.searchTerm("field", "term", limit);
const results = await index.searchVector("field", [0.1, ...], limit);
const results = await index.searchVectorText("field", "text", limit);

// 統計情報
const stats = index.stats();
// { documentCount: 42, vectorFields: {
//     embedding: { count: 42, dimension: 384 }
// } }
```

### Schema

```javascript
const schema = new Schema();
schema.addTextField("title", true, true, false, "lindera-ipadic");
schema.addIntegerField("year");
schema.addFloatField("price");
schema.addBooleanField("active");
schema.addDatetimeField("created_at");
schema.addGeoField("location");
schema.addBytesField("thumbnail");
schema.addHnswField("embedding", 384, "cosine", 16, 200, "minilm");
schema.addFlatField("embedding", 384);
schema.addIvfField("embedding", 384, "cosine", 100, 1);
schema.addEmbedder("minilm", {
  type: "callback",
  embed: async (text) => {
    // エンベディング関数を指定（例: Transformers.js）
    return [0.1, 0.2, ...];
  },
});
schema.setDefaultFields(["title", "body"]);
```

## サンプル

[examples/](examples/) ディレクトリに、Transformers.js エンベディングと
OPFS 永続化を使用したデモがあります。

## ソースからのビルド

```bash
cd laurus-wasm

# 開発ビルド
wasm-pack build --target web --dev

# リリースビルド
wasm-pack build --target web --release

# OPFS ヘルパ（./opfs サブパス）を pkg/ に同梱
./scripts/postbuild.sh

# デモの起動
python3 -m http.server 8080
# http://localhost:8080/examples/ を開く
```

ポストビルドスクリプトは `js/opfs.js` と `js/opfs.d.ts` を `pkg/` に
コピーし、利用者が
`import { downloadDictionary } from "laurus-wasm/opfs"` できるよう
`pkg/package.json` を書き換えます。`wasm-pack build` 後に毎回実行
してください（冪等です）。

## ライセンス

MIT
