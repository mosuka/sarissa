# アーキテクチャ

このページでは、Laurus の内部構造について説明します。アーキテクチャを理解することで、スキーマ設計、Analyzer の選択、検索戦略についてより適切な判断ができるようになります。

## プロジェクト構成

Laurus は Cargo workspace として 9 つのクレート（コアライブラリ 1、自製バイナリ 3、言語バインディング 5）で構成されています。

```mermaid
graph TB
    CLI["laurus-cli\n(Binary)\nCLI + REPL + serve + mcp"]
    SRV["laurus-server\n(Library + Binary)\ngRPC + HTTP Gateway"]
    MCP["laurus-mcp\n(Binary)\nMCP stdio server"]
    PY["laurus-python\n(cdylib)\nPyO3 / Maturin"]
    NJS["laurus-nodejs\n(cdylib)\nNAPI-RS"]
    WASM["laurus-wasm\n(WebAssembly)\nwasm-bindgen"]
    RB["laurus-ruby\n(cdylib)\nmagnus + rb-sys"]
    PHP["laurus-php\n(PHP extension)\next-php-rs"]
    LIB["laurus\n(Library)\nコア検索エンジン"]

    CLI --> LIB
    CLI --> SRV
    CLI --> MCP
    SRV --> LIB
    MCP --> SRV
    MCP --> LIB
    PY --> LIB
    NJS --> LIB
    WASM --> LIB
    RB --> LIB
    PHP --> LIB
```

| クレート | 種類 | 説明 |
| :--- | :--- | :--- |
| **laurus** | Library | コア検索エンジン -- Lexical 検索、Vector 検索、ハイブリッド検索 |
| **laurus-cli** | Binary | インデックス管理と検索のためのコマンドラインインターフェース |
| **laurus-server** | Library + Binary | オプションの HTTP/JSON ゲートウェイ付き gRPC サーバー |
| **laurus-mcp** | Binary | laurus-server へプロキシする MCP（Model Context Protocol）stdio サーバー |
| **laurus-python** | cdylib | PyO3 / Maturin による Python バインディング |
| **laurus-nodejs** | cdylib | NAPI-RS による Node.js バインディング |
| **laurus-wasm** | WebAssembly | wasm-bindgen によるブラウザ・エッジバインディング |
| **laurus-ruby** | cdylib | magnus / rb-sys による Ruby バインディング |
| **laurus-php** | PHP 拡張 | ext-php-rs による PHP バインディング（ワークスペースからは除外） |

各クレートの詳細については以下を参照してください。

- [ライブラリ概要](laurus.md)
- [CLI 概要](laurus-cli.md)
- [サーバー概要](laurus-server.md)
- [MCP サーバー概要](laurus-mcp.md)
- [Python バインディング概要](laurus-python.md)
- [Node.js バインディング概要](laurus-nodejs.md)
- [WASM バインディング概要](laurus-wasm.md)
- [Ruby バインディング概要](laurus-ruby.md)
- [PHP バインディング概要](laurus-php.md)

## 全体概要

Laurus は単一の `Engine` を中心に構成されており、4 つの内部コンポーネントを統括します。

```mermaid
graph TB
    subgraph Engine
        SCH["Schema"]
        LS["LexicalStore\n(Inverted Index)"]
        VS["VectorStore\n(HNSW / Flat / IVF)"]
        DL["DocumentLog\n(WAL + Document Storage)"]
    end

    Storage["Storage (trait)\nMemory / File / File+Mmap"]

    LS --- Storage
    VS --- Storage
    DL --- Storage
```

| コンポーネント | 役割 |
| :--- | :--- |
| **Schema** | フィールドとその型を宣言し、各フィールドのルーティング先を決定する |
| **LexicalStore** | キーワード検索のための転置インデックス（Inverted Index）（BM25 スコアリング） |
| **VectorStore** | 類似度検索のためのベクトルインデックス（Flat、HNSW、または IVF） |
| **DocumentLog** | 耐久性のための WAL（Write-Ahead Log）+ ドキュメントストレージ |

3 つのストアはすべて単一の `Storage` バックエンドを共有し、キープレフィックス（`lexical/`、`vector/`、`documents/`）によって分離されています。

## Engine のライフサイクル

### Engine の構築

`EngineBuilder` が各パーツから Engine を組み立てます。

```rust
let engine = Engine::builder(storage, schema)
    .analyzer(analyzer)      // optional: for text fields
    .embedder(embedder)      // optional: for vector fields
    .build()
    .await?;
```

```mermaid
sequenceDiagram
    participant User
    participant EngineBuilder
    participant Engine

    User->>EngineBuilder: new(storage, schema)
    User->>EngineBuilder: .analyzer(analyzer)
    User->>EngineBuilder: .embedder(embedder)
    User->>EngineBuilder: .build().await
    EngineBuilder->>EngineBuilder: split_schema()
    Note over EngineBuilder: Separate fields into\nLexicalIndexConfig\n+ VectorIndexConfig
    EngineBuilder->>Engine: Create LexicalStore
    EngineBuilder->>Engine: Create VectorStore
    EngineBuilder->>Engine: Create DocumentLog
    EngineBuilder->>Engine: Recover from WAL
    EngineBuilder-->>User: Engine ready
```

`build()` 時に Engine は以下の処理を行います。

1. **スキーマの分割** — Lexical フィールドは `LexicalIndexConfig` へ、Vector フィールドは `VectorIndexConfig` へ振り分けられる
2. **プレフィックス付きストレージの作成** — 各コンポーネントが分離された名前空間を取得する（`lexical/`、`vector/`、`documents/`）
3. **ストアの初期化** — `LexicalStore` と `VectorStore` がそれぞれの設定で構築される
4. **WAL からの復旧** — 前回のセッションからの未コミット操作を再生する

### スキーマの分割

`Schema` には Lexical フィールドと Vector フィールドの両方が含まれています。ビルド時に `split_schema()` がこれらを分離します。

```mermaid
graph LR
    S["Schema\ntitle: Text\nbody: Text\ncategory: Text\npage: Integer\ncontent_vec: HNSW"]

    S --> LC["LexicalIndexConfig\ntitle: TextOption\nbody: TextOption\ncategory: TextOption\npage: IntegerOption\n_id: KeywordAnalyzer"]

    S --> VC["VectorIndexConfig\ncontent_vec: HnswOption\n(dim=384, m=16, ef=200)"]
```

主なポイント:

- 予約フィールド `_id` は常に `KeywordAnalyzer`（完全一致）で Lexical 設定に追加される
- `PerFieldAnalyzer` はフィールドごとの Analyzer 設定をラップする。単純な `StandardAnalyzer` を渡した場合、すべてのテキストフィールドのデフォルトとなる
- `PerFieldEmbedder` も Vector フィールドに対して同様に動作する

## インデクシングのデータフロー

`engine.add_document(id, doc)` を呼び出した場合の処理:

```mermaid
sequenceDiagram
    participant User
    participant Engine
    participant WAL as DocumentLog (WAL)
    participant Lexical as LexicalStore
    participant Vector as VectorStore

    User->>Engine: add_document("doc-1", doc)
    Engine->>WAL: Append to WAL
    Engine->>Engine: Assign internal ID (u64)

    loop For each field in document
        alt Lexical field (text, integer, etc.)
            Engine->>Lexical: Analyze + index field
        else Vector field
            Engine->>Vector: Embed + index field
        end
    end

    Note over Engine: Document is buffered\nbut NOT yet searchable

    User->>Engine: commit()
    Engine->>Lexical: Flush segments to storage
    Engine->>Vector: Flush segments to storage
    Engine->>WAL: Truncate WAL
    Note over Engine: Documents are\nnow searchable
```

主なポイント:

- **WAL 優先**: すべての書き込みは、インメモリ構造を変更する前にログに記録される
- **デュアルインデクシング**: 各フィールドはスキーマに基づいて Lexical ストアまたは Vector ストアのいずれかにルーティングされる
- **コミットが必要**: ドキュメントは `commit()` の後にのみ検索可能になる

## 検索のデータフロー

`engine.search(request)` を呼び出した場合の処理:

```mermaid
sequenceDiagram
    participant User
    participant Engine
    participant Lexical as LexicalStore
    participant Vector as VectorStore
    participant Fusion

    User->>Engine: search(request)

    opt Filter query present
        Engine->>Lexical: Execute filter query
        Lexical-->>Engine: Allowed document IDs
    end

    par Lexical search
        Engine->>Lexical: Execute lexical query
        Lexical-->>Engine: Ranked hits (BM25)
    and Vector search
        Engine->>Vector: Execute vector query
        Vector-->>Engine: Ranked hits (similarity)
    end

    alt Both lexical and vector results
        Engine->>Fusion: Fuse results (RRF or WeightedSum)
        Fusion-->>Engine: Merged ranked list
    end

    Engine->>Engine: Apply offset + limit
    Engine-->>User: Vec of SearchResult
```

検索パイプラインは 3 つのステージで構成されています。

1. **フィルタ**（オプション） — Lexical インデックスに対してフィルタクエリを実行し、許可されたドキュメント ID のセットを取得する
2. **検索** — Lexical クエリと Vector クエリを並列に実行する
3. **フュージョン** — 両方のクエリタイプが存在する場合、RRF（デフォルト、k=60）または WeightedSum を使用して結果をマージする

## ストレージアーキテクチャ

すべてのコンポーネントは単一の `Storage` trait 実装を共有しますが、キープレフィックスを使用してデータを分離します。

```mermaid
graph TB
    Engine --> PS1["PrefixedStorage\nprefix: 'lexical/'"]
    Engine --> PS2["PrefixedStorage\nprefix: 'vector/'"]
    Engine --> PS3["PrefixedStorage\nprefix: 'documents/'"]

    PS1 --> S["Storage Backend\n(Memory / File / File+Mmap)"]
    PS2 --> S
    PS3 --> S
```

| バックエンド | 説明 | 最適な用途 |
| :--- | :--- | :--- |
| `MemoryStorage` | すべてのデータをメモリ上に保持 | テスト、小規模データセット、一時的な利用 |
| `FileStorage` | 標準的なファイル I/O | 一般的な本番利用 |
| `FileStorage` (mmap) | メモリマップドファイル（`use_mmap = true`） | 大規模データセット、読み取り負荷の高いワークロード |

## フィールドごとのディスパッチ

`PerFieldAnalyzer` が提供されている場合、Engine はフィールド固有の Analyzer に解析処理をディスパッチします。同様のパターンが `PerFieldEmbedder` にも適用されます。

```mermaid
graph LR
    PFA["PerFieldAnalyzer"]
    PFA -->|"title"| KA["KeywordAnalyzer"]
    PFA -->|"body"| SA["StandardAnalyzer"]
    PFA -->|"description"| JA["JapaneseAnalyzer"]
    PFA -->|"_id"| KA2["KeywordAnalyzer\n(always)"]
    PFA -->|other fields| DEF["Default Analyzer\n(StandardAnalyzer)"]
```

これにより、同一 Engine 内で異なるフィールドに異なる解析戦略を使用できます。

## まとめ

| 項目 | 詳細 |
| :--- | :--- |
| **コア構造体** | `Engine` — すべての操作を統括する |
| **ビルダー** | `EngineBuilder` — Storage + Schema + Analyzer + Embedder から Engine を組み立てる |
| **スキーマ分割** | Lexical フィールド → `LexicalIndexConfig`、Vector フィールド → `VectorIndexConfig` |
| **書き込みパス** | WAL → インメモリバッファ → `commit()` → 永続ストレージ |
| **読み取りパス** | クエリ → 並列 Lexical/Vector 検索 → フュージョン → ランク付き結果 |
| **ストレージ分離** | `PrefixedStorage` による `lexical/`、`vector/`、`documents/` プレフィックス |
| **フィールドごとのディスパッチ** | `PerFieldAnalyzer` と `PerFieldEmbedder` がフィールド固有の実装にルーティングする |

## 次のステップ

- フィールドタイプとスキーマ設計を理解する: [スキーマとフィールド](concepts/schema_and_fields.md)
- テキスト解析について学ぶ: [テキスト解析](concepts/analysis.md)
- Embedding について学ぶ: [Embedding](concepts/embedding.md)
