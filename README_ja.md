# Laurus : Lexical Augmented Unified Retrieval Using Semantics

[![Crates.io](https://img.shields.io/crates/v/laurus.svg)](https://crates.io/crates/laurus)
[![Documentation](https://docs.rs/laurus/badge.svg)](https://docs.rs/laurus)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Laurus は Rust で書かれた検索プラットフォームです — Lexical Augmented Unified Retrieval Using Semantics のために構築されています。
Lexical 検索、ベクトル検索、ハイブリッド検索をカバーするコアライブラリを基盤に、すぐに使える複数のインターフェースを提供します:

- **コアライブラリ** — アプリケーションに組み込み可能なモジュール式検索エンジン
- **CLI & REPL** — インタラクティブな検索体験を実現するコマンドラインツール
- **gRPC サーバー & HTTP Gateway** — マイクロサービスや既存システムへのシームレスな統合
- **MCP サーバー** — Claude などの AI アシスタントとの直接連携
- **Python バインディング** — データサイエンスや AI ワークフローで利用できるネイティブ Python パッケージ
- **Node.js バインディング** — サーバーサイド JavaScript アプリケーション向けネイティブ Node.js アドオン
- **WebAssembly** — wasm-bindgen によるブラウザおよびエッジランタイムサポート
- **Ruby バインディング** — Rails や Ruby アプリケーション向けネイティブ Ruby gem
- **PHP バインディング** — Web アプリケーションや CLI ツール向けネイティブ PHP 拡張

ライブラリとして組み込むことも、スタンドアロンのサーバーとしてデプロイすることも、Python / Node.js / Ruby / PHP から呼び出すことも、ブラウザで WASM として実行することも、AI ワークフローに検索機能を組み込むことも可能な、コンポーザブルな検索基盤です。

## ライブデモ

すべてのサンプルが WebAssembly でブラウザ上で完結する形で動作します。インストール不要で試せます:

**[https://mosuka.github.io/laurus/demo/](https://mosuka.github.io/laurus/demo/)**

| サンプル | 内容 |
| --- | --- |
| [basic](https://mosuka.github.io/laurus/demo/basic/) | 統一された Query DSL による日本語の全文検索・ベクトル検索・ハイブリッド検索 |
| [geo](https://mosuka.github.io/laurus/demo/geo/) | Leaflet マップ上の東京 POI を、ビューポートのバウンディングボックス + テキスト + ベクトルで検索 |
| [geo3d](https://mosuka.github.io/laurus/demo/geo3d/) | CesiumJS 3D 地球儀上の航空機をリアルタイム表示。`geo3d_bbox` / `geo3d_nearest`（高度を含む真の ECEF 3D）を活用 |

## ドキュメント

包括的なドキュメントがオンラインで利用できます:

- **English**: [https://mosuka.github.io/laurus/](https://mosuka.github.io/laurus/)
- **日本語**: [https://mosuka.github.io/laurus/ja/](https://mosuka.github.io/laurus/ja/)

### 目次

- **はじめに**
  - [インストール](https://mosuka.github.io/laurus/ja/getting_started/installation.html)
  - [クイックスタート](https://mosuka.github.io/laurus/ja/getting_started/quickstart.html)
  - [サンプル](https://mosuka.github.io/laurus/ja/getting_started/examples.html)
- **コアコンセプト**
  - [スキーマとフィールド](https://mosuka.github.io/laurus/ja/concepts/schema_and_fields.html)
  - [テキスト解析](https://mosuka.github.io/laurus/ja/concepts/analysis.html)
  - [エンベディング](https://mosuka.github.io/laurus/ja/concepts/embedding.html)
  - [ストレージ](https://mosuka.github.io/laurus/ja/concepts/storage.html)
  - [インデックス](https://mosuka.github.io/laurus/ja/concepts/indexing.html)（Lexical / Vector）
  - [検索](https://mosuka.github.io/laurus/ja/concepts/search.html)（Lexical / Vector / ハイブリッド）
  - [Query DSL](https://mosuka.github.io/laurus/ja/concepts/query_dsl.html)
- **クレートガイド**
  - [laurus（ライブラリ）](https://mosuka.github.io/laurus/ja/laurus.html) — Engine、スコアリング、ファセット、ハイライト、スペル修正、永続化 & WAL
  - [laurus-cli](https://mosuka.github.io/laurus/ja/laurus-cli.html) — コマンドラインインターフェース、REPL、スキーマフォーマット
  - [laurus-server](https://mosuka.github.io/laurus/ja/laurus-server.html) — gRPC サーバー、HTTP Gateway、設定
  - [laurus-mcp](https://mosuka.github.io/laurus/ja/laurus-mcp.html) — AI アシスタント（Claude など）向け MCP サーバー
  - [laurus-python](https://mosuka.github.io/laurus/ja/laurus-python.html) — Python バインディング（PyPI パッケージ）
  - [laurus-nodejs](https://mosuka.github.io/laurus/ja/laurus-nodejs.html) — Node.js バインディング（npm パッケージ）
  - [laurus-wasm](https://mosuka.github.io/laurus/ja/laurus-wasm.html) — WebAssembly バインディング（npm パッケージ）
  - [laurus-ruby](https://mosuka.github.io/laurus/ja/laurus-ruby.html) — Ruby バインディング（RubyGems パッケージ）
  - [laurus-php](https://mosuka.github.io/laurus/ja/laurus-php.html) — PHP バインディング（PHP 拡張）
- **開発**
  - [ビルドとテスト](https://mosuka.github.io/laurus/ja/development/build_and_test.html)
  - [フィーチャーフラグ](https://mosuka.github.io/laurus/ja/development/feature_flags.html)
  - [プロジェクト構成](https://mosuka.github.io/laurus/ja/development/project_structure.html)
- [**API リファレンス (docs.rs)**](https://docs.rs/laurus)

## 特徴

- **Pure Rust 実装**: ゼロコスト抽象化によるメモリ安全かつ高速なパフォーマンス。
- **ハイブリッド検索**: BM25 Lexical 検索と HNSW ベクトル検索を設定可能なフュージョン戦略でシームレスに統合。
- **マルチモーダル対応**: CLIP エンベディングによるテキストから画像、画像から画像への検索をネイティブサポート。
- **豊富な Query DSL**: Term、Phrase、Boolean、Fuzzy、Wildcard、Range、2D / 3D Geographic（球、バウンディングボックス、k-NN）、Span クエリに対応。
- **柔軟な解析**: トークン化、正規化、ステミングの設定可能なパイプライン（[Lindera](https://github.com/lindera/lindera) による CJK サポートを含む）。
- **プラガブルストレージ**: インメモリ、ファイルシステム、メモリマップドストレージバックエンドのインターフェース。
- **動的スキーマ**: オプションのスキーマオンライト — 未宣言フィールドは値から型を推論し、`Strict` / `Dynamic` / `Ignore` ポリシーで細やかに制御可能。
- **多値数値フィールド**: Integer / Float フィールドは 1 ドキュメントあたり複数値を保持でき、Range クエリは「いずれかの値が条件を満たす」ドキュメントにマッチする（Lucene 互換の "any match" 意味論）。
- **スコアリングとランキング**: BM25 スコアリングとハイブリッド結果向けのカスタマイズ可能なフュージョン戦略。
- **ファセットとハイライト**: ファセットナビゲーションと検索結果ハイライトの組み込みサポート。
- **スペル修正**: スペルミスのあるクエリ用語の修正候補を提案。

## ワークスペース構成

Laurus は 9 つのクレートで構成された Cargo ワークスペースです:

| クレート | 説明 |
| --- | --- |
| [`laurus`](laurus/) | コア検索ライブラリ — スキーマ、解析、インデックス、検索、ストレージ |
| [`laurus-cli`](laurus-cli/) | 対話型検索のための REPL 付きコマンドラインインターフェース |
| [`laurus-server`](laurus-server/) | Laurus をサービスとしてデプロイするための HTTP Gateway 付き gRPC サーバー |
| [`laurus-mcp`](laurus-mcp/) | AI アシスタント（Claude など）向け stdio トランスポートの MCP サーバー |
| [`laurus-python`](laurus-python/) | PyO3 と Maturin で構築した Python バインディング（PyPI パッケージ） |
| [`laurus-nodejs`](laurus-nodejs/) | NAPI-RS で構築した Node.js バインディング（npm パッケージ） |
| [`laurus-wasm`](laurus-wasm/) | wasm-bindgen で構築した WebAssembly バインディング（npm パッケージ） |
| [`laurus-ruby`](laurus-ruby/) | magnus と rb-sys で構築した Ruby バインディング（RubyGems パッケージ） |
| [`laurus-php`](laurus-php/) | ext-php-rs で構築した PHP バインディング（PHP 拡張） |

## フィーチャーフラグ

`laurus` クレートはエンベディングサポート用のオプションのフィーチャーフラグを提供します:

| フィーチャー | 説明 |
| --- | --- |
| `embeddings-candle` | [Candle](https://github.com/huggingface/candle) によるローカル BERT エンベディング |
| `embeddings-openai` | OpenAI API によるクラウドベースのエンベディング |
| `embeddings-multimodal` | CLIP ベースのマルチモーダル（テキスト + 画像）エンベディング |
| `embeddings-all` | すべてのエンベディングバックエンドを有効化 |

## クイックスタート

```rust
use laurus::lexical::{TermQuery, TextOption};
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{Document, Engine, LexicalSearchRequest, Schema, SearchRequestBuilder};

#[tokio::main]
async fn main() -> laurus::Result<()> {
    // 1. ストレージの作成
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;

    // 2. スキーマの定義
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .build();

    // 3. エンジンの作成
    let engine = Engine::new(storage, schema).await?;

    // 4. ドキュメントのインデックス
    engine
        .add_document(
            "doc1",
            Document::builder()
                .add_text("title", "Introduction to Rust")
                .add_text(
                    "body",
                    "Rust is a systems programming language focused on safety and performance.",
                )
                .build(),
        )
        .await?;
    engine
        .add_document(
            "doc2",
            Document::builder()
                .add_text("title", "Python for Data Science")
                .add_text(
                    "body",
                    "Python is a versatile language widely used in data science and machine learning.",
                )
                .build(),
        )
        .await?;
    engine.commit().await?;

    // 5. 検索
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_search_request(LexicalSearchRequest::new(Box::new(TermQuery::new(
                    "body", "rust",
                ))))
                .limit(5)
                .build(),
        )
        .await?;

    for hit in &results {
        println!("score={:.4}", hit.score);
    }

    Ok(())
}
```

## サンプル

使用例は [`laurus/examples/`](laurus/examples/) ディレクトリにあります:

| サンプル | 説明 | フィーチャーフラグ |
| --- | --- | --- |
| [quickstart](laurus/examples/quickstart.rs) | 基本的な全文検索 | — |
| [lexical_search](laurus/examples/lexical_search.rs) | 全クエリタイプ（Term、Phrase、Boolean、Fuzzy、Wildcard、Range、Geo、Span） | — |
| [vector_search](laurus/examples/vector_search.rs) | エンベディングによるセマンティック類似度検索 | — |
| [hybrid_search](laurus/examples/hybrid_search.rs) | フュージョンによる Lexical 検索とベクトル検索の統合 | — |
| [geo3d_search](laurus/examples/geo3d_search.rs) | 3D ECEF 地理空間検索（球、バウンディングボックス、k-NN） | — |
| [synonym_graph_filter](laurus/examples/synonym_graph_filter.rs) | 解析パイプラインでの同義語展開 | — |
| [search_with_candle](laurus/examples/search_with_candle.rs) | Candle によるローカル BERT エンベディング | `embeddings-candle` |
| [search_with_openai](laurus/examples/search_with_openai.rs) | OpenAI によるクラウドベースエンベディング | `embeddings-openai` |
| [multimodal_search](laurus/examples/multimodal_search.rs) | テキストから画像、画像から画像への検索 | `embeddings-multimodal` |

## コントリビューション

コントリビューションを歓迎します！

1. リポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。
