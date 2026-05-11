# プロジェクト構成

Laurus は 9 つのクレートで構成された Cargo ワークスペースです。コアライブラリ、3 つの自製バイナリ（CLI、gRPC サーバー、MCP サーバー）、および 5 つの言語バインディングから成ります。

## ワークスペースレイアウト

```text
laurus/                          # リポジトリルート
├── Cargo.toml                   # ワークスペース定義（members + workspace.package）
├── laurus/                      # コア検索エンジンライブラリ
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs               # パブリック API とモジュール宣言
│   │   ├── engine.rs            # Engine, EngineBuilder, SearchRequest
│   │   ├── analysis/            # テキスト解析パイプライン
│   │   ├── lexical/             # 転置インデックス（Inverted Index）と Lexical 検索
│   │   ├── vector/              # ベクトルインデックス（Flat, HNSW, IVF）
│   │   ├── embedding/           # Embedder 実装
│   │   ├── storage/             # ストレージバックエンド（memory, file, mmap）
│   │   ├── store/               # ドキュメントログ（WAL）
│   │   ├── spelling/            # スペル修正
│   │   ├── data/                # DataValue, Document 型
│   │   └── error.rs             # LaurusError 型
│   └── examples/                # 実行可能なサンプル
├── laurus-cli/                  # コマンドラインインターフェース
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs              # CLI エントリーポイント（clap）
│       ├── cli.rs               # サブコマンド定義
│       └── commands/            # サブコマンドごとの実装
├── laurus-server/               # gRPC サーバー + HTTP ゲートウェイ
│   ├── Cargo.toml
│   ├── proto/laurus/v1/         # Protobuf サービス定義
│   └── src/
│       ├── lib.rs               # サーバーライブラリ
│       ├── config.rs            # TOML 設定
│       ├── service/             # gRPC サービス実装（tonic）
│       └── gateway/             # HTTP/JSON ゲートウェイ（axum）
├── laurus-mcp/                  # MCP（Model Context Protocol）stdio サーバー
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── server.rs            # rmcp ツールルーター（12 ツール）
│       └── convert.rs           # JSON ↔ DataValue 変換ヘルパー
├── laurus-python/               # Python バインディング（PyO3 + Maturin）
├── laurus-nodejs/               # Node.js バインディング（NAPI-RS）
├── laurus-wasm/                 # WebAssembly バインディング（wasm-bindgen）
├── laurus-ruby/                 # Ruby バインディング（magnus + rb-sys）
├── laurus-php/                  # PHP バインディング（ext-php-rs）
└── docs/                        # mdBook ドキュメント
    ├── book.toml                # 英語版 mdBook 設定
    ├── src/                     # 英語版ソース
    │   └── SUMMARY.md           # 英語版目次
    └── ja/                      # 日本語版 mdBook（独立ビルド）
        ├── book.toml
        └── src/
            └── SUMMARY.md
```

## クレートの役割

| クレート | 種類 | 説明 |
| :--- | :--- | :--- |
| `laurus` | ライブラリ | Lexical 検索、ベクトル検索、ハイブリッド検索を備えたコア検索エンジン |
| `laurus-cli` | バイナリ | インデックス管理、ドキュメント CRUD、検索、REPL、および `serve` / `mcp` ランチャーを提供する CLI ツール |
| `laurus-server` | ライブラリ + バイナリ | オプションの HTTP/JSON ゲートウェイ付き gRPC サーバー |
| `laurus-mcp` | バイナリ | 稼働中の laurus-server へツール呼び出しをプロキシする MCP stdio サーバー |
| `laurus-python` | 動的ライブラリ | PyO3 / Maturin で構築する Python パッケージ（PyPI） |
| `laurus-nodejs` | 動的ライブラリ | NAPI-RS で構築する Node.js パッケージ（npm） |
| `laurus-wasm` | WebAssembly | wasm-bindgen で構築するブラウザ・エッジランタイム向け npm パッケージ |
| `laurus-ruby` | 動的ライブラリ | magnus と rb-sys で構築する Ruby gem |
| `laurus-php` | PHP 拡張 | ext-php-rs で構築する PHP 拡張（スタンドアロン構成。ワークスペースからは除外。詳細は [ビルドとテスト](build_and_test.md) を参照） |

すべてのバインディングクレートと 3 つの自製バイナリは `laurus` に依存します。

## 設計規約

- **モジュールスタイル**: ファイルベースのモジュール（Rust 2018 edition スタイル）、`mod.rs` は使用しない
  - `src/tokenizer.rs` + `src/tokenizer/dictionary.rs`
  - 不可: `src/tokenizer/mod.rs`
- **エラーハンドリング**: ライブラリのエラー型には `thiserror`、`anyhow` はバイナリクレートのみ
- **`unwrap()` / `expect()` 禁止**: 本番コードでは使用不可（テストでは使用可）
- **非同期**: すべてのパブリック API は Tokio ランタイムで async/await を使用
- **Unsafe**: すべての `unsafe` ブロックに `// SAFETY: ...` コメントが必須
- **ドキュメント**: すべてのパブリックな型、関数、列挙型にドキュメントコメント（`///`）が必須
- **ライセンス**: 依存クレートは MIT または Apache-2.0 互換であること
