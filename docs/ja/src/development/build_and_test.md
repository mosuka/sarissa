# ビルドとテスト

## 前提条件

- **Rust** 1.85 以降（edition 2024）
- **Cargo**（Rust に付属）
- **protobuf コンパイラ**（`protoc`）-- `laurus-server` のビルドに必要

## ビルド

```bash
# すべてのクレートをビルド
cargo build

# 特定の Feature を指定してビルド
cargo build --features embeddings-candle

# リリースモードでビルド
cargo build --release
```

## テスト

```bash
# すべてのワークスペーステストを実行（デフォルト Feature）
cargo test

# 名前を指定して特定のテストを実行
cargo test <test_name>

# 特定のクレートのテストを実行
cargo test -p laurus
cargo test -p laurus-cli
cargo test -p laurus-server
cargo test -p laurus-mcp
```

### 言語バインディングのテスト

各言語バインディングは固有のツールチェーン（Python virtualenv、Node.js
npm、Ruby Bundler、PHP Composer、`wasm32-unknown-unknown` ターゲット）を持ちます。
Makefile はこれらをラップし、各ターゲットがツールチェーンを準備したうえで
テストを実行します。

```bash
make test-laurus-python   # cargo test -p laurus-python + Maturin 経由の pytest
make test-laurus-nodejs   # npm run build:debug + npm test
make test-laurus-wasm     # cargo build -p laurus-wasm --target wasm32-unknown-unknown
make test-laurus-ruby     # cargo test -p laurus-ruby + Ruby minitest
make test-laurus-php      # cargo build -p laurus-php --release + PHPUnit
```

`laurus-php` は `laurus-ruby` との `links = "clang"` 競合のため Cargo ワークスペースから
除外されており、上記の Makefile ターゲット経由でスタンドアロンクレートとしてビルド・テストします。
対応する `format-laurus-*` / `lint-laurus-*` / `build-laurus-*` のバリアントを含む全ターゲットは
[`Makefile`](../../../Makefile) を参照してください。

## Lint

```bash
# clippy を警告エラー扱いで実行
cargo clippy -- -D warnings
```

## フォーマット

```bash
# フォーマットチェック
cargo fmt --check

# フォーマットを適用
cargo fmt
```

## ドキュメント

### API ドキュメント

```bash
# Rust API ドキュメントを生成して開く
cargo doc --no-deps --open
```

### mdBook ドキュメント

```bash
# ドキュメントサイトをビルド
mdbook build docs

# ローカルプレビューサーバーを起動 (http://localhost:3000)
mdbook serve docs

# Markdown ファイルを Lint
markdownlint-cli2 "docs/src/**/*.md"
```
