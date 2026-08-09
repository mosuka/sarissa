# 開発環境のセットアップ

このページでは `laurus-php` バインディングのローカル開発環境の構築、ビルド、テストスイートの実行方法について説明します。

## 前提条件

- **Rust** 1.85 以降（Cargo 含む）
- **PHP** 8.1 以降（開発ヘッダー付き: `php-dev` / `php-devel`）
- **Composer**（依存関係管理用）
- リポジトリがローカルにクローンされていること

```bash
git clone https://github.com/mosuka/laurus.git
cd laurus
```

## ビルド

### 開発ビルド

Rust ネイティブ拡張をデバッグモードでコンパイルします。Rust ソースを変更した場合は再実行してください。

```bash
cd laurus-php
cargo build
```

ビルド成果物は `../target/debug/liblaurus_php.so` に生成されます。

### リリースビルド

```bash
cd laurus-php
cargo build --release
```

ビルド成果物は `../target/release/liblaurus_php.so` に生成されます。

### ビルドの確認

```bash
php -d extension=../target/release/liblaurus_php.so -r "
use Laurus\Index;
\$index = new Index();
print_r(\$index->stats());
"
# Array ( [documentCount] => 0 [vectorFields] => Array ( ) )
```

## テスト

テストは [PHPUnit](https://phpunit.de/) を使用しており、`tests/` ディレクトリにあります。
Composer は開発時の PHP 依存（PHPUnit）の取得のみに使用し、
ランタイム拡張本体は Cargo で直接ビルド・ロードします。

```bash
# テスト依存関係をインストール（PHPUnit のみ）
composer install

# 全テスト実行
php -d extension=../target/release/liblaurus_php.so vendor/bin/phpunit tests/
```

特定のテストファイルを実行する場合：

```bash
php -d extension=../target/release/liblaurus_php.so vendor/bin/phpunit tests/LaurusTest.php
```

## Lint とフォーマット

```bash
# Rust lint（Clippy）
cargo clippy -p laurus-php -- -D warnings

# Rust フォーマットチェック
cargo fmt -p laurus-php --check

# フォーマット適用
cargo fmt -p laurus-php
```

## クリーンアップ

```bash
# ビルド成果物を削除
cargo clean

# Composer 依存関係を削除
rm -rf vendor/
```

## Workspace 統合と clang-sys

`laurus-php` は [ext-php-rs](https://github.com/extphprs/ext-php-rs) を使用しており、
そのバインディング生成（`ext-php-rs-bindgen`）は `bindgen` で PHP ヘッダーを解析し、
`clang-sys` 経由で `libclang` をロードします。一方、`laurus-ruby` は
`magnus` → `rb-sys` のビルドで同じく `bindgen` + `clang-sys` を使用します。
Cargo は同一 workspace 内で同じ `links` 値を持つパッケージを 2 つ許可しないため、
PHP と Ruby のバインディングが workspace メンバーとして共存できるのは、両者が
**同一の** `clang-sys` パッケージに解決される場合のみです。

かつての `ext-php-rs` は `ext-php-rs-clang-sys`（`links = "clang"` を宣言する
`clang-sys` のフォーク）に依存しており、`laurus-ruby` 側のオリジナル `clang-sys` と
衝突していました。当時は `links` 宣言を除去したローカルコピー（`patches/` 配下）への
`[patch.crates-io]` オーバーライドが必要でした。`ext-php-rs-bindgen
0.72.1-extphprs.2`（`ext-php-rs` 0.15.15 が使用）以降はフォークが廃止され、
`ext-php-rs` は通常の `clang-sys` に戻ったため、両バインディングは 1 つの
`clang-sys` パッケージを共有し、パッチは削除済みです。

将来の `ext-php-rs` アップグレードでフォーク版 clang-sys が再導入された場合
（`cargo build -p laurus-php -p laurus-ruby` で `links = "clang"` の衝突エラーが
出たら要注意）、vendored パッチを復活させてください: フォーククレートのソースを
`patches/` にコピーし、その `links = "clang"` 行をコメントアウトし、ルート
`Cargo.toml` の `[patch.crates-io]` からそこを指します。`clang-sys` は
`libclang` をビルド時のみ使用し（`bindgen` によるヘッダー解析）、最終バイナリには
リンクされないため、この変更は安全です。

## macOS リンカーフラグ (`-undefined dynamic_lookup`)

PHP 拡張は共有ライブラリ（`.so` / `.dylib`）であり、実行時に PHP インタプリタに
ロードされます。PHP API シンボル（`zend_*`, `php_*` 等）は PHP バイナリ本体に
定義されており、拡張がリンクするライブラリには含まれません。Linux ではリンカーが
共有ライブラリ内の未定義シンボルをデフォルトで許容するため問題ありませんが、
macOS ではリンカーが未定義シンボルをエラーとして扱い、ビルドが失敗します:

```text
ld: symbol(s) not found for architecture arm64
```

修正方法は `-Wl,-undefined,dynamic_lookup` をリンカーに渡すことです。これにより
シンボル解決がロード時（PHP が拡張を `dlopen` する時点）まで延期されます。

このフラグは `.cargo/config.toml` には設定**しません**。設定すると workspace 内の
全クレートに適用され、PHP 以外のクレートでも未定義シンボルがエラーにならなくなる
ためです。代わりに `laurus-php` のビルド時のみ適用します:

**Makefile**（ローカル開発）:

```makefile
build-laurus-php:
ifeq ($(shell uname -s),Darwin)
    RUSTFLAGS="-C link-args=-Wl,-undefined,dynamic_lookup" cargo build -p laurus-php --release
else
    cargo build -p laurus-php --release
endif
```

**CI**（GitHub Actions）:

```yaml
- name: Build PHP extension
  shell: bash
  run: |
    if [ "$RUNNER_OS" == "macOS" ]; then
      export RUSTFLAGS="-C link-args=-Wl,-undefined,dynamic_lookup"
    fi
    cargo build --release -p laurus-php
```

macOS でビルドする際は、`cargo build -p laurus-php` を直接実行するのではなく、
`make build-laurus-php` または `make test-laurus-php` を使用してください。

## プロジェクト構成

```text
laurus-php/
├── Cargo.toml          # Rust クレートマニフェスト
├── composer.json       # Composer パッケージ定義
├── composer.lock       # ロックされた依存関係バージョン
├── src/                # Rust ソース（ext-php-rs バインディング）
│   ├── lib.rs          # モジュール登録
│   ├── index.rs        # Index クラス
│   ├── schema.rs       # Schema クラス
│   ├── query.rs        # クエリクラス
│   ├── search.rs       # SearchRequest / SearchResult / Fusion
│   ├── analysis.rs     # Tokenizer / Filter / Token
│   ├── convert.rs      # PHP <-> DataValue 変換
│   └── errors.rs       # エラーマッピング
├── tests/              # PHPUnit テスト
│   └── LaurusTest.php
└── examples/           # 実行可能な PHP サンプル
```
