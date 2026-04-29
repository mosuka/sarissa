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

```bash
# テスト依存関係をインストール
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

## Workspace 統合と clang-sys パッチ

`laurus-php` は [ext-php-rs](https://github.com/extphprs/ext-php-rs) を使用しており、
ext-php-rs は `ext-php-rs-clang-sys`（`clang-sys` のフォーク）に依存しています。
一方、`laurus-ruby` は `magnus` → `rb-sys` → `bindgen` → `clang-sys`（オリジナル）に依存しています。
両方のクレートが `links = "clang"` を宣言しており、Cargo は同一 workspace 内で同じ `links` 値を持つ
パッケージを 2 つ許可しません。

`laurus-php` と `laurus-ruby` を workspace メンバーとして共存させるため、ルートの `Cargo.toml` で
`ext-php-rs-clang-sys` を `links` 宣言を除去したローカルコピーに patch しています：

```toml
# Cargo.toml（workspace ルート）
[patch.crates-io]
ext-php-rs-clang-sys = { path = "patches/ext-php-rs-clang-sys" }
```

パッチは `patches/ext-php-rs-clang-sys/` にあります。上流クレートからの唯一の変更点は
`Cargo.toml` の `links = "clang"` の除去です。`clang-sys` と `ext-php-rs-clang-sys` は
どちらも `libclang` をビルド時のみ使用し（`bindgen` によるヘッダー解析）、最終バイナリにはリンク
されないため、この変更は安全です。

### パッチが必要な条件

このパッチは `laurus-php` と `laurus-ruby` が同一の Cargo workspace のメンバーである場合にのみ
必要です。`laurus-ruby` を workspace から除外するか、`laurus-php` を `[workspace] exclude` で
除外すれば、`links = "clang"` の競合は発生しないため、パッチとルート `Cargo.toml` の
`[patch.crates-io]` セクションを削除できます。

### パッチの更新

`ext-php-rs` をアップグレードして新しいバージョンの `ext-php-rs-clang-sys` が
使われるようになった場合、パッチを更新してください：

```bash
# 1. laurus-php/Cargo.toml で ext-php-rs を更新した後：
cargo update -p ext-php-rs

# 2. 新しい ext-php-rs-clang-sys ソースをコピー
cp -r ~/.cargo/registry/src/index.crates.io-*/ext-php-rs-clang-sys-<NEW_VERSION>/* \
      patches/ext-php-rs-clang-sys/

# 3. links 宣言を除去
sed -i 's/^links = "clang"/# links = "clang"/' patches/ext-php-rs-clang-sys/Cargo.toml

# 4. ビルドを確認
cargo build -p laurus-php -p laurus-ruby
```

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
