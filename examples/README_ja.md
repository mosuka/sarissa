# Examples

このディレクトリには、laurus を使ったデータセットのインデックスと検索のサンプルスクリプトが含まれています。

## サンプル一覧

| サンプル | データセット | 内容 |
| --- | --- | --- |
| [movies](movies/) | Meilisearch movies（約 32,000 件） | 英語のレキシカル全文検索 + CLIP マルチモーダルベクトル検索 |
| [aozora](aozora/) | 青空文庫（著作権フリー作品） | Lindera + IPADIC による日本語形態素解析全文検索、フィールド別アナライザー |

## 共通の前提条件

- Rust ツールチェーン（リポジトリルートで `cargo build` が通ること）

各サンプル固有の前提条件（データセットの取得方法、追加ツール、build フィーチャー等）は、それぞれのセクションを参照してください。

## Movies

Meilisearch の movies データセットから約 32,000 件の映画をインデックスして検索します。

### 前提条件（Movies 固有）

- [jq](https://jqlang.org/) — JSON データセットの解析に使用します
- [curl](https://curl.se/) — ポスター画像のダウンロードに使用します
- [python3](https://www.python.org/) — バイナリから JSON 配列への変換に使用します
- ビルド時に `embeddings-multimodal` フィーチャーを有効にすること
- データセット: [meilisearch/datasets](https://github.com/meilisearch/datasets) を laurus プロジェクトの隣にクローンしてください

  ```bash
  cd ..
  git clone https://github.com/meilisearch/datasets.git
  ```

  期待されるディレクトリ構成:

  ```text
  parent/
  ├── datasets/       # meilisearch/datasets のクローン
  │   └── datasets/
  │       └── movies/
  │           └── movies.json
  └── laurus/         # このプロジェクト
      └── examples/
  ```

### 実行

```bash
# 1. インデックスを作成
bash examples/movies/scripts/create_index.sh

# 2. 全映画をインデックス
bash examples/movies/scripts/index_movies.sh

# 3. 検索例を実行
bash examples/movies/scripts/search_movies.sh
```

スキーマ定義は [examples/movies/schema.toml](movies/schema.toml) を参照してください。

## Aozora Bunko（青空文庫）

[青空文庫](https://www.aozora.gr.jp/) の著作権フリー作品をインデックスして日本語全文検索を行います。
Lindera + IPADIC による形態素解析、フィールドごとのアナライザー割り当て、フレーズ検索と OR 緩和検索の使い分け、そして Candle BERT テキスト embedder による意味的ベクトル検索・ハイブリッド検索を示します。

### 前提条件（Aozora 固有）

- [python3](https://www.python.org/) — 作品リスト CSV の解析、本文の CP932 デコード、ルビ除去に使用します
- [curl](https://curl.se/)、[unzip](https://linux.die.net/man/1/unzip) — 辞書と本文アーカイブの展開に使用します
- ネットワーク接続（初回実行時に Lindera IPADIC 辞書 約 15MB、テキスト埋め込みモデル 約 470MB、作品本文をダウンロードします。以降はキャッシュされます）
- `embeddings-candle` build フィーチャーの有効化が必要（このサンプルの各スクリプトは既に指定済みです）。データセットのクローンは不要です

### 実行

```bash
# 1. インデックスを作成（IPADIC 辞書の取得を含む）
bash examples/aozora/scripts/create_index.sh

# 2. 作品を投入（既定 1,000 件）
bash examples/aozora/scripts/index_aozora.sh

# 3. 検索例を実行
bash examples/aozora/scripts/search_aozora.sh
```

> 青空文庫はボランティアによって運営されています。
> `--limit 0`（全件）を指定すると約 17,000 件をダウンロードするため、
> 特別な理由が無い限り既定値のままお使いください。

スキーマ定義は [examples/aozora/schema.toml](aozora/schema.toml) を参照してください。
