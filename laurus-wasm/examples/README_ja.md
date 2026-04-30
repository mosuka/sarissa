# Laurus WASM — サンプル集

`examples/` 直下の各サブディレクトリは、`../pkg/` から
laurus-wasm を直接読み込み、ブラウザだけで動作する独立した
シングルページアプリです。サンプル一覧トップとしては
[`examples/index.html`](./index.html) をローカル HTTP サーバーで
開いてください。個別のサンプルにも直接アクセスできます。

## サンプル一覧

| サンプル | 内容 |
| --- | --- |
| [`basic/`](./basic/) | 日本語の全文検索・ベクトル検索・ハイブリッド検索を統合クエリ DSL で実演します。ドキュメントの追加もインタラクティブに行えます。 |
| [`geo/`](./geo/) | 東京の観光スポットを Leaflet 地図上にプロットし、ビューポートから `location:geo_bbox(...)` を生成してテキスト検索・ベクトル検索と組み合わせるサンプルです。 |

[`shared/`](./shared/) には、全サンプルで再利用するアセット（テーマ
スタイル、ロガー、辞書ローダー、Embedder ヘルパ）を配置しています。

## 実行方法

```bash
cd laurus-wasm
wasm-pack build --target web --dev
./scripts/postbuild.sh
```

各サンプルは `examples/dict/lindera-unidic.zip`（約 52 MB）から
UniDic を読み込みます。デプロイワークフローでは自動取得されます
が、ローカル開発では [Lindera のリリース][lindera-releases] から
バージョンの合う zip をダウンロードして `examples/dict/` に置いて
ください。

```bash
# リポジトリのルートから実行
mkdir -p laurus-wasm/examples/dict
curl -fsSL -o laurus-wasm/examples/dict/lindera-unidic.zip \
  "https://github.com/lindera/lindera/releases/download/v<version>/lindera-unidic-<version>.zip"
```

その後、任意の HTTP サーバーを起動します（WASM は `file://` では
動作しません）。

```bash
# Python
python3 -m http.server 8080
# または Node.js
npx serve .
```

ブラウザで <http://localhost:8080/examples/> を開きます。

[lindera-releases]: https://github.com/lindera/lindera/releases

## 新しいサンプルの追加方法

1. `examples/<名前>/index.html` を作成。`../../pkg/laurus_wasm.js`
   から laurus-wasm を、`../shared/` からヘルパを import します。
2. `examples/<名前>/README.md` と `README_ja.md` を追加します。
3. この README とランディング `examples/index.html` から新しい
   サンプルへのリンクを追加します。
4. デプロイワークフロー
   ([`.github/workflows/deploy-docs.yml`][deploy]) が `examples/`
   全体を公開先にコピーするため、サンプル追加時の CI 変更は
   不要です。

[deploy]: ../../.github/workflows/deploy-docs.yml
