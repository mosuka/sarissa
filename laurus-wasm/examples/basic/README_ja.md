# 基本的なハイブリッド検索サンプル

laurus-wasm を使用したブラウザ上での日本語全文検索・ベクトル検索・
ハイブリッド検索を実演するシングルページアプリケーションです。
OPFS によるデータ永続化により、ページリロード後もデータが保持されます。

サンプル一覧と共通のビルド手順は [examples README](../README_ja.md)
を参照してください。最短手順は次の通りです。

```bash
cd laurus-wasm
wasm-pack build --target web --dev
./scripts/postbuild.sh

# UniDic zip を examples/dict/lindera-unidic.zip に配置してから、
# 任意の HTTP サーバーで配信します:
python3 -m http.server 8080

# ブラウザで http://localhost:8080/examples/basic/ を開きます。
```

## このサンプルでできること

- OPFS 永続化ストレージを使用した検索インデックスを作成。ロード時に
  `version()` のスタンプを照合し、別ビルドの laurus-wasm が書いた
  インデックス（オンディスクフォーマットが変わっている可能性がある）
  は自動的に破棄して再構築します
  （ページリロード後もデータが保持されます）
- 初回アクセス時に 8 件のサンプルドキュメントを投入。
  既存データが OPFS にある場合はロードのみ
- Transformers.js（`paraphrase-multilingual-MiniLM-L12-v2`）による
  384 次元セマンティック Embedding をコールバック Embedder 経由で
  自動生成
- 統合クエリ DSL 対応の検索ボックス:
  - Lexical 検索: `rust`、`title:wasm`、`"memory safety"`
  - Vector 検索: `embedding:"how to make code faster"`、
    `embedding:python`
  - Hybrid 検索: `rust embedding:"systems programming"`
- 新しいドキュメントをインタラクティブに追加可能
- 関連度スコア付きの検索結果を表示
- すべての操作をコンソールパネルにログ出力

## ファイル構成

このサンプルは `examples/shared/` を経由して、辞書ローダー・
Embedder・ログヘルパ・テーマスタイルを他のサンプルと共有しています。
UniDic zip（約 52 MB）は `examples/dict/lindera-unidic.zip`（この
サンプルから見て一階層上）から取得されるため、複数のサンプルで
同じキャッシュ辞書を共有できます。
