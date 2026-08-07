# Movies サンプル

[Meilisearch movies データセット](https://github.com/meilisearch/datasets) の約 32,000 件の映画データをインデックスして検索するサンプルです。
レキシカル全文検索に加え、ポスター画像に対するマルチモーダル（CLIP）ベクトル検索にも対応しています。

## 前提条件

- [jq](https://jqlang.github.io/jq/) — JSON 処理
- [curl](https://curl.se/) — ポスター画像のダウンロード
- [python3](https://www.python.org/) — バイナリから JSON 配列への変換
- ビルド時に `embeddings-multimodal` フィーチャーを有効にすること

## スキーマ

[schema.toml](schema.toml) で以下のフィールドを定義しています:

| フィールド | 型 | インデックス | 保存 | 説明 |
| --------- | ---- | ---------- | ---- | ---- |
| `title` | Text | あり | あり | 映画タイトル |
| `overview` | Text | あり | あり | あらすじ |
| `genres` | Text | あり | あり | カンマ区切りのジャンル一覧 |
| `poster` | Text | なし | あり | ポスター画像の URL |
| `release_date` | Integer | あり | あり | Unix タイムスタンプ |
| `poster_vec` | Hnsw | あり | なし | ポスター画像の CLIP 埋め込み（512 次元） |

デフォルト検索フィールド: `title`, `overview`

### Embedder

スキーマでは [CLIP](https://openai.com/index/clip/)（`openai/clip-vit-base-patch32`）を使用する `clip_embedder` を定義しています。
`poster_vec` フィールドがこの Embedder を参照しているため、インデックス時にポスター画像が自動的に 512 次元のベクトル空間に埋め込まれます。

## ドキュメント例

`index_movies.sh` は中間ファイルを書き出しません。データセットの各行を `jq` で document JSON に変換し、
`add doc <id> <document>` コマンドとして単一の `laurus repl` プロセスへ直接パイプで投入します
（[scripts/index_movies.sh](scripts/index_movies.sh) 参照）。
[『マトリックス』](https://www.themoviedb.org/movie/603) の場合、生成されるドキュメントは次のようになります:

```json
{
  "fields": {
    "title": {"Text": "The Matrix"},
    "overview": {"Text": "Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a group of underground insurgents fighting the vast and powerful computers who now rule the earth."},
    "genres": {"Text": "Action, Science Fiction"},
    "poster": {"Text": "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg"},
    "release_date": {"Int64": 922752000},
    "poster_vec": {"Bytes": [[255, 216, 255, 224, "…"], "image/jpeg"]}
  }
}
```

`poster_vec.Bytes` は `[<生JPEGバイト配列>, <MIMEタイプ>]` という形です。上記のバイト配列は
読みやすさのため JPEG のマジックナンバー（`FF D8 FF E0`）の後で省略していますが、実際の配列には
ダウンロードしたファイルの全バイトが入ります。`poster_vec` は `examples/movies/images/<id>.jpg`
へのポスター画像ダウンロードが成功した場合のみ追加されます。ポスターが無い映画はこのフィールド無しで
インデックスされます。

## 使い方

### 1. インデックスの作成

```bash
bash examples/movies/scripts/create_index.sh
```

release バイナリをビルドし、スキーマを使って `examples/movies/index/` に空のインデックスを作成します。

### 2. 映画データの投入

```bash
bash examples/movies/scripts/index_movies.sh
```

一部だけインデックスする場合（例: 最初の 100 件でクイックテスト）:

```bash
bash examples/movies/scripts/index_movies.sh --limit 100
```

このスクリプトは以下を行います:

1. `embeddings-multimodal` フィーチャー付きで release バイナリをビルド
2. TMDB からポスター画像を `examples/movies/images/` にダウンロード（並列、冪等）
3. 各映画をレキシカルフィールドとポスターバイトを持つ laurus ドキュメントに変換
4. 全ドキュメントを REPL にパイプで投入し、1,000 件ごとにコミット
5. エンジンがポスターバイトを自動的に 512 次元の CLIP ベクトルに埋め込み

### 3. 検索例の実行

```bash
bash examples/movies/scripts/search_movies.sh
```

以下の検索例を実行します:

**レキシカル検索:**

- `star wars` — デフォルトフィールドに対する全文検索
- `title:nemo` — フィールド指定検索
- `genres:comedy` — ジャンルで検索
- `overview:robot` — あらすじ内を検索
- JSON 形式での出力

**マルチモーダル（ベクトル）検索:**

- `poster_vec:"space adventure"` — 宇宙冒険風のポスターの映画を検索
- `poster_vec:"romantic couple"` — ロマンチックなポスターの映画を検索
- `poster_vec:"scary monster horror"` — ホラー風のポスターの映画を検索

### 手動検索

直接コマンドで検索することもできます:

```bash
# レキシカル検索
./target/release/laurus --index-dir examples/movies/index search "title:matrix" --limit 10

# マルチモーダルベクトル検索（テキスト→画像）
./target/release/laurus --index-dir examples/movies/index search 'poster_vec:"action hero"' --limit 10
```

対話モードで操作する場合:

```bash
./target/release/laurus --index-dir examples/movies/index repl
```

## ファイル構成

```text
examples/movies/
├── README.md
├── README_ja.md
├── schema.toml          # インデックスのスキーマ定義（レキシカル＋ベクトル）
├── scripts/
│   ├── create_index.sh  # インデックスの作成
│   ├── index_movies.sh  # 画像ダウンロードとデータセットの投入
│   └── search_movies.sh # 検索例（レキシカル＋マルチモーダル）
├── images/              # ダウンロードされたポスター画像（git 管理外）
└── index/               # 生成されるインデックスデータ（git 管理外）
```
