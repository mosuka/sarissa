# ハンズオンチュートリアル

このチュートリアルでは、laurus CLI を使った一連のワークフローを体験します。スキーマの作成、インデックスの構築、ドキュメントの登録、検索、更新、削除、そしてインタラクティブ REPL の使い方を順を追って説明します。

## 前提条件

- laurus CLI がインストール済み（[インストール](installation.md) を参照）

## Step 1: スキーマの作成

まず、インデックスの構造を定義するスキーマファイルを作成します。対話形式で生成することもできます:

```bash
laurus create schema
```

対話ウィザードがフィールドの定義をガイドします。このチュートリアルでは、手動でスキーマファイルを作成します:

```bash
cat > schema.toml << 'EOF'
default_fields = ["title", "body"]

[fields.title.Text]
indexed = true
stored = true
term_vectors = true

[fields.body.Text]
indexed = true
stored = true
term_vectors = true

[fields.category.Text]
indexed = true
stored = true
term_vectors = false
EOF
```

3 つのテキストフィールドを定義しています。`default_fields` を設定することで、フィールド指定なしのクエリは `title` と `body` の両方を検索します。

## Step 2: インデックスの作成

スキーマを使ってインデックスを作成します:

```bash
laurus --index-dir ./tutorial_data create index --schema schema.toml
```

インデックスが作成されたことを確認します:

```bash
laurus --index-dir ./tutorial_data get stats
```

ドキュメント数が 0 であることが表示されます。

## Step 3: ドキュメントの登録

ドキュメントをインデックスに追加します。各ドキュメントには ID と `{"fields": {...}}` 形式の JSON オブジェクトが必要です:

```bash
laurus --index-dir ./tutorial_data add doc \
  --id doc001 \
  --data '{"fields":{"title":"Introduction to Rust Programming","body":"Rust is a modern systems programming language that focuses on safety, speed, and concurrency.","category":"programming"}}'
```

```bash
laurus --index-dir ./tutorial_data add doc \
  --id doc002 \
  --data '{"fields":{"title":"Web Development with Rust","body":"Building web applications with Rust has become increasingly popular. Frameworks like Actix and Rocket make it easy to create fast and secure web services.","category":"web-development"}}'
```

```bash
laurus --index-dir ./tutorial_data add doc \
  --id doc003 \
  --data '{"fields":{"title":"Python for Data Science","body":"Python is the most popular language for data science and machine learning. Libraries like NumPy and Pandas provide powerful tools for data analysis.","category":"data-science"}}'
```

## Step 4: 変更のコミット

ドキュメントはコミットするまで検索対象になりません:

```bash
laurus --index-dir ./tutorial_data commit
```

## Step 5: ドキュメントの検索

### 基本的な検索

"rust" を含むドキュメントを検索します:

```bash
laurus --index-dir ./tutorial_data search "rust"
```

デフォルトフィールド（`title` と `body`）が検索されます。`doc001` と `doc002` が返されます。

### フィールド指定検索

`title` フィールドのみを検索します:

```bash
laurus --index-dir ./tutorial_data search "title:python"
```

`doc003` のみが返されます。

### カテゴリ検索

```bash
laurus --index-dir ./tutorial_data search "category:programming"
```

`doc001` のみが返されます。

### ブーリアンクエリ

`+`（必須）と `-`（除外）で条件を組み合わせます:

```bash
laurus --index-dir ./tutorial_data search "+body:rust -body:web"
```

"rust" を含み "web" を含まない `doc001` のみが返されます。

### フレーズ検索

完全一致するフレーズを検索します:

```bash
laurus --index-dir ./tutorial_data search 'body:"data science"'
```

`doc003` のみが返されます。

### あいまい検索

`~` を使ってタイプミスに対応した検索を行います:

```bash
laurus --index-dir ./tutorial_data search "body:programing~1"
```

タイプミスがあっても "programming" にマッチします。

### JSON 出力

プログラムでの利用に向けて JSON 形式で結果を取得します:

```bash
laurus --index-dir ./tutorial_data --format json search "rust"
```

## Step 6: ドキュメントの取得

ID を指定して特定のドキュメントを取得します:

```bash
laurus --index-dir ./tutorial_data get docs --id doc001
```

## Step 7: ドキュメントの削除

ドキュメントを削除してコミットします:

```bash
laurus --index-dir ./tutorial_data delete docs --id doc003
laurus --index-dir ./tutorial_data commit
```

削除されたことを確認します:

```bash
laurus --index-dir ./tutorial_data search "python"
```

結果は返されません。

## Step 8: REPL を使う

REPL はインデックスを対話的に操作するためのインタラクティブセッションです:

```bash
laurus --index-dir ./tutorial_data repl
```

REPL で以下のコマンドを試してみてください:

```text
> get stats
> search rust
> add doc doc004 {"fields":{"title":"Go Programming","body":"Go is a statically typed language designed for simplicity and efficiency.","category":"programming"}}
> commit
> search programming
> get docs doc004
> delete docs doc004
> commit
> quit
```

REPL はコマンド履歴（上下キー）や行編集に対応しています。

## Step 9: クリーンアップ

チュートリアルで作成したデータを削除します:

```bash
rm -rf ./tutorial_data schema.toml
```

## 次のステップ

- [スキーマフォーマット](schema_format.md)で高度なフィールド設定を学ぶ
- [コマンド](commands.md)リファレンスで全コマンドを確認する
- [REPL](repl.md)でインタラクティブな操作を深める
- [サーバーチュートリアル](../laurus-server/tutorial.md)で gRPC/HTTP アクセスを試す
