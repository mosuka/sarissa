# 位置情報検索サンプル

laurus-wasm 上で全文検索・ベクトル検索・地理空間検索を組み合わせる
シングルページアプリです。東京の観光スポットを
[Leaflet](https://leafletjs.com/) の地図にプロットし、テキスト・
Embedding・地図ビューポートから生成した bbox（境界矩形）で絞り
込めます。

サンプル一覧と共通のビルド手順は
[examples README](../README_ja.md) を参照してください。最短手順は
次の通りです。

```bash
cd laurus-wasm
wasm-pack build --target web --dev
./scripts/postbuild.sh

# UniDic zip を examples/dict/lindera-unidic.zip に配置してから、
# 任意の HTTP サーバーで配信します:
python3 -m http.server 8080

# ブラウザで http://localhost:8080/examples/geo/ を開きます。
```

## このサンプルでできること

- OPFS 永続化された geo インデックス（`geo-demo-index`）を作成し、
  初回アクセス時に東京の観光スポット約 14 件を投入します
- スキーマは日本語形態素解析対応のテキストフィールド 3 種
  （`title`、`description`、`category`）、geo フィールド 1 種
  （`location` — BKD ツリーでインデックス化）、HNSW ベクトル
  フィールド 1 種（`embedding`、multilingual MiniLM 384 次元）
- 検索ボックスは統合クエリ DSL の文字列を生成します。
  *Filter by current map view* が ON のときは
  `+(<クエリ>) +location:geo_bbox(min_lat, min_lon, max_lat, max_lon)`
  という文字列を投げます。両方の clause に `+`（必須）を付けて
  いるため、テキスト条件と現在の Leaflet ビューポート条件の
  どちらも満たすドキュメントだけがマッチします
- 地図のピンは検索結果と連動し、マッチしなかったポイントは
  検索のたびに地図から取り除かれるため、表示されているピンは
  常に結果リストと一致します
- 地図をパン・ズームすると検索が自動で再実行されます
  （*Filter by current map view* が ON のときのみ）。bbox 条件が
  現在のビューに追従します
- 結果リストの項目をクリックすると、地図上の該当マーカーが
  展開され、ビューがそのマーカーへパンします
- Debug カードに、地図の中心・ズームレベル・`geo_bbox(...)` に
  渡される（必要に応じてクランプされた）bbox 値・最後に
  `index.search()` に投げた DSL 文字列をライブ表示します。
  CLI やユニットテストへ DSL をコピーしたいときに便利です

### クエリ例

| トグル | クエリ | 動作 |
| --- | --- | --- |
| ON | （空欄） | 現在のビューポート内の全ポイントを返します。 |
| ON | `公園` | `公園` の語彙マッチ AND ビューポート内 |
| ON | `embedding:"夜景がきれい"` | 意味マッチ AND ビューポート内 |
| OFF | `title:浅草寺` | データセット全体に対する純粋な語彙マッチ |
| OFF | `embedding:"街並みが歴史的"` | データセット全体に対する純粋な意味マッチ |

## ファイル構成

このサンプルは `examples/shared/` を経由して、辞書ローダー・
Embedder・ログヘルパ・テーマスタイルを他のサンプルと共有します。
Leaflet 本体は SRI ハッシュ付きで [`unpkg.com`][unpkg] から取得
しています。オフラインビルドを行う場合はアセットをローカルに
配置してください。

OPFS の辞書キー（`unidic`）は basic サンプルと共有しているため、
約 52 MB の UniDic zip はサンプルをまたいでも 1 回しかダウンロード
されません。

[unpkg]: https://unpkg.com/leaflet@1.9.4/

## 注意事項

- 地図タイルは OpenStreetMap から取得します。laurus 本体はローカル
  で動作しますが、地図描画にはネットワーク接続が必要です。
- 地図を縮小していくとビューが世界全体を超えたり、日付変更線を
  またいだりして、Leaflet が `[-90, 90]` × `[-180, 180]` の範囲
  外の bounds を返すことがあります。そのまま `geo_bbox(...)` に
  渡すとバリデーションに失敗し、検索結果が 0 件になってしまうため、
  本デモでは bbox を有効範囲にクランプします（Debug カードに
  クランプが発生したことを表示します）。日付変更線をまたぐ場合の
  bbox 分割クエリは本サンプルのスコープ外です。
