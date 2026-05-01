# 3D 位置情報検索サンプル（航空機）

laurus の 3D 地理空間検索（`Geo3d` フィールド、ECEF 直交座標）を
[CesiumJS](https://cesium.com/platform/cesiumjs/) の 3D 地球儀上で
体験できるシングルページアプリです。航空機の位置データはコミュニ
ティ ADS-B フィードである [airplanes.live](https://airplanes.live/)
からページロード時および Refresh ボタン押下時にライブ取得します。

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

# ブラウザで http://localhost:8080/examples/geo3d/ を開きます。
```

## このサンプルでできること

- ページロードのたびに OPFS をクリアして作り直す揮発インデックス
  （`flights3d-demo-index`）。前日の位置情報を保持しても無価値な
  Live データに合わせた運用
- 日本語形態素解析対応のテキストフィールド 5 種
  （`callsign`、`registration`、`aircraft_type`、`description`、
  `category`）、boolean フィールド `on_ground`、float フィールド
  `altitude_m`、そして本サンプルの主役である `position` フィールド
  （`geo3d` 型、3D BKD ツリーでインデックス化）を組み合わせた
  スキーマ
- WGS84 → ECEF 変換ヘルパーを JS 内に実装。`laurus/src/util/ecef.rs`
  と同じ式を使うため、`{ x, y, z }` オブジェクトを
  `index.putDocument` にそのまま渡せます（WASM の変換層が
  `DataValue::GeoEcef` として扱います）
- 検索ボックスは統合クエリ DSL の文字列を生成します。自由テキスト
  と組み合わせられる相互排他の 3D 制約を 2 種類用意しています:
  - **Filter by 3D bbox around camera** — カメラ ECEF 位置を中心と
    する 200km の AABB から
    `+(<text>) +position:geo3d_bbox(minX, minY, minZ, maxX, maxY, maxZ)`
    を生成
  - **Nearest 20 aircraft to camera target** — 画面中央と WGS84
    楕円体の交点（地表交点が無い場合はカメラ位置）を中心に
    `position:geo3d_nearest(targetX, targetY, targetZ, 20)` を発行
- Cesium Ion トークン**未使用**で動作する CesiumJS ビューア。
  地図画像は OpenStreetMap、地形プロバイダは ellipsoid のみで、
  Ion 経由のリクエストを一切しません
- Cesium のエンティティは検索結果と連動し、マッチしなかった機体は
  検索のたびに地球上から取り除かれるため、表示されているマーカーは
  常に結果リストと一致します
- 3D 制約が ON のときは、地球をドラッグ・ズーム・回転すると検索が
  自動で再実行されます。空間条件が常にカメラの状態に追従します
- 結果リストの項目をクリックすると、カメラがその機体に flyTo します
- Debug カードに、カメラの測地座標（緯度/経度/高度）、カメラ ECEF、
  カメラのターゲット ECEF、現在の 3D bbox、最後に
  `index.search()` に投げた DSL 文字列をライブ表示します。CLI や
  ユニットテストへ DSL をコピーしたいときに便利です

### クエリ例

| フィルタ | クエリ | 動作 |
| --- | --- | --- |
| OFF | （空欄） | クエリ無し（検索しない） |
| OFF | `callsign:JAL*` | スナップショット全体に対する callsign 前方一致 |
| OFF | `description:Boeing` | 機種記述の語彙マッチ |
| OFF | `category:heavy` | スナップショット内のワイドボディ機すべて |
| BBOX | （空欄） | カメラ周囲 200km 立方内の全機 |
| BBOX | `aircraft_type:B38M` | 3D bbox 内の 737 MAX 8 |
| NEAREST | （空欄） | カメラターゲットに 3D で最も近い 20 機 |
| NEAREST | `category:heavy` | カメラターゲットに最も近い heavy 20 機 |

## ファイル構成

このサンプルは `examples/shared/` 経由で、辞書ローダー・ログ
ヘルパ・テーマスタイルを他のサンプルと共有します。CesiumJS 本体は
SRI ハッシュ付きで [unpkg.com][unpkg-cesium] から取得しています。
オフラインビルドを行う場合はアセットをローカルに配置してください。

OPFS の辞書キー（`unidic`）は他サンプルと共有しているため、
約 52 MB の UniDic zip はサンプルをまたいでも 1 回しかダウンロード
されません。

[unpkg-cesium]: https://unpkg.com/cesium@1.121.0/Build/Cesium/

## 注意事項

- 地図画像は OpenStreetMap から取得します。laurus 本体はローカル
  で動作しますが、地球儀の描画にはネットワーク接続が必要です。
- 航空機データは `https://api.airplanes.live/v2/...` から取得します。
  CORS 対応済（`Access-Control-Allow-Origin: *`）でブラウザから直接
  取得できますが、コミュニティ運営のベストエフォート提供のため、
  瞬間的に 0 件や HTTP エラーが返ることがあります。Refresh ボタンは
  5 秒に 1 回までのレート制限を設けて負荷をかけすぎないようにして
  います。
- 3D bbox はカメラ ECEF 位置を中心とした 200km 立方の粗い AABB で
  あり、レンダリングシーンの厳密な視錐台ではありません。本格的な
  カメラ視錐台 3D bbox は本サンプルのスコープ外です。
- デフォルトの取得は日本中心（`lat=36, lon=138, dist=250nm`）です。
  この半径外の機体はスナップショットに含まれません。グローバル
  カバレッジが必要な場合は URL を変更して Refresh してください。
- インデックスは意図的に非永続です。ページロードのたびに OPFS を
  クリアし、airplanes.live の最新スナップショットから作り直します。
  OPFS 永続化を確認したい場合は basic / geo サンプルを参照して
  ください。
