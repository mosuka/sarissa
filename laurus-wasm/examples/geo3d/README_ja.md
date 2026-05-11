# 3D 位置情報検索サンプル（航空機）

laurus の 3D 地理空間検索（`Geo3d` フィールド、ECEF 直交座標）を
[CesiumJS](https://cesium.com/platform/cesiumjs/) の 3D 地球儀上で
体験できるシングルページアプリです。航空機の位置データはコミュニ
ティ ADS-B フィードである [airplanes.live](https://airplanes.live/)
からページロード時および Refresh ボタン（または Auto セレクトの
自動取得）でライブ取得します。

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

## 触り方

1. ページを開くと、東京に黄色のピン📌が立ち、ピン中心に 250 nm の
   範囲（airplanes.live の `/v2/point` エンドポイントが受け付ける
   最大半径）で機体データを取得し、ピンに 3D で最も近い 50 機が
   オレンジマーカー + 高度方向の垂直線で表示されます。
2. **地球をクリック**するとピンが即座にその地点に移動します。
   クリック地点が**最後に fetch した中心から 125 海里より遠い場合のみ**、
   新しいピン中心で 250 nm の範囲を再取得して検索を再実行します。
   それ以内のクリックは前回の snapshot がカバーしているので、既存の
   in-memory インデックスをそのまま使い回します（HTTP 取得なし、
   検索は瞬時）。再取得が発生する場合は手動 Refresh ボタンと同じ
   3 秒のレート制限を共有します。
3. 検索ボックスにテキスト（例: `JAL`、`Boeing`、`category:heavy`）を
   入れる、または Quick filter（`Heavies` / `Helicopters` /
   `JAL flights` / `ANA flights` / `Clear`）でテキストフィルタを
   かけ、**Show:** セレクト（10 / 25 / 50（デフォルト）/ 100 / 200）
   で表示件数を選びます。結果は常にピンとの 3D ユークリッド距離で
   ソートされ、ドロップダウンの件数までの「最も近い N 機」が表示
   されます。
4. 結果リストの行をクリックすると、その機体にカメラが flyTo します。
5. **Refresh data** で手動更新、**Auto** セレクトで 5s / 10s / 30s /
   60s の間隔を選ぶと自動更新を有効にできます（タブが非表示の間は
   一時停止）。手動 / 自動更新は 125 海里の閾値に関係なく必ず
   fetch します。
6. 地球儀の右上にある **↺ Reset view** ボタンで、初期のカメラ
   位置・角度（日本上空のオブリークビュー）に戻れます。

### マウス・タッチ操作

| 操作 | 動作 |
| --- | --- |
| 左ドラッグ | globe を回転（orbit） |
| 右ドラッグ | tilt — カメラの pitch / heading を変更 |
| 中ドラッグ | tilt（補助） |
| ホイール / ピンチ | ズーム |
| 左クリック | クリックした地表点に検索ピン📌を配置 |

## このサンプルでできること

- ページロードのたびに OPFS をクリアして作り直す揮発インデックス
  （`flights3d-demo-index`）。前日の位置情報を保持しても無価値な
  Live データに合わせた運用です。
- 日本語形態素解析対応のテキストフィールド 5 種
  （`callsign`、`registration`、`aircraft_type`、`description`、
  `category`）、boolean フィールド `on_ground`、float フィールド
  `altitude_m`、そして本サンプルの主役である `position` フィールド
  （`geo3d` 型、3D BKD ツリーでインデックス化）を組み合わせた
  スキーマ。
- WGS84 → ECEF 変換ヘルパーを JS 内に実装。`laurus/src/util/ecef.rs`
  と同じ式を使うため、`{ x, y, z }` オブジェクトを
  `index.putDocument` にそのまま渡せます（WASM の変換層が
  `DataValue::GeoEcef` として扱います）。
- ピン位置を中心とした `geo3d_nearest` クエリ。`index.search()` に
  投げている DSL 文字列は、開発者向け詳細カード（折り畳み）から
  そのままコピーできます。
- **増分更新**: Refresh のたびに index 全体を再構築するのではなく、
  古いスナップショットと新しいスナップショットの差分を計算して、
  消えた機体だけ delete、残りは `putDocument` で上書きします。
  ハイライトマーカーも in-place で位置だけ更新するため、
  自動更新中に視覚的なちらつきが発生しません。
- Cesium Ion トークン**未使用**で動作する CesiumJS ビューア。
  地図画像は OpenStreetMap、地形プロバイダは ellipsoid のみで、
  Ion 経由のリクエストを一切しません。

### クエリ例

ピンは常に存在するため、空間制約 ON のときは必ず `geo3d_nearest`
が DSL に乗ります。チェックボックスを OFF にすると空間制約を外して
テキストにマッチした全機を返します。

| 空間制約 | クエリ | 動作 |
| --- | --- | --- |
| ON | （空欄） | ピンに 3D で最も近い 50 機 |
| ON | `callsign:JAL*` | 最寄り 50 機のうち JAL 便のみ |
| ON | `category:heavy` | 最寄り 50 機のうちワイドボディ機のみ |
| OFF | `callsign:JAL*` | スナップショット内の JAL 便全件 |
| OFF | `description:Boeing` | スナップショット内の Boeing 機全件 |
| OFF | `category:rotorcraft` | スナップショット内のヘリコプター全件 |

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
  瞬間的に 0 件や HTTP エラーが返ることがあります。手動の Refresh
  ボタンは 3 秒に 1 回まで、自動更新はデフォルト OFF で、選択肢
  （5 / 10 / 30 / 60 秒）も上流フィードの負荷を抑えるように設計して
  います。
- 各 fetch は現在のピン位置を中心に半径 250 nm（約 463 km）で
  実行されます。これは上流の `/v2/point` エンドポイントが受け付ける
  最大半径で、これより大きな値を渡すと HTTP 403 が返ります。403 の
  レスポンスには CORS ヘッダが付かないため、ブラウザ側では
  「Failed to fetch」という誤解を招く CORS エラーとして報告されます。
  ピンは初回ロード時に東京に置かれ、地球をクリックするたびに移動
  します。手動 Refresh と Auto refresh はどちらも現在のピン位置を
  使います。半径を小さくしたい場合は `index.html` 内の
  `FETCH_RADIUS_NM` 定数を編集してください。
- インデックスは意図的に非永続です。ページロードのたびに OPFS を
  クリアし、airplanes.live の最新スナップショットから作り直します。
  OPFS 永続化を確認したい場合は basic / geo サンプルを参照して
  ください。
