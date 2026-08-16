# 3D 位置情報検索サンプル（人工衛星）

laurus の 3D 地理空間検索（`Geo3d` フィールド、ECEF 直交座標）を
[CesiumJS](https://cesium.com/platform/cesiumjs/) の 3D 地球儀上で
体験できるシングルページアプリです。軌道要素（element set）は
[CelesTrak](https://celestrak.org/) からセッション毎に 1 回だけ
ダウンロードし、各衛星の位置は SGP4
（[satellite.js](https://github.com/shashwatak/satellite-js)）で
**ブラウザ内で伝播計算**します。Refresh ボタン（または Auto
セレクトの自動更新）は純粋なクライアントサイド計算のみで位置を
更新するため、繰り返しの API 呼び出しは発生しません。

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

1. ページを開くと、東京に黄色のピン📌が立ち、選択中の衛星グループ
   （デフォルト Starlink）の軌道要素を CelesTrak から取得、全衛星を
   SGP4 で現在時刻へ伝播し、ピンに 3D で最も近い 50 機がオレンジ
   マーカー + 高度方向の垂直線で表示されます。
2. **地球をクリック**するとピンが即座にその地点に移動します。
   スナップショットは全球をカバーしているため再取得は不要で、
   空間制約の中心が変わって検索が即座に再実行されるだけです。
3. 検索ボックスにテキスト（例: `STARLINK`、`category:LEO`、
   `category:GEO`）を入れる、または Quick filter（`LEO` / `GEO` /
   `Starlink` / `ISS` / `Clear`）でテキストフィルタをかけ、
   **Show:** セレクト（10 / 25 / 50（デフォルト）/ 100 / 200）
   で表示件数を選びます。結果は常にピンとの 3D ユークリッド距離で
   ソートされ、ドロップダウンの件数までの「最も近い N 機」が表示
   されます。
4. 結果リストの行をクリックすると、その衛星にカメラが flyTo します。
5. **Refresh positions** で現在時刻への再伝播、**Auto** セレクトで
   5s / 10s / 30s / 60s の間隔を選ぶと自動更新を有効にできます
   （タブが非表示の間は一時停止）。更新のたびに衛星が実際に動くのが
   見えます。**Group** セレクトを切り替えると、そのグループの軌道
   要素を（セッション毎 1 回）取得してインデックスを作り直します。
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
  （`callsign`、`registration`、`satellite_type`、`description`、
  `category`）、float フィールド
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
  消えた衛星だけ delete、残りは `putDocument` で上書きします。
  ハイライトマーカーも in-place で位置だけ更新するため、
  自動更新中に視覚的なちらつきが発生しません。
- Cesium Ion トークン**未使用**で動作する CesiumJS ビューア。
  地図画像は OpenStreetMap、地形プロバイダは ellipsoid のみで、
  Ion 経由のリクエストを一切しません。

### クエリ例

ピンは常に存在するため、空間制約 ON のときは必ず `geo3d_nearest`
が DSL に乗ります。チェックボックスを OFF にすると空間制約を外して
テキストにマッチした全衛星を返します。

| 空間制約 | クエリ | 動作 |
| --- | --- | --- |
| ON | （空欄） | ピンに 3D で最も近い 50 機 |
| ON | `callsign:STARLINK*` | 最寄り 50 機のうち Starlink のみ |
| ON | `category:LEO` | 最寄り 50 機のうち低軌道衛星のみ |
| OFF | `callsign:ISS*` | スナップショット内の ISS モジュール |
| OFF | `category:GEO` | スナップショット内の静止衛星全件 |
| OFF | `category:MEO` | スナップショット内の中軌道衛星（GNSS 等）全件 |

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
- 軌道要素は
  `https://celestrak.org/NORAD/elements/gp.php?GROUP=...&FORMAT=json`
  から CORS 対応（`Access-Control-Allow-Origin: *`）で取得します。
  軌道要素は 1 日に数回しか更新されず、CelesTrak は同じデータを
  繰り返しダウンロードするクライアントを一時的にブロックするため、
  デモは各グループを**セッション毎に 1 回だけ**取得し、更新は
  ローカルの再伝播で行います。初回取得がタイムアウトする場合は
  数時間おいてから再試行してください（ブロックは一時的です）。
- 位置は公開平均軌道要素の SGP4 伝播によるもので、精密軌道暦とは
  km オーダーの差があります。減衰済み・不正な軌道要素はスキップ
  され、ログに件数が表示されます。
- 大きなグループは先頭 500 件（`index.html` の `MAX_SATELLITES`）で
  打ち切り、Cesium 描画とインデックス構築を軽量に保ちます。
- インデックスは意図的に非永続です。ページロードのたびに OPFS を
  クリアし、新しい伝播スナップショットから作り直します。
  OPFS 永続化を確認したい場合は basic / geo サンプルを参照して
  ください。
