# BKD-Tree

Laurus は数値・日時・地理ポイントなどのデータを **BKD-Tree** (Block KD-Tree)
に格納する。BKD-Tree はディスク常駐の多次元インデックスで、レンジ・バウンディング
ボックス・距離・k 近傍 (k-NN) の各クエリを単一のファイル形式で扱える。

「空間的な形」を持つあらゆるフィールド型はこの BKD プリミティブを共有する：

| フィールド型 | 次元数 | 座標空間 |
| :--- | :---: | :--- |
| `Integer` / `Float`（単一値・多値） | 1 | スカラー |
| `DateTime` | 1 | Unix マイクロ秒（UTC） |
| `Geo` | 2 | 緯度・経度（度） |
| `Geo3d` | 3 | ECEF 直交座標（メートル） |

新しい空間フィールド型を追加する作業は、次元数を選んでクエリ側の
[`IntersectVisitor`](#intersectvisitor-プロトコル) を書くだけに帰着する。
ライタ・リーダ・オンディスクレイアウトはそのまま再利用される。

## ファイルフォーマット (Version 2)

`.bkd` セグメントファイルは自己完結型のバイナリで、3 つの領域から成る：

```text
+----------------------------------------+
| File Header                            |   固定長・バージョンタグ付き
+----------------------------------------+
| Leaf Blocks                            |   row-major points + doc_ids
|   leaf 0                               |
|   leaf 1                               |
|   ...                                  |
+----------------------------------------+
| Index Nodes                            |   内部ナビゲーションノード
|   node N-1                             |
|   ...                                  |
|   node 0  (root, written last)         |
+----------------------------------------+
```

ヘッダ (`BKDFileHeader`) は `magic`、`version`（現在は `2`）、`num_dims`、
`bytes_per_dim`、総ポイント数、リーフブロック数、**全体の軸ごと min/max**、
インデックス領域とルートノードへのオフセットを保持する。

### Leaf Block レイアウト

各リーフブロックは、その部分木に属するポイントを格納する：

```text
count               u32                       — リーフ内のポイント数
leaf_min            [f64; num_dims]           — リーフレベルの AABB 下端
leaf_max            [f64; num_dims]           — リーフレベルの AABB 上端
point_values        [f64; count * num_dims]   — row-major のポイント座標
doc_ids             [u64; count]              — ドキュメント ID の並列配列
```

リーフごとに AABB を持たせることで、クエリ領域がリーフの外側にある場合
(`Outside`) や全内包される場合 (`Inside`) には、ポイントを 1 つも読まずに
リーフ全体を判定できる。

### 内部 Index Node レイアウト

内部ノードは分割情報に加え、子ごとの AABB も保持する：

```text
split_dim           u32                       — 分割する軸
split_value         f64                       — 分割しきい値
left_min            [f64; num_dims]           — 左部分木の AABB 下端
left_max            [f64; num_dims]           — 左部分木の AABB 上端
right_min           [f64; num_dims]           — 右部分木の AABB 下端
right_max           [f64; num_dims]           — 右部分木の AABB 上端
left_offset         u64                       — 左子ノードのファイルオフセット
right_offset        u64                       — 右子ノードのファイルオフセット
```

> ノードごとの AABB（v2 で追加）は、分割値だけを持っていた v1 レイアウトを
> 置き換える。これにより `Inside` / `Outside` の枝刈りが、再帰的な
> 探索ではなく定数時間の矩形判定で済むようになった。

## ビルドアルゴリズム

`BKDWriter::write` は、平坦な row-major のポイントバッファと並列の `doc_ids`
バッファからツリーを構築する。構築は **最も広い軸で分割する (widest-axis
split)** ヒューリスティクスで進む：

1. 入力部分集合の AABB を計算する。
2. `(max - min)` レンジが最も広い軸を選ぶ（同点時は次元番号の小さい方を
   採用して決定的にする）。
3. インデックスの並びをその軸でソートし、中央値で分割する。
4. 部分木が `block_size`（既定 `512`）以下のポイント数になるまで再帰し、
   そうなったらリーフとして書き出す。
5. 子が flush された後、各親の `left_offset` / `right_offset` を後埋めする。

ビルダはポイント／doc_id バッファ自体ではなく**インデックスの並び**
（permutation）をソートするため、ポイント数に関わらずポイント単位の
ヒープ確保を一切おこなわない。

### 数値ロバスト性

座標は全順序で比較できる必要がある。`BKDWriter::write` は `NaN` を明示的に
拒否する。`NaN` には順序が定義されておらず、分割判定とノードごとの AABB
不変条件を破壊するためである。`±INFINITY` は両方とも受理され、クエリでは
「無限大」を表す自然なセンチネルとして機能する。

## IntersectVisitor プロトコル

BKD インデックスへのクエリは
[`IntersectVisitor`](https://github.com/mosuka/laurus/blob/main/laurus/src/lexical/index/structures/visitor.rs)
の実装として表現する。リーダはツリーを辿りながら、ビジタに 3 種類の
情報を尋ねる：

```rust
pub enum CellRelation {
    Inside,   // 部分木全体がヒット — ポイント単位の照合不要
    Outside,  // 部分木全体をスキップできる
    Crosses,  // 再帰、もしくはリーフをポイント単位で照合
}

pub trait IntersectVisitor {
    fn compare(&self, cell: &AABB) -> CellRelation;
    fn visit_inside(&mut self, doc_id: u64);
    fn visit(&mut self, doc_id: u64, point: &[f64]);
}
```

リーダのトラバーサルは次のように進む：

```mermaid
graph TD
    A["compare(node.aabb)"]
    A -->|Inside| B["部分木の各 doc_id を visit_inside(doc_id) で報告<br/>（座標は読まない）"]
    A -->|Outside| C["部分木をスキップ"]
    A -->|Crosses, 内部ノード| D["子ノードへ再帰"]
    A -->|Crosses, リーフ| E["各ポイントについて visit(doc_id, point)<br/>ヒット判定はビジタに委ねる"]
```

この 3 値分類こそが枝刈りを実現する鍵である。常に `Crosses` を返すビジタを
書いても結果は正しい — 単にリーフ全件走査に退化するだけだ。

### レンジクエリ

レガシーの `BKDTree::range_search` API は `intersect` の薄いラッパに
なっている。`RangeQueryVisitor` を半開区間／閉区間のパラメータから組み立て、
無限大の `None` スロットを `±INFINITY` に変換する。境界の包含・排他は
ビジタ自身が処理する。

### 3D 地理クエリ

3 つのビジタが [`laurus::lexical::query::geo3d`](geo3d.md) に存在し、
`Geo3d` (3D ECEF) フィールドをターゲットにする：

| クエリ | `compare` の判定領域 | `visit` の点ごとの判定 |
| :--- | :--- | :--- |
| `Geo3dDistanceQuery` | 球 `(centre, radius)` と AABB | ユークリッド距離 ≤ radius |
| `Geo3dBoundingBoxQuery` | クエリ AABB と セル AABB | 点がクエリ AABB に内包 |
| `Geo3dNearestQuery` (k-NN) | クエリ点を中心に拡大していく球 | 距離 ≤ 現在の k 番目最良値 |

同じプリミティブで将来のあらゆる空間クエリ（ポリゴンクエリや 2D `Geo`
の大円距離クエリなど）を新しいビジタとして実装できる。

## リーダの内部実装

`BKDReader::intersect` は 1 クエリにつき 1 つのスクラッチバッファ
(`IntersectScratch`) を使う。バッファは出会った最大のリーフサイズまで
拡大されたあと、後続のリーフでも再利用される。結果として、何枚のリーフを
辿っても、1 クエリの間にアロケータに触れる回数はごく少数で済む。

単一リーフだけのツリー（非常に小さなフィールド）は特別扱いされる：
「ルートオフセット」がそのまま唯一のリーフを指すため、内部ノードの
降下処理は完全にスキップされる。

## 関連項目

- [3D 地理検索 (ECEF)](geo3d.md) — ECEF 距離・バウンディング
  ボックス・k-NN を実装した具体的な BKD ベースのビジタ。
- [Lexical インデクシング](indexing/lexical_indexing.md) — `.bkd`
  セグメントファイルがセグメント全体のレイアウト内のどこに位置するか。
- [Lexical 検索](search/lexical_search.md) — `NumericRangeQuery`、
  `GeoQuery`、`Geo3dDistanceQuery` の Rust API エントリポイント。
