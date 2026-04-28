# laurus-python

[![PyPI](https://img.shields.io/pypi/v/laurus.svg)](https://pypi.org/project/laurus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Laurus](https://github.com/mosuka/laurus) 検索エンジンの Python バインディングです。[PyO3](https://github.com/PyO3/pyo3) と [Maturin](https://github.com/PyO3/maturin) を使ってビルドされたネイティブ Rust 拡張を通じて、Python から Lexical 検索、Vector 検索、ハイブリッド検索を利用できます。

## 機能

- **Lexical 検索** -- BM25 スコアリングを備えた転置インデックスによる全文検索
- **Vector 検索** -- Flat、HNSW、IVF インデックスを使用した近似最近傍（ANN）検索
- **ハイブリッド検索** -- フュージョンアルゴリズム（RRF、WeightedSum）で Lexical と Vector の結果を統合
- **豊富なクエリ DSL** -- Term、Phrase、Fuzzy、Wildcard、NumericRange、Geo、Boolean、Span クエリ
- **テキスト解析** -- トークナイザー、フィルター、ステマー、同義語展開
- **柔軟なストレージ** -- インメモリ（一時的）またはファイルベース（永続的）インデックス
- **Python らしい API** -- 型情報を備えた直感的な Python クラス

## インストール

```bash
pip install laurus
```

ソースからビルドする場合（Rust ツールチェーンが必要）:

```bash
pip install maturin
maturin develop
```

## クイックスタート

```python
import laurus

# インメモリインデックスを作成
index = laurus.Index()

# ドキュメントをインデックス
index.put_document("doc1", {"title": "Rust 入門", "body": "システムプログラミング言語です。"})
index.put_document("doc2", {"title": "Python データサイエンス", "body": "Python によるデータ解析。"})
index.commit()

# DSL 文字列で検索
results = index.search("title:rust", limit=5)
for r in results:
    print(f"[{r.id}] score={r.score:.4f}  {r.document['title']}")

# クエリオブジェクトで検索
results = index.search(laurus.TermQuery("body", "python"), limit=5)
```

## インデックスの種類

### インメモリ（一時的）

```python
index = laurus.Index()
```

### ファイルベース（永続的）

```python
schema = laurus.Schema()
schema.add_text_field("title")
schema.add_text_field("body")
schema.add_hnsw_field("embedding", dimension=384)

index = laurus.Index(path="./myindex", schema=schema)
```

## クエリタイプ

| クエリクラス | 説明 |
| :--- | :--- |
| `TermQuery(field, term)` | 完全一致検索 |
| `PhraseQuery(field, [terms])` | フレーズ検索（順序一致） |
| `FuzzyQuery(field, term, max_edits)` | 近似一致検索 |
| `WildcardQuery(field, pattern)` | ワイルドカード検索（`*`、`?`） |
| `NumericRangeQuery(field, min, max)` | 数値範囲検索（int または float） |
| `GeoDistanceQuery.within_radius(field, lat, lon, distance_km)` | 地理的距離検索（半径指定） |
| `GeoBoundingBoxQuery.within_bounding_box(field, min_lat, min_lon, max_lat, max_lon)` | 地理的範囲検索（バウンディングボックス） |
| `Geo3dDistanceQuery.within_sphere(field, x, y, z, radius_m)` | 3D ECEF 球距離検索 |
| `Geo3dBoundingBoxQuery.within_box(field, min_x, min_y, min_z, max_x, max_y, max_z)` | 3D ECEF AABB 検索 |
| `Geo3dNearestQuery.k_nearest(field, x, y, z, k)` | 3D ECEF k 最近傍検索 |
| `BooleanQuery(must, should, must_not)` | 複合ブール検索 |
| `SpanNearQuery(field, [terms], slop)` | 近接検索（スパン） |
| `VectorQuery(field, vector)` | 事前計算済みベクトルによる類似度検索 |
| `VectorTextQuery(field, text)` | テキストからベクトルへの変換と類似度検索（エンベダーが必要） |

## ハイブリッド検索

```python
request = laurus.SearchRequest(
    lexical_query=laurus.TermQuery("body", "rust"),
    vector_query=laurus.VectorQuery("embedding", query_vec),
    fusion=laurus.RRF(k=60.0),
    limit=10,
)
results = index.search(request)
```

### フュージョンアルゴリズム

| クラス | 説明 |
| :--- | :--- |
| `RRF(k=60.0)` | 逆順位フュージョン（ランクベース、ハイブリッドのデフォルト） |
| `WeightedSum(lexical_weight=0.5, vector_weight=0.5)` | スコア正規化後の加重和 |

## テキスト解析

```python
syn_dict = laurus.SynonymDictionary()
syn_dict.add_synonym_group(["ml", "machine learning"])

tokenizer = laurus.WhitespaceTokenizer()
filt = laurus.SynonymGraphFilter(syn_dict, keep_original=True, boost=0.8)

tokens = tokenizer.tokenize("ml tutorial")
tokens = filt.apply(tokens)
for tok in tokens:
    print(tok.text, tok.position, tok.boost)
```

## サンプル

使用例は [`examples/`](examples/) ディレクトリにあります:

| サンプル | 説明 |
| :--- | :--- |
| [quickstart.py](examples/quickstart.py) | 基本的なインデックスと全文検索 |
| [lexical_search.py](examples/lexical_search.py) | 全クエリタイプ（Term、Phrase、Boolean、Fuzzy、Wildcard、Range、Geo、Span） |
| [vector_search.py](examples/vector_search.py) | エンベディングによるセマンティック類似度検索 |
| [hybrid_search.py](examples/hybrid_search.py) | フュージョンによる Lexical 検索と Vector 検索の統合 |
| [synonym_graph_filter.py](examples/synonym_graph_filter.py) | 解析パイプラインでの同義語展開 |
| [search_with_openai.py](examples/search_with_openai.py) | OpenAI によるクラウドベースエンベディング |
| [multimodal_search.py](examples/multimodal_search.py) | テキストから画像、画像から画像への検索 |

## ドキュメント

- [Python バインディングガイド](https://mosuka.github.io/laurus/ja/laurus-python.html)

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](../LICENSE) ファイルを参照してください。
