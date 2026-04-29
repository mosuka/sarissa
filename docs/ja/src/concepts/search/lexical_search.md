# Lexical 検索

Lexical 検索は、転置インデックスに対してキーワードをマッチングすることでドキュメントを検索します。Laurus は、完全一致、フレーズ一致、あいまい一致など、豊富なクエリタイプを提供します。

## 基本的な使い方

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::search::searcher::LexicalSearchQuery;

let request = SearchRequestBuilder::new()
    .lexical_query(
        LexicalSearchQuery::Obj(
            Box::new(TermQuery::new("body", "rust"))
        )
    )
    .limit(10)
    .build();

let results = engine.search(request).await?;
```

## クエリタイプ

### TermQuery

特定のフィールドに完全一致するタームを含むドキュメントをマッチングします。

```rust
use laurus::lexical::TermQuery;

// Find documents where "body" contains the term "rust"
let query = TermQuery::new("body", "rust");
```

> **注意:** タームは解析後にマッチングされます。フィールドが `StandardAnalyzer` を使用している場合、インデキシングされたテキストとクエリタームの両方が小文字化されるため、`TermQuery::new("body", "rust")` は元テキスト内の "Rust" にもマッチします。

### PhraseQuery

正確なタームの並びを含むドキュメントをマッチングします。

```rust
use laurus::lexical::query::phrase::PhraseQuery;

// Find documents containing the exact phrase "machine learning"
let query = PhraseQuery::new("body", vec!["machine".to_string(), "learning".to_string()]);

// Or use the convenience method from a phrase string:
let query = PhraseQuery::from_phrase("body", "machine learning");
```

フレーズクエリは、ターム位置情報が格納されている必要があります（`TextOption` のデフォルト設定）。

### BooleanQuery

複数のクエリをブーリアン論理で結合します。

```rust
use laurus::lexical::query::boolean::{BooleanQuery, BooleanQueryBuilder, Occur};

let query = BooleanQueryBuilder::new()
    .must(Box::new(TermQuery::new("body", "rust")))       // AND
    .must(Box::new(TermQuery::new("body", "programming"))) // AND
    .must_not(Box::new(TermQuery::new("body", "python")))  // NOT
    .build();
```

| Occur | 意味 | DSL での表現 |
| :--- | :--- | :--- |
| `Must` | ドキュメントが必ずマッチしなければならない | `+term` または `AND` |
| `Should` | ドキュメントがマッチすべき（スコアをブースト） | `term` または `OR` |
| `MustNot` | ドキュメントがマッチしてはならない | `-term` または `NOT` |
| `Filter` | 必ずマッチする必要があるが、スコアには影響しない | （DSL に相当するものなし） |

### FuzzyQuery

指定された編集距離（レーベンシュタイン距離）内のタームをマッチングします。

```rust
use laurus::lexical::query::fuzzy::FuzzyQuery;

// Find documents matching "programing" within edit distance 2
// This will match "programming", "programing", etc.
let query = FuzzyQuery::new("body", "programing");  // default max_edits = 2
```

### WildcardQuery

ワイルドカードパターンを使用してタームをマッチングします。

```rust
use laurus::lexical::query::wildcard::WildcardQuery;

// '?' matches exactly one character, '*' matches zero or more
let query = WildcardQuery::new("filename", "*.pdf")?;
let query = WildcardQuery::new("body", "pro*")?;
let query = WildcardQuery::new("body", "col?r")?;  // matches "color" and "colour"
```

### PrefixQuery

特定のプレフィックスで始まるタームを含むドキュメントをマッチングします。

```rust
use laurus::lexical::query::prefix::PrefixQuery;

// Find documents where "body" contains terms starting with "pro"
// This matches "programming", "program", "production", etc.
let query = PrefixQuery::new("body", "pro");
```

### RegexpQuery

正規表現パターンにマッチするタームを含むドキュメントをマッチングします。

```rust
use laurus::lexical::query::regexp::RegexpQuery;

// Find documents where "body" contains terms matching the regex
let query = RegexpQuery::new("body", "^pro.*ing$")?;

// Match version-like patterns
let query = RegexpQuery::new("version", r"^v\d+\.\d+")?;
```

> **注意:** `RegexpQuery::new()` は `Result` を返します。正規表現パターンは構築時にバリデーションされ、無効なパターンの場合はエラーが返されます。

### NumericRangeQuery

数値フィールドの値が指定された範囲内にあるドキュメントをマッチングします。

```rust
use laurus::lexical::NumericRangeQuery;
use laurus::lexical::core::field::NumericType;

// Find documents where "price" is between 10.0 and 100.0 (inclusive)
let query = NumericRangeQuery::new(
    "price",
    NumericType::Float,
    Some(10.0),   // min
    Some(100.0),  // max
    true,         // include min
    true,         // include max
);

// Open-ended range: price >= 50
let query = NumericRangeQuery::new(
    "price",
    NumericType::Float,
    Some(50.0),
    None,     // no upper bound
    true,
    false,
);
```

### GeoQuery

2D 地理座標（WGS84 緯度・経度）に基づいてドキュメントをマッチングします。

```rust
use laurus::lexical::query::geo::GeoQuery;

// Find documents within 10 km (= 10 000 m) of Tokyo Station (35.6812, 139.7671)
let query = GeoQuery::within_radius("location", 35.6812, 139.7671, 10_000.0)?; // distance in metres

// Find documents within a bounding box (min_lat, min_lon, max_lat, max_lon)
let query = GeoQuery::within_bounding_box(
    "location",
    35.0, 139.0,  // min (lat, lon)
    36.0, 140.0,  // max (lat, lon)
)?;
```

### Geo3dDistanceQuery / Geo3dBoundingBoxQuery / Geo3dNearestQuery

3 種類のクエリが、ECEF 直交座標（メートル）にバックアップされた 3D の
`Geo3d` フィールドを対象とします。高度が意味を持つ用途や、2D の `Geo`
フィールドでは極特異点が問題になるケースで使ってください。座標系・WGS84
変換ヘルパー・実例については [3D 地理検索](../geo3d.md) を参照してください。

```rust
use laurus::GeoEcefPoint;
use laurus::lexical::query::geo3d::{
    Geo3dDistanceQuery, Geo3dBoundingBoxQuery, Geo3dNearestQuery,
};

let centre = GeoEcefPoint::new(-3_955_182.0, 3_350_553.0, 3_700_276.0);

// 球: `centre` から 5 km 以内のドキュメント
let q = Geo3dDistanceQuery::new("position", centre, 5_000.0);

// 軸並行 3D バウンディングボックス（コンストラクタが軸ごとに min ≤ max を検証）
let min = GeoEcefPoint::new(-4_000_000.0, 3_300_000.0, 3_650_000.0);
let max = GeoEcefPoint::new(-3_900_000.0, 3_400_000.0, 3_750_000.0);
let q = Geo3dBoundingBoxQuery::new("position", min, max)?;

// k-NN: 最も近い 10 件、半径スケジュールをカスタマイズ
let q = Geo3dNearestQuery::new("position", centre, 10)
    .with_initial_radius(500.0)
    .with_max_radius(1_000_000.0);
```

| クエリ | スコア |
| :--- | :--- |
| `Geo3dDistanceQuery` | `1 - distance / radius` を `[0, 1]` にクランプ |
| `Geo3dBoundingBoxQuery` | マッチした全ドキュメントで定数 `1.0` |
| `Geo3dNearestQuery` | 最も近いヒットが `1.0`、返却された集合内で最も遠いヒットが `0.0` となるよう正規化 |

### SpanQuery

ドキュメント内のタームの近接度に基づいてマッチングします。`SpanTermQuery` と `SpanNearQuery` を使用して近接クエリを構築します。

```rust
use laurus::lexical::query::span::{SpanQuery, SpanTermQuery, SpanNearQuery};

// Find documents where "quick" appears near "fox" (within 3 positions)
let query = SpanNearQuery::new(
    "body",
    vec![
        Box::new(SpanTermQuery::new("body", "quick")) as Box<dyn SpanQuery>,
        Box::new(SpanTermQuery::new("body", "fox")) as Box<dyn SpanQuery>,
    ],
    3,    // slop (max distance between terms)
    true, // in_order (terms must appear in order)
);
```

## スコアリング

Lexical 検索結果は **BM25** を使用してスコアリングされます。スコアは、ドキュメントがクエリに対してどの程度関連性があるかを反映します。

- ドキュメント内のターム頻度が高いほどスコアが上昇する
- インデックス全体でタームが希少なほどスコアが上昇する
- 短いドキュメントは長いドキュメントに対してブーストされる

### フィールドブースト

特定のフィールドをブーストして関連性に影響を与えることができます。

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::search::searcher::LexicalSearchQuery;

let request = SearchRequestBuilder::new()
    .lexical_query(LexicalSearchQuery::Obj(Box::new(query)))
    .add_field_boost("title", 2.0)  // title matches count double
    .add_field_boost("body", 1.0)
    .build();
```

## Lexical 検索オプション（SearchRequestBuilder 経由）

Lexical 検索の動作パラメータは `SearchRequestBuilder` のメソッドで設定します。これらは `SearchRequest` の `lexical_options` フィールドに格納されます。

| オプション | デフォルト | 説明 |
| :--- | :--- | :--- |
| `field_boosts` | 空 | フィールドごとのスコア倍率 |
| `min_score` | 0.0 | 最小スコア閾値 |
| `timeout_ms` | None | 検索タイムアウト（ミリ秒） |
| `parallel` | false | セグメント間の並列検索を有効にする |
| `sort_by` | `Score` | 関連性スコアでソート、またはフィールドでソート（`asc` / `desc`） |

### ビルダーメソッド

`SearchRequestBuilder` を使用してクエリとオプションを設定します。

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::search::searcher::{LexicalSearchQuery, SortField};

let request = SearchRequestBuilder::new()
    .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new("body", "rust"))))
    .limit(20)
    .lexical_min_score(0.5)
    .lexical_timeout_ms(5000)
    .lexical_parallel(true)
    .sort_by(SortField::FieldDesc("date".to_string()))
    .add_field_boost("title", 2.0)
    .add_field_boost("body", 1.0)
    .build();
```

## Query DSL の使用

プログラマティックにクエリを構築する代わりに、テキストベースの Query DSL を使用できます。

```rust
use laurus::lexical::QueryParser;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use std::sync::Arc;

let analyzer = Arc::new(StandardAnalyzer::default());
let parser = QueryParser::new(analyzer).with_default_field("body");

// Simple term
let query = parser.parse("rust")?;

// Boolean
let query = parser.parse("rust AND programming")?;

// Phrase
let query = parser.parse("\"machine learning\"")?;

// Field-specific
let query = parser.parse("title:rust AND body:programming")?;

// Fuzzy
let query = parser.parse("programing~2")?;

// Range
let query = parser.parse("year:[2020 TO 2024]")?;
```

完全な構文リファレンスは [Query DSL](../query_dsl.md) を参照してください。

## 次のステップ

- 意味的類似性検索: [Vector 検索](vector_search.md)
- Lexical + Vector の組み合わせ: [ハイブリッド検索](hybrid_search.md)
- DSL の完全な構文リファレンス: [Query DSL](../query_dsl.md)
