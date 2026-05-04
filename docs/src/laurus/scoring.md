# Scoring & Ranking

Laurus provides multiple scoring algorithms for lexical search and uses distance-based similarity for vector search. This page covers all scoring mechanisms and how they interact in hybrid search.

## Lexical Scoring

### BM25 (Default)

BM25 is the default scoring function for lexical search. It balances term frequency with document length normalization:

```text
score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
```

Where:

- **tf** -- term frequency in the document
- **IDF** -- inverse document frequency (rarity of the term across all documents)
- **k1** -- term frequency saturation parameter
- **b** -- document length normalization factor
- **doc_len / avg_doc_len** -- ratio of document length to average document length

### ScoringConfig

`ScoringConfig` controls BM25 and other scoring parameters:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `k1` | `f32` | 1.2 | Term frequency saturation. Higher values give more weight to term frequency. |
| `b` | `f32` | 0.75 | Field length normalization. 0.0 = no normalization, 1.0 = full normalization. |
| `tf_idf_boost` | `f32` | 1.0 | Global TF-IDF boost factor |
| `enable_field_norm` | `bool` | true | Enable field length normalization |
| `field_boosts` | `HashMap<String, f32>` | empty | Per-field score multipliers |
| `enable_coord` | `bool` | true | Enable query coordination factor (matched_terms / total_query_terms) |

### Alternative Scoring Functions

| Function | Description |
| :--- | :--- |
| `BM25ScoringFunction` | BM25 with configurable k1 and b (default) |
| `TfIdfScoringFunction` | Log-normalized TF-IDF with field length normalization |
| `VectorSpaceScoringFunction` | Cosine similarity over document term vector space |
| `CustomScoringFunction` | User-provided closure for custom scoring logic |

### ScoringRegistry

The `ScoringRegistry` provides a central registry for scoring algorithms:

```rust
// Pre-registered algorithms:
// - "bm25"          -> BM25ScoringFunction
// - "tf_idf"        -> TfIdfScoringFunction
// - "vector_space"  -> VectorSpaceScoringFunction
```

### BM25Plan (precomputed query)

When scoring many documents against the same query, `BM25Plan` materializes per-term IDF, length-normalization invariants, and `k1 + 1` once at query build time, then scores each document with a tight numeric loop. Two scoring methods are exposed:

- `BM25Plan::score(&doc_stats)` — drop-in faster replacement for `BM25ScoringFunction::score`. Still looks up term frequencies via the existing `HashMap<String, u64>` on `DocumentStats`.
- `BM25Plan::score_packed(&term_freqs, doc_length)` — accepts a packed `&[u64]` of term frequencies indexed by the position of each term in the original `query_terms`. Avoids the per-document string-keyed `HashMap` lookup entirely; use it when the caller can pre-pack frequencies once per document.

```rust
use laurus::lexical::search::scoring::bm25::{BM25Plan, ScoringConfig};

let plan = BM25Plan::new(&query_terms, &collection_stats, &ScoringConfig::default());

// Hash-map path (drop-in faster):
let score = plan.score(&doc_stats);

// Packed path (avoids per-doc hash lookup):
let term_freqs: Vec<u64> = query_terms
    .iter()
    .map(|t| *doc_stats.term_frequencies.get(t).unwrap_or(&0))
    .collect();
let score = plan.score_packed(&term_freqs, doc_stats.doc_length);
```

`BM25ScoringFunction::score` is preserved for callers that score one document at a time.

### Field Boosts

Field boosts multiply the score contribution from specific fields. This is useful when some fields are more important than others:

```rust
use std::collections::HashMap;

let mut field_boosts = HashMap::new();
field_boosts.insert("title".to_string(), 2.0);  // title matches score 2x
field_boosts.insert("body".to_string(), 1.0);   // body matches score 1x
```

### Coordination Factor

When `enable_coord` is true, the `AdvancedScorer` applies a coordination factor:

```text
coord = matched_query_terms / total_query_terms
```

This rewards documents that match more query terms. For example, if the query has 3 terms and a document matches 2 of them, the coordination factor is 2/3 = 0.667.

## Vector Scoring

Vector search ranks results by distance-based similarity:

```text
similarity = 1 / (1 + distance)
```

The distance is computed using the configured distance metric:

| Metric | Description | Best For |
| :--- | :--- | :--- |
| `Cosine` | 1 - cosine similarity | Text embeddings (most common) |
| `Euclidean` | L2 distance | Spatial data |
| `Manhattan` | L1 distance | Feature vectors |
| `DotProduct` | Negated dot product | Pre-normalized vectors |
| `Angular` | Angular distance | Directional similarity |

## Hybrid Search Score Normalization

When lexical and vector results are combined, their scores must be made comparable.

### RRF (Reciprocal Rank Fusion)

RRF avoids score normalization entirely by using ranks instead of raw scores:

```text
rrf_score = sum(1 / (k + rank))
```

The `k` parameter (default: 60) controls smoothing. Higher values give less weight to top-ranked results.

### WeightedSum

WeightedSum normalizes scores from each search type independently using min-max normalization, then combines them:

```text
norm_score = (score - min_score) / (max_score - min_score)
final_score = (norm_lexical * lexical_weight) + (norm_vector * vector_weight)
```

Both weights are clamped to [0.0, 1.0].
