# BKD-Tree

Laurus stores numeric, datetime, and geographic point data in a **BKD-Tree**
(Block KD-Tree) — a disk-resident, multi-dimensional index that supports
range, bounding-box, distance, and k-nearest-neighbour queries on the same
underlying file format.

The BKD primitive is shared by every "spatial-shaped" field type:

| Field type | Dimensions | Coordinate space |
| :--- | :---: | :--- |
| `Integer` / `Float` (single- or multi-valued) | 1 | scalar |
| `DateTime` | 1 | Unix microseconds (UTC) |
| `Geo` | 2 | latitude / longitude (degrees) |
| `Geo3d` | 3 | ECEF Cartesian (metres) |

Adding a new spatial field type therefore boils down to picking a
dimensionality and writing a query-side
[`IntersectVisitor`](#query-the-intersectvisitor-protocol) — the writer,
reader, and on-disk layout are reused unchanged.

## File Format (Version 2)

A `.bkd` segment file is a self-contained binary blob made of three regions:

```text
+----------------------------------------+
| File Header                            |   fixed-size, version-tagged
+----------------------------------------+
| Leaf Blocks                            |   row-major points + doc_ids
|   leaf 0                               |
|   leaf 1                               |
|   ...                                  |
+----------------------------------------+
| Index Nodes                            |   internal navigation nodes
|   node N-1                             |
|   ...                                  |
|   node 0  (root, written last)         |
+----------------------------------------+
```

The header (`BKDFileHeader`) records `magic`, `version` (currently `2`),
`num_dims`, `bytes_per_dim`, the total point count, the number of leaf
blocks, the **per-axis global min/max** for the whole tree, and offsets to
the index region and the root node.

### Leaf Block Layout

Each leaf block stores the points that fall inside its subtree:

```text
count               u32                       — number of points in the leaf
leaf_min            [f64; num_dims]           — leaf-level AABB minimum
leaf_max            [f64; num_dims]           — leaf-level AABB maximum
point_values        [f64; count * num_dims]   — row-major point coordinates
doc_ids             [u64; count]              — parallel doc-id buffer
```

The per-leaf AABB lets the reader prune the leaf without scanning any of its
points when the query region is fully outside (`Outside`) or fully inside
(`Inside`) the leaf.

### Internal Index Node Layout

Internal nodes carry both the split decision and the per-child AABB:

```text
split_dim           u32                       — axis to split on
split_value         f64                       — split threshold on that axis
left_min            [f64; num_dims]           — left subtree AABB min
left_max            [f64; num_dims]           — left subtree AABB max
right_min           [f64; num_dims]           — right subtree AABB min
right_max           [f64; num_dims]           — right subtree AABB max
left_offset         u64                       — file offset of left child
right_offset        u64                       — file offset of right child
```

> Per-node AABBs (added in v2) replace the v1 layout that only carried the
> split value. They make `Inside` / `Outside` pruning a constant-time
> rectangle test instead of a recursive descent.

## Build Algorithm

`BKDWriter::write` builds the tree from a flat row-major point buffer plus a
parallel `doc_ids` buffer. Construction is driven by the **widest-axis split**
heuristic:

1. Compute the AABB of the input subset.
2. Pick the axis whose `(max - min)` range is the widest (ties broken by
   lower dimension index for determinism).
3. Sort the index permutation by that axis and split at the median.
4. Recurse on the two halves until a subtree fits in `block_size` points
   (default `512`); emit it as a leaf.
5. Back-patch each parent's `left_offset` / `right_offset` once the children
   have been flushed.

The builder sorts an `index permutation` rather than the point/doc-id buffers
themselves, so it does **not** allocate per-point storage no matter how many
points are written.

### Numeric Robustness

Coordinates must be totally orderable. `BKDWriter::write` rejects `NaN`
explicitly because `NaN` has no defined ordering and would corrupt the
split decisions and per-node AABB invariants. Both `±INFINITY` are accepted
and act as natural sentinels for "unbounded" semantics in queries.

## Query: The IntersectVisitor Protocol

Queries against a BKD index are expressed as an
[`IntersectVisitor`](https://github.com/mosuka/laurus/blob/main/laurus/src/lexical/index/structures/visitor.rs)
implementation. The reader walks the tree and asks the visitor three things:

```rust
pub enum CellRelation {
    Inside,   // entire subtree is a hit — collect without per-point checks
    Outside,  // entire subtree can be skipped
    Crosses,  // recurse, or filter the leaf per-point
}

pub trait IntersectVisitor {
    fn compare(&self, cell: &AABB) -> CellRelation;
    fn visit_inside(&mut self, doc_id: u64);
    fn visit(&mut self, doc_id: u64, point: &[f64]);
}
```

The reader's traversal is:

```mermaid
graph TD
    A["compare(node.aabb)"]
    A -->|Inside| B["visit_inside(doc_id) for every doc<br/>under the subtree — no point read"]
    A -->|Outside| C["skip subtree"]
    A -->|Crosses, internal| D["recurse into children"]
    A -->|Crosses, leaf| E["visit(doc_id, point) per point<br/>visitor decides what is a hit"]
```

This three-valued classification is what unlocks pruning. A visitor that
always returns `Crosses` would still produce correct results — it would just
degrade to a full leaf scan.

### Range Queries

The legacy `BKDTree::range_search` API is now a thin wrapper around
`intersect`: it constructs a `RangeQueryVisitor` from the half-open / closed
range parameters and converts unbounded `None` slots into `±INFINITY`. The
visitor handles inclusive-vs-exclusive boundary checks itself.

### 3D Geographic Queries

Three additional visitors live in [`laurus::lexical::query::geo3d`](geo3d.md)
and target `Geo3d` (3D ECEF) fields:

| Query | Region tested by `compare` | Per-point check in `visit` |
| :--- | :--- | :--- |
| `Geo3dDistanceQuery` | sphere `(centre, radius)` vs. cell AABB | Euclidean distance ≤ radius |
| `Geo3dBoundingBoxQuery` | query AABB vs. cell AABB | point inside query AABB |
| `Geo3dNearestQuery` (k-NN) | expanding sphere around the query point | distance ≤ current k-th best |

The same primitive could host any future spatial query — for example, polygon
queries or great-circle 2D `Geo` queries — by writing a new visitor.

## Reader Internals

`BKDReader::intersect` uses a single per-query scratch buffer
(`IntersectScratch`) that grows to the size of the largest leaf encountered
and is then reused for every subsequent leaf, so a query touches the
allocator at most a handful of times regardless of how many leaves it visits.

Single-leaf trees (very small fields) are handled as a special case: the
"root offset" points directly at the only leaf and the reader skips the
internal-node descent entirely.

## See Also

- [3D Geographic Search](geo3d.md) — concrete BKD-backed visitors for ECEF
  distance, bounding-box, and k-NN queries.
- [Lexical Indexing](indexing/lexical_indexing.md) — where the `.bkd` segment
  file fits within the broader segment layout.
- [Lexical Search](search/lexical_search.md) — `NumericRangeQuery`,
  `GeoDistanceQuery` / `GeoBoundingBoxQuery`, and `Geo3dDistanceQuery` programmatic entry points.
