# VectorEngine Architecture Draft

Last updated: 2025-11-21

## 1. Goals & Non-Goals

- **Doc-centric multi-vector storage**: treat `doc_id` as the primary unit that holds multiple named vector fields (title/body/summary/etc.).
- **Field-specific index tuning**: allow per-field engines (HNSW, IVF, Flat) plus per-field dimensions, distance metrics, quantization/multivector settings.
- **Query flexibility**: support querying a subset of fields, combining multiple query vectors (possibly from different embedders/roles) with explicit weights.
- **Update ergonomics**: enable replacing specific vector fields without re-indexing the full document, while keeping lifecycle bookkeeping consistent.
- **Future sharding/replication**: design configuration structs so that multi-shard replicas can be added later without having to rebuild the API surface.

_Non-goals for the first iteration_:

- Distributed coordination / consensus (single-node assumption stays).
- Learned fusion models (keep `VectorScoreMode` pluggable but ship with heuristic modes only).

## 2. High-Level Structure

```text
VectorEngine
 ├── fields: HashMap<String, VectorField>
 │     └── VectorField = Arc<dyn VectorIndex> + config metadata
 ├── registry: DocumentVectorRegistry
 ├── wal: VectorWal (append-only log of doc/vector mutations)
 └── storage: Storage backend handles snapshots + index blobs
```

- `VectorEngineConfig` is loaded/persisted alongside the registry snapshot.
- Each `VectorField` owns its own ANN index instance (`FlatIndex`, `HnswIndex`, `IvfIndex`, …) but shares the `Storage` handle so segments land under `storage/vector/<field_name>/`.
- `DocumentVectorRegistry` preserves doc-level metadata needed to rehydrate indexes (similar to Qdrant payload catalog).

## 3. Configuration Model

```rust
pub struct VectorEngineConfig {
    pub fields: HashMap<String, VectorFieldConfig>,
    pub default_fields: Vec<String>,
    pub shard_number: NonZeroU32,           // future-proofing
    pub replication_factor: NonZeroU32,     // future-proofing
    pub wal: WalConfig,
    pub metadata: HashMap<String, serde_json::Value>,
}

pub struct VectorFieldConfig {
    pub dimension: usize,
    pub distance: DistanceMetric,
    pub index: VectorIndexKind,             // Flat/Hnsw/Ivf
    pub hnsw: Option<HnswConfig>,
    pub quantization: Option<QuantizationConfig>,
    pub multivector: Option<MultiVectorConfig>,
    pub embedder_id: String,
    pub role: VectorRole,                   // Text/Image/Intent...
    pub base_weight: f32,
    pub allow_variants: bool,               // multiple embedder versions
}
```

### Validation & warnings

- Reject configs where `default_fields` contains undefined fields.
- Warn when `hnsw.inline_storage` is true but `quantization` is absent (mirrors Qdrant’s warning logic).
- Warn when `multivector` is configured but `index` is Flat with `dimension` > 2048 (potential perf cliff).

```rust
pub struct DocumentVectors {
    pub doc_id: u64,
    pub fields: HashMap<String, FieldVectors>,
    pub metadata: HashMap<String, String>,
}

pub struct FieldVectors {
    pub vectors: Vec<StoredVector>,
    pub weight: f32,                        // overrides VectorFieldConfig.base_weight
    pub metadata: HashMap<String, String>,  // e.g. language, section info
}

pub struct StoredVector {
    pub data: Arc<[f32]>,
    pub embedder_id: String,
    pub role: VectorRole,
    pub weight: f32,
    pub attributes: HashMap<String, String>, // chunk ids, original text, etc.
}
```

- `DocumentVectorRegistry` keeps `doc_id -> Vec<FieldEntry>` where `FieldEntry` stores `field_name`, `version`, `vector_count`, `weight`, `metadata`, and a pointer to WAL offsets.
- A lightweight `documents.json` snapshot persists the latest `DocumentVectors` payloads (the same structures written into the WAL) along with the last applied WAL sequence so the in-memory stores can be rebuilt even if the WAL is truncated.
- `manifest.json` mirrors the snapshot metadata (`snapshot_wal_seq`, `wal_last_seq`) so startup can verify that the snapshot and WAL pair is self-consistent before replaying anything, preventing partial/manual edits from sneaking in.
- WAL entries now carry the full `DocumentVectors` payload so the in-memory field stores can be rebuilt after a restart, e.g. `Upsert { document: DocumentVectors }` and `Delete { doc_id }`. Once the WAL grows beyond a small threshold it is compacted back down to a single `Upsert` per live document (tombstones are dropped) to keep recovery times predictable.

## 5. Query API Sketch

```rust
pub struct VectorEngineQuery {
    pub query_vectors: Vec<QueryVector>,
    pub fields: Option<Vec<FieldSelector>>, // default => config.default_fields
    pub limit: usize,
    pub score_mode: VectorScoreMode,
    pub overfetch: f32,
    pub filter: Option<VectorEngineFilter>,
}

pub struct QueryVector {
    pub vector: StoredVector,
    pub weight: f32,
}

pub enum FieldSelector {
    Exact(String),
    Prefix(String),
    Role(VectorRole),
}

pub enum VectorScoreMode {
    WeightedSum,
    MaxSim,
    LateInteraction, // reserved for future rerankers
}

pub struct VectorEngineFilter {
    pub document: MetadataFilter,
    pub field: MetadataFilter,
}

pub struct MetadataFilter {
    pub equals: HashMap<String, String>,
}
```

### Execution Flow

1. **Resolve fields**: evaluate selectors (or fall back to defaults). Provide error with available names if result is empty.
2. **Match query vectors**: filter `query_vectors` per field by `(embedder_id, role)`; reject the query if nothing matches.
3. **Field-local search**: run `VectorIndex::search(FieldSearchInput)` with per-field limit = `ceil(limit * overfetch)` (minimum = `limit`).
4. **Doc-level merge**: combine hits via `score_mode`, factoring `VectorFieldConfig.base_weight`, `FieldVectors.weight`, and `QueryVector.weight`.
5. **Materialize response**: produce `VectorEngineHit { doc_id, score, field_hits, metadata }`.

### Filters & Constraints

- `VectorEngineFilter.document.equals` enforces exact-match key/value checks against `DocumentVectors.metadata` before any field searches are executed.
- `VectorEngineFilter.field.equals` performs the same equality checks on per-document, per-field metadata (`FieldVectors.metadata`). Only hits coming from fields whose metadata satisfy the filter are allowed to bubble up.
- Query vectors must declare an `embedder_id` + `VectorRole` that matches the target field configuration; otherwise the field is skipped and the query errors if no fields remain.
- Later we can reuse `Filter` structs from `lexical::search` for richer comparisons (prefix ranges, numeric ops, etc.).
- 実際の `VectorEngineQuery` 構築例は `examples/vector_search.rs` を参照。
- 実際のエンドツーエンドの利用例は `examples/vector_search.rs` を参照。

## 5.1 Hybrid Engine Integration

- `HybridSearchRequest` exposes `vector_fields`, `vector_filter`, `vector_score_mode`, and `vector_overfetch` so doc-centric options can be layered on top of the familiar lexical-first builder. Internally, the engine now constructs a `VectorEngineQuery` for you and reapplies these overrides every time the request changes.
- Document-level metadata filters are executed before any ANN probes (`VectorEngineFilter.document`), letting you scope hybrid queries to a tenant, language, or workflow flag without paying per-field costs.
- The vector helper enforces both `vector_params.top_k` and `HybridSearchParams::min_vector_similarity`, so hybrid results respect the same constraints you would expect when hitting the vector engine directly.
- `HybridSearchResult::vector_field_hits` retains the field-level matches returned by `VectorEngine`, making it straightforward to surface explanations (“body_embedding matched summary chunk #2”).
- Focused unit tests live next to the engine (`src/hybrid/engine.rs`) and merger (`src/hybrid/search/merger.rs`). Run `cargo test hybrid::engine` or `cargo test hybrid::search::merger` to see how the overrides and metadata propagation behave end to end.

### Reference fixtures & tests

- `resources/vector_engine_sample.json`: 3 件の `DocumentVectors` を収録したサンプルで、フィールド/ドキュメントのメタデータ（`section` や `lang` など）と複数ロールの組み合わせを含む。`MetadataFilter` や `FieldSelector` の挙動確認に利用可能。
- `tests/vector_engine_scenarios.rs`: 上記サンプルを読み込み、フィールド指定、`VectorScoreMode`、ドキュメント/フィールド両方のメタデータフィルタを通す統合テスト。`MemoryStorage` で完結するため CI で高速に実行できる。`cargo test --test vector_engine_scenarios` で単独実行可能。

## 6. Update / Delete Lifecycle

```rust
pub enum UpdatePolicy {
    ReplaceAll,
    ReplaceFields(Vec<String>),
    MergeIncremental,
}

impl VectorEngine {
    pub async fn upsert_document(
        &self,
        doc_vectors: DocumentVectors,
        policy: UpdatePolicy,
    ) -> Result<UpdateResult>;

    pub async fn delete_document(&self, doc_id: u64) -> Result<UpdateResult>;
}
```

Flow:

1. Append intent to WAL (acknowledge once persisted).
2. Resolve policy → determine affected fields.
3. Update registry entries with new `field_version`s.
4. Dispatch vectors to per-field writers (`VectorFieldWriter::apply(FieldVectors, doc_id, version)`).
5. On success, emit `UpdateResult { status: Completed, operation_id }`. On partial failure, rollback registry and mark WAL entry as failed.

## 6.1 API Skeleton (Draft)

```rust
pub struct VectorEngine {
    config: Arc<VectorEngineConfig>,
    fields: HashMap<String, Arc<dyn VectorField>>, // wraps writer+reader
    registry: Arc<DocumentVectorRegistry>,
    wal: Arc<VectorWal>,
    storage: Arc<dyn Storage>,
}

pub trait VectorField: Send + Sync {
    fn name(&self) -> &str;
    fn config(&self) -> &VectorFieldConfig;
    fn writer(&self) -> &dyn VectorFieldWriter;
    fn reader(&self) -> &dyn VectorFieldReader;
}

pub trait VectorFieldWriter: Send + Sync {
    fn add_field_vectors(&self, doc_id: u64, field: &FieldVectors, version: u64) -> Result<()>;
    fn delete_document(&self, doc_id: u64, version: u64) -> Result<()>;
    fn flush(&self) -> Result<()>;
}

pub trait VectorFieldReader: Send + Sync {
    fn search(&self, request: FieldSearchInput) -> Result<FieldSearchResults>;
    fn stats(&self) -> Result<VectorFieldStats>;
}

pub struct DocumentVectorRegistry {
    pub fn upsert(&self, doc_id: u64, fields: &[FieldEntry]) -> Result<RegistryVersion>;
    pub fn delete(&self, doc_id: u64) -> Result<RegistryVersion>;
    pub fn get(&self, doc_id: u64) -> Option<DocumentEntry>;
    pub fn snapshot(&self) -> Result<Vec<u8>>;
}

pub struct VectorWal {
    pub fn append(&self, record: WalRecord) -> Result<SeqNumber>;
    pub fn replay(&self, from: SeqNumber, handler: impl FnMut(WalRecord)) -> Result<()>;
}
```

This sketch keeps the public API async-friendly while hiding storage/index specifics behind `VectorField*` traits.

## 7. Storage & Rebuild Strategy

- **Snapshot**: periodic `registry.snapshot`, `documents.json` (with `last_wal_seq`), plus per-field index checkpoints. Snapshot metadata includes WAL offset.
- **Recovery**: load snapshot → replay WAL from saved offset → rebuild indexes for fields missing segments (if config changed). Since WAL compaction keeps only the latest `DocumentVectors` for active docs, replay cost stays proportional to the current document count.
- Storage layout proposal:

    ```text
    storage/
        registry/
            snapshot.bin
            wal/
        vector/
            <field>/
                segments/
                manifest.json
    ```

### 7.1 DocumentVectorRegistry Persistence Options

| Approach | Pros | Cons | Notes |
|----------|------|------|-------|
| Integrate with existing `storage::structured` backends (Memory/File/Column) | Reuses current abstractions; fewer dependencies; unified backup tooling | Must extend storage traits with WAL + snapshot semantics; existing column store not optimized for high write rate; coupling may complicate future distributed rollouts | Suitable if we keep single-node assumption for a while and want minimal infra footprint |
| Dedicated KV engine (sled / RocksDB) per collection | Mature WAL/snapshot support out of the box; compaction/backpressure handled; easy to tune independently | Adds new dependency/runtime footprint; backup/restore needs coordination with vector segments; multi-tenant resource limits to manage | Preferable if registry write rate is high or we expect future sharding: Qdrant-like design keeps payload catalog in RocksDB |
| Hybrid: structured storage for snapshot, append-only WAL file per collection | Simple implementation, easy to inspect logs; can roll our own compaction | Requires building custom crash recovery logic; WAL growth must be managed manually; eventually might need to migrate to real KV | Good starting point if we want to avoid heavy deps initially, but plan migration path |

Decision guideline: start with **hybrid WAL + structured snapshot** for Phase 1, while keeping the code isolated so we can swap in RocksDB (or similar) once multi-node/sharding work begins.

## 8. Module Impact Overview

| Area | Required Work |
|------|---------------|
| `src/vector/core` | Introduce `StoredVector`, `DocumentVectors`, `VectorRole`, normalization helpers. |
| `src/vector/index` | Add `VectorFieldWriter/Reader` traits; adapt existing Flat/HNSW/IVF writers to accept doc-centric inputs. |
| `src/vector/search` | Expand search params to handle multiple query vectors, field filters, new score modes. |
| `src/hybrid/` | Later consume `VectorEngine::search` outputs instead of dealing with raw `Vector`. |
| `src/storage/` | Provide registry snapshot/WAL primitives, unify with existing columnar/file storage as needed. |
| `src/util/` | Add config validation + warning helpers (similar to Qdrant). |

## 9. Incremental Rollout Plan

1. **Phase 1 – Data model scaffolding**
    - Land new structs + traits directly in the crate while keeping legacy ingestion paths as thin adapters.
    - Add adapters to wrap legacy `(doc_id, field, Vector)` APIs → new DocumentVectors.
2. **Phase 2 – Field-local indexes**
   - Update Flat/HNSW/IVF writers/readers to implement `VectorFieldWriter/Reader`.
   - Build `VectorEngine` layer with in-memory registry.
3. **Phase 3 – Persistence**
   - Introduce registry snapshot/WAL, integrate with existing `Storage` backends.
4. **Phase 4 – Hybrid integration**
   - Switch hybrid search to use `VectorEngineQuery`.
   - Deprecate old vector APIs.
5. **Phase 5 – Advanced scoring / sharding**
   - Add LateInteraction implementation, expose sharding knobs, plan for distributed replication.

## 10. Open Questions

1. Should `DocumentVectorRegistry` live inside `storage::structured` or be a dedicated sled/rocksdb instance?
2. How aggressively do we GC deleted doc_ids? E.g., background compaction vs. reference counting across fields.
3. Minimum safe default for `overfetch_factor` to balance recall vs. latency (Qdrant tends to double K).
4. Do we allow multiple embedders per field simultaneously (`allow_variants`), or force a full field rebuild when switching models?

## 11. Implementation TODOs

### `src/vector/core`

- Introduce `VectorRole`, `StoredVector`, and `DocumentVectors` structs plus helpers for normalization and metadata handling.
- Provide conversion adapters from legacy `Vector` APIs to the new document-centric structures for backward compatibility during rollout.
- Implement `FieldVectors` utilities (weight resolution, metadata merging) and expose serde impls for config snapshots.
- Ensure `Vector` stays usable by legacy callers by offering `From<StoredVector>` conversions where feasible.

### `src/vector/index`

- Define `VectorFieldWriter`/`VectorFieldReader` traits that operate on `(doc_id, FieldVectors)` batches.
- Update Flat/HNSW/IVF implementations to satisfy the new traits and to surface field-local stats (vector count, dim, distance).
- Add shard-aware wrappers (`FieldShardManager`) so multiple shards per field can be introduced later without API churn.
- Provide compatibility layer (`LegacyVectorWriterAdapter`) to bridge existing ingestion paths until full migration.
- Add per-field metrics hooks (ingest QPS, pending docs) for observability, mirroring current `VectorIndexStats` fields.

### `src/vector/search`

- Extend search params to accept `VectorEngineQuery` inputs, including multiple query vectors, field selectors, and score modes.
- Implement score combiners for `MaxSim`, `WeightedSum`, and stub out hooks for `LateInteraction`.
- Add constraint filtering that can read `FieldVectors.metadata` and registry-level attributes.

### `src/vector/collection` (new)

- Implement `VectorEngine` façade that wires configs, registry, field indexes, and WAL together.
- Provide `upsert_document`, `delete_document`, `search`, `stats` APIs with async interfaces.
- Emit warnings/info (`VectorEngineInfo`) similar to Qdrant’s `CollectionInfo` for observability.

### `src/storage`

- Build `DocumentVectorRegistry` backed by snapshot + WAL files, exposing atomic read/write APIs and compaction hooks.
- Ensure storage layout under `storage/vector/<field>` is coordinated with existing `Storage` implementations (memory/file/column).

### `src/hybrid`

- Update hybrid search flow to consume `VectorEngine::search` outputs instead of raw `VectorSearcher` responses.
- Revisit score fusion to allow vector-side `score_mode` outputs and lexical weights to combine cleanly.

### Tooling & Testing

- Keep regression tests for legacy vector APIs so the new data model can coexist while wiring progresses.
- Create integration tests that ingest multi-field documents and validate query/update semantics end-to-end.
- Provide migration utilities (CLI or scripts) to convert existing indexes into VectorEngine snapshots once stable.

## 12. Phase 0 Issue Checklist

Goal: land the minimum scaffolding directly in-tree while still keeping existing query paths untouched until the new layer is fully wired.

| ID | Scope | Deliverable |
|----|-------|------------|
| P0-01 | Define `VectorRole`, `StoredVector`, `DocumentVectors` in `src/vector/core` with serde + conversion helpers | PR adding structs, docs, unit tests |
| P0-02 | Introduce `VectorFieldWriter/Reader` traits and adapt Flat writer to implement them (read path can stay no-op) | PR touching `src/vector/index/flat` + trait module |
| P0-03 | Create `VectorEngine` skeleton with `search` stub returning `unimplemented!()` and ensure it compiles even when unused | PR adding new module + plumbing |
| P0-04 | Implement in-memory `DocumentVectorRegistry` + WAL stub (Vec-backed) for tests | PR under `src/storage/vector_registry` |
| P0-05 | Provide adapter that converts legacy `(doc_id, field, Vector)` ingestion calls into `DocumentVectors` so existing pipelines keep compiling | PR updating ingestion path + runtime toggles |
| P0-06 | Add crate-level documentation + example describing the doc-centric defaults and how to opt into the new API surface | PR updating `README.md`/`docs/vector_engine.md` |

Tracking template for GitHub Issues:

```text
## Summary
- <1–2 sentences>

## Acceptance Criteria
- [ ] User-facing behavior or API description
- [ ] Tests / docs updates

## Dependencies
- Link to preceding P0 tasks if any
```

## 13. Prototype Implementation Plan

1. **Config plumbing anchored in main crate**
    - Expose `VectorEngineConfig` loader and ensure unused code paths stay no-op until higher layers consume them.
2. **Minimal ingestion path**
    - Use in-memory registry + Flat writer to accept `DocumentVectors` and allow `stats()` inspection.
3. **Search happy path stub**
    - Implement `VectorEngine::search` that only supports a single field + single vector (wired to Flat reader) but already obeys the new request struct.
4. **Integration test**
    - Example test in `examples/vector_engine_smoke.rs` that builds a collection, ingests two docs, and queries them.
5. **Feedback loop**
    - Once the prototype compiles and tests pass, gather API feedback before expanding to multi-field/hnsw support.

## 14. Implementation Log

| Date | Change |
|------|--------|
| 2025-11-19 | Added `vector-collection` feature flag and document-centric core types (`src/vector/core/document.rs`). |
| 2025-11-19 | Introduced `src/vector/collection/mod.rs` skeleton (`VectorEngine`, registry + WAL stubs, field traits). |
| 2025-11-20 | Enabled the VectorEngine scaffolding by default (removed feature flag) and flattened the module to `src/vector/collection.rs`. |

---
This draft intentionally mirrors proven ideas from Qdrant’s collection layer while keeping room for Platypus-specific extensions (hybrid search integration, custom score modes). Feedback welcome before evolving into an RFC.
