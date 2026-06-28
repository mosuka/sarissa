pub mod query;
pub mod schema;
pub mod search;
pub mod type_coercion;
pub mod type_inference;

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::keyword::KeywordAnalyzer;
use crate::analysis::analyzer::per_field::PerFieldAnalyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::data::Document;
use crate::embedding::cache::{EmbeddingCache, embed_batch_with_cache};
use crate::embedding::embedder::Embedder;
use crate::error::Result;
use crate::lexical::store::LexicalStore;
use crate::lexical::store::config::LexicalIndexConfig;
use crate::storage::Storage;
use crate::storage::prefixed::PrefixedStorage;
use crate::store::log::{DocumentLog, LogEntry, WalSyncPolicy};
use crate::vector::store::VectorStore;
use crate::vector::store::config::VectorIndexConfig;

use self::schema::Schema;

/// Combined statistics from both the lexical and vector stores.
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    /// Total number of documents in the index (from the lexical store).
    pub document_count: u64,
    /// Per-field vector statistics, keyed by field name.
    /// Empty when the schema contains no vector fields.
    pub vector_fields: HashMap<String, crate::vector::index::field::VectorFieldStats>,
}

/// Background timer that periodically forces the WAL durable under a
/// [`WalSyncPolicy::Group`] configured with a `max_interval` (Issue #542, Phase
/// 4b).
///
/// Runs a dedicated thread that calls [`DocumentLog::flush_wal`] every interval
/// — a no-op when nothing is pending, thanks to the dirty guard — so a trailing
/// partial batch under a low ingest rate is not left unsynced indefinitely (the
/// record/byte thresholds may never be reached). Dropping the timer wakes the
/// thread immediately and joins it. Native targets only; on `wasm32` (no
/// background threads) the timer is never constructed and the interval is
/// ignored.
#[cfg(not(target_arch = "wasm32"))]
struct WalFlushTimer {
    /// Sending on (or dropping) this channel signals the thread to stop.
    stop: std::sync::mpsc::Sender<()>,
    /// Join handle for the flush thread, taken and joined on drop.
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl WalFlushTimer {
    /// Spawn the flush thread for `doc_log`, forcing the WAL durable every
    /// `interval`.
    ///
    /// # Arguments
    ///
    /// * `doc_log` - The document log whose WAL is flushed on each tick.
    /// * `interval` - How often to flush the WAL.
    ///
    /// # Errors
    ///
    /// Returns an error if the OS thread cannot be spawned.
    fn spawn(doc_log: Arc<DocumentLog>, interval: std::time::Duration) -> Result<Self> {
        use std::sync::mpsc::RecvTimeoutError;

        let (stop, rx) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::Builder::new()
            .name("laurus-wal-flush".to_string())
            .spawn(move || {
                // Keep ticking while each wait ends in a timeout: flush the WAL
                // (a no-op when there is nothing pending). Any other outcome — a
                // stop signal (`Ok`) or the sender being dropped (`Disconnected`)
                // — ends the loop and the thread.
                while let Err(RecvTimeoutError::Timeout) = rx.recv_timeout(interval) {
                    if let Err(e) = doc_log.flush_wal() {
                        log::warn!("WAL flush timer: failed to flush WAL: {e}");
                    }
                }
            })?;
        Ok(Self {
            stop,
            handle: Some(handle),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for WalFlushTimer {
    fn drop(&mut self) {
        // Wake the thread immediately so it exits without waiting out the
        // current interval, then join it.
        let _ = self.stop.send(());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Unified Engine that manages both Lexical and Vector indices.
///
/// This engine acts as a facade, coordinating document ingestion and search
/// across the underlying specialized engines. All index mutations are
/// WAL-backed via [`DocumentLog`] for crash-recovery durability.
///
/// A system field `_id` is automatically injected into every indexed document
/// to track the external document identifier.
pub struct Engine {
    schema: RwLock<Schema>,
    lexical: LexicalStore,
    vector: VectorStore,
    log: Arc<DocumentLog>,
    /// Pre-constructed analyzers registered at build time and consulted
    /// before built-in names and `schema.analyzers` when resolving
    /// per-field analyzer references. See
    /// [`EngineBuilder::register_runtime_analyzer`].
    runtime_analyzers: HashMap<String, Arc<dyn Analyzer>>,
    /// Optional LRU cache for query-time embeddings (Issue #678).
    ///
    /// `None` when the cache is disabled (the default); enabled via
    /// [`EngineBuilder::embedding_cache_capacity`]. Shared (via `Arc`) with
    /// the [`VectorQueryParser`](crate::vector::query::parser::VectorQueryParser)
    /// built in [`Self::unified_query_parser`], so both the direct
    /// `Payloads` path and the DSL path hit the same cache.
    embedding_cache: Option<Arc<EmbeddingCache>>,
    /// Background WAL flush timer for a [`WalSyncPolicy::Group`] configured with
    /// a `max_interval`. `None` when no interval is set. Held only to keep the
    /// timer thread alive for the engine's lifetime; dropping the engine stops
    /// it. Absent on `wasm32` (no background threads — the interval is ignored).
    #[cfg(not(target_arch = "wasm32"))]
    _wal_flush_timer: Option<WalFlushTimer>,
}

use crate::engine::search::{FusionAlgorithm, SearchResult};

impl Engine {
    /// Create a new Unified Engine with default analyzer and no embedder.
    ///
    /// For custom analyzer or embedder configuration, use [`Engine::builder`].
    ///
    /// # Errors
    ///
    /// Returns an error if storage initialization, index creation, or
    /// WAL recovery fails.
    pub async fn new(storage: Arc<dyn Storage>, schema: Schema) -> Result<Self> {
        EngineBuilder::new(storage, schema).build().await
    }

    /// Create an [`EngineBuilder`] for custom configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let engine = Engine::builder(storage, schema)
    ///     .analyzer(Arc::new(StandardAnalyzer::default()))
    ///     .embedder(Arc::new(MyEmbedder))
    ///     .build()
    ///     .await?;
    /// ```
    pub fn builder(storage: Arc<dyn Storage>, schema: Schema) -> EngineBuilder {
        EngineBuilder::new(storage, schema)
    }

    /// Recover index state from the document log.
    ///
    /// Replays every WAL record that is newer than each store's persisted
    /// `last_wal_seq` checkpoint. Recovery is **idempotent**: each record is
    /// re-applied under its originally recorded `doc_id`, so re-running it
    /// overwrites rather than duplicates. The lexical and vector stores track
    /// their checkpoints independently, so a commit that failed partway (leaving
    /// the stores at different `last_wal_seq` values) is reconciled here — each
    /// store re-applies only what it is missing. See [`Self::commit`] for the
    /// ordering guarantees that make this safe (Issue #821).
    async fn recover(&self) -> Result<()> {
        // read_all() internally syncs next_doc_id with doc_store segments.
        let records = self.log.read_all()?;

        if records.is_empty() {
            return Ok(());
        }

        let vector_last_seq = self.vector.last_wal_seq();
        let lexical_last_seq = self.lexical.last_wal_seq();

        for record in records {
            if record.seq <= vector_last_seq && record.seq <= lexical_last_seq {
                continue;
            }

            match record.entry {
                LogEntry::Upsert {
                    doc_id,
                    external_id: _,
                    document,
                } => {
                    // Restore document into document store
                    let stored_doc = self.filter_stored_fields(&document);
                    self.log.store_document(doc_id, stored_doc);

                    // Re-index into both stores using the recorded doc_id.
                    // Update seq only after BOTH stores succeed to maintain atomicity.
                    if record.seq > lexical_last_seq {
                        self.lexical.upsert_document(doc_id, document.clone())?;
                    }

                    if record.seq > vector_last_seq {
                        // Filter for vector fields
                        let mut vector_doc = Document::new();
                        {
                            let schema = self.schema.read();
                            for (name, val) in &document.fields {
                                if schema.fields.get(name).is_some_and(|fc| fc.is_vector()) {
                                    vector_doc.fields.insert(name.clone(), val.clone());
                                }
                            }
                        }
                        self.vector
                            .upsert_document_by_internal_id(doc_id, vector_doc)
                            .await?;
                    }

                    // Both stores succeeded — now update seq trackers
                    if record.seq > lexical_last_seq {
                        self.lexical.set_last_wal_seq(record.seq)?;
                    }
                    if record.seq > vector_last_seq {
                        self.vector.set_last_wal_seq(record.seq);
                    }
                }
                LogEntry::Delete {
                    doc_id,
                    external_id: _,
                } => {
                    if record.seq > lexical_last_seq {
                        self.lexical.delete_document_by_internal_id(doc_id)?;
                    }
                    if record.seq > vector_last_seq {
                        self.vector.delete_document_by_internal_id(doc_id).await?;
                    }

                    // Both stores succeeded — now update seq trackers
                    if record.seq > lexical_last_seq {
                        self.lexical.set_last_wal_seq(record.seq)?;
                    }
                    if record.seq > vector_last_seq {
                        self.vector.set_last_wal_seq(record.seq);
                    }
                }
            }
        }
        Ok(())
    }

    /// Put (upsert) a document.
    ///
    /// If a document with the same external ID exists, all its chunks are
    /// deleted before the new document is indexed. A `_id` field is
    /// automatically inserted into the document with the provided `id` value.
    /// A WAL entry is written before any index mutations to ensure durability.
    ///
    /// The document fields are routed to the appropriate underlying stores
    /// (lexical or vector) based on the schema field configuration. If the
    /// vector store indexing fails after the lexical store has already been
    /// updated, the lexical insert is rolled back to maintain cross-store
    /// consistency.
    ///
    /// # Parameters
    ///
    /// - `id` - The external document identifier.
    /// - `doc` - The document to index.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL write, deletion of existing documents,
    /// or indexing into either the lexical or vector store fails.
    pub async fn put_document(&self, id: &str, doc: Document) -> Result<()> {
        let _ = self.index_internal(id, doc, false).await?;
        Ok(())
    }

    /// Add a document as a new chunk (always appends, never deletes existing).
    ///
    /// Unlike [`put_document`](Self::put_document), this method does **not**
    /// delete existing documents with the same external ID. Multiple chunks
    /// can share the same ID, which is useful for indexing parts of a large
    /// document (e.g. paragraphs or pages) separately while keeping them
    /// associated with the same logical document.
    ///
    /// A `_id` field is automatically inserted into the document with the
    /// provided `id` value. A WAL entry is written before any index mutations
    /// to ensure durability.
    ///
    /// # Parameters
    ///
    /// - `id` - The external document identifier (may duplicate existing IDs).
    /// - `doc` - The document chunk to index.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL write or indexing into either the lexical
    /// or vector store fails.
    pub async fn add_document(&self, id: &str, doc: Document) -> Result<()> {
        let _ = self.index_internal(id, doc, true).await?;
        Ok(())
    }

    async fn index_internal(&self, id: &str, mut doc: Document, as_chunk: bool) -> Result<u64> {
        // 1. Inject _id field
        use crate::data::DataValue;
        doc.fields
            .insert("_id".to_string(), DataValue::Text(id.to_string()));

        // 1b. Validate reserved field-name namespace, then apply the schema's
        // DynamicFieldPolicy to add / coerce / drop user fields.
        self.apply_dynamic_schema(&mut doc).await?;

        if !as_chunk {
            self.delete_documents(id).await?;
        }

        // 2. Write-Ahead Log: assign doc_id + persist (before any index updates)
        let (doc_id, seq) = self.log.append(id, doc.clone())?;

        // 3. Store only stored fields for retrieval (WAL has full data for recovery)
        let stored_doc = self.filter_stored_fields(&doc);
        self.log.store_document(doc_id, stored_doc);

        // 4. Prepare vector document (extract vector fields only)
        let mut vector_doc = Document::new();
        {
            let schema = self.schema.read();
            for (name, val) in &doc.fields {
                if schema.fields.get(name).is_some_and(|fc| fc.is_vector()) {
                    vector_doc.fields.insert(name.clone(), val.clone());
                }
            }
        }

        // 5. Index into Lexical and Vector stores
        self.lexical.upsert_document(doc_id, doc)?;
        if let Err(e) = self
            .vector
            .upsert_document_by_internal_id(doc_id, vector_doc)
            .await
        {
            // Rollback lexical insert to maintain consistency
            let _ = self.lexical.delete_document_by_internal_id(doc_id);
            return Err(e);
        }

        // 6. Update sub-stores sequence tracker AFTER both stores succeed.
        // This ensures failed index operations are retried on recovery.
        self.lexical.set_last_wal_seq(seq)?;
        self.vector.set_last_wal_seq(seq);

        Ok(doc_id)
    }

    /// Apply the schema's [`DynamicFieldPolicy`](schema::DynamicFieldPolicy)
    /// to an incoming document's fields.
    ///
    /// For each user-supplied field:
    ///
    /// - **Reserved names**: any field name starting with `_` other than
    ///   `_id` is rejected regardless of policy.
    /// - **Declared fields**: the value is coerced to the declared type (see
    ///   [`type_coercion::coerce_value`]).
    /// - **Undeclared fields**: handled according to the policy:
    ///   - `Strict`: ingest fails with an error.
    ///   - `Dynamic`: the field type is inferred (see
    ///     [`type_inference::infer_option_from_data_value`]) and the field
    ///     is added to the schema.
    ///   - `Ignore`: the field is silently dropped.
    ///
    /// # Arguments
    ///
    /// * `doc` - The document to normalise in place.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LaurusError::invalid_argument`] when:
    ///
    /// - A field name collides with the reserved namespace.
    /// - Strict policy is set and an undeclared field is encountered.
    /// - A declared field's value cannot be coerced to its type.
    /// - Dynamic policy is set and an undeclared field has a value whose
    ///   type cannot be inferred (e.g. raw vectors or bytes).
    async fn apply_dynamic_schema(&self, doc: &mut Document) -> Result<()> {
        // 1. Validate reserved field-name namespace for user-supplied keys.
        //    `_id` was just injected by the engine and is always allowed.
        for name in doc.fields.keys() {
            if name == schema::RESERVED_ID_FIELD {
                continue;
            }
            schema::validate_field_name(name)?;
        }

        // 2. Snapshot the current policy and declared-field set.
        let (policy, declared): (
            schema::DynamicFieldPolicy,
            std::collections::HashSet<String>,
        ) = {
            let s = self.schema.read();
            (s.dynamic_field_policy, s.fields.keys().cloned().collect())
        };

        // 3. Partition fields into declared vs undeclared.
        let mut undeclared: Vec<(String, crate::data::DataValue)> = Vec::new();
        let mut declared_updates: Vec<(String, crate::data::DataValue)> = Vec::new();
        for (name, value) in doc.fields.drain() {
            if name == schema::RESERVED_ID_FIELD || declared.contains(&name) {
                declared_updates.push((name, value));
            } else {
                undeclared.push((name, value));
            }
        }

        // 4. Handle undeclared fields per policy.
        match policy {
            schema::DynamicFieldPolicy::Strict => {
                if !undeclared.is_empty() {
                    let names: Vec<&str> = undeclared.iter().map(|(n, _)| n.as_str()).collect();
                    return Err(crate::error::LaurusError::invalid_argument(format!(
                        "undeclared fields {names:?} are not permitted \
                         (DynamicFieldPolicy::Strict)"
                    )));
                }
            }
            schema::DynamicFieldPolicy::Ignore => {
                // Silently drop undeclared fields.
                for (name, _) in &undeclared {
                    log::debug!(
                        target: "laurus::engine::dynamic_schema",
                        "dropping undeclared field '{name}' \
                         (DynamicFieldPolicy::Ignore)",
                    );
                }
                undeclared.clear();
            }
            schema::DynamicFieldPolicy::Dynamic => {
                // Infer a FieldOption for each undeclared field and add it to
                // the schema. Keep the original values on the document so they
                // are indexed under the newly-added fields.
                let mut kept: Vec<(String, crate::data::DataValue)> = Vec::new();
                for (name, value) in undeclared.drain(..) {
                    match type_inference::infer_option_from_data_value(&value)? {
                        Some(option) => {
                            match self.add_field(&name, option).await {
                                Ok(_) => {}
                                Err(e) => {
                                    // Another concurrent ingest may have added
                                    // this field in the meantime. Accept it
                                    // silently; any other failure propagates.
                                    let msg = e.to_string();
                                    if !msg.contains("already exists") {
                                        return Err(e);
                                    }
                                }
                            }
                            kept.push((name, value));
                        }
                        None => {
                            // Null value — skip this field entirely.
                        }
                    }
                }
                undeclared = kept;
            }
        }

        // 5. Coerce declared-field values to their declared types.
        let coerced_declared: Vec<(String, crate::data::DataValue)> = {
            let s = self.schema.read();
            let mut out = Vec::with_capacity(declared_updates.len());
            for (name, value) in declared_updates {
                if name == schema::RESERVED_ID_FIELD {
                    out.push((name, value));
                    continue;
                }
                // The field is declared (we partitioned above) so this lookup
                // is infallible in practice, but guard just in case.
                match s.fields.get(&name) {
                    Some(option) => {
                        let coerced = match type_coercion::coerce_value(&name, option, value) {
                            Ok(v) => v,
                            Err(e) => match policy {
                                schema::DynamicFieldPolicy::Ignore => {
                                    log::debug!(
                                        target: "laurus::engine::dynamic_schema",
                                        "dropping declared field '{name}' due to coercion \
                                         failure ({e}) (DynamicFieldPolicy::Ignore)",
                                    );
                                    continue;
                                }
                                _ => return Err(e),
                            },
                        };
                        out.push((name, coerced));
                    }
                    None => out.push((name, value)),
                }
            }
            out
        };

        // 6. Re-populate the document with processed fields.
        for (name, value) in coerced_declared {
            doc.fields.insert(name, value);
        }
        for (name, value) in undeclared {
            doc.fields.insert(name, value);
        }

        Ok(())
    }

    /// Delete all documents (including chunks) by external ID.
    ///
    /// Looks up all internal document IDs associated with the given external
    /// `id` via the `_id` field in the lexical index, then removes each one
    /// from both the lexical and vector stores. A WAL delete entry is written
    /// for each matched document before mutation.
    ///
    /// If no documents match the given ID, the operation completes
    /// successfully without error (non-existent IDs are silently ignored).
    ///
    /// # Parameters
    ///
    /// - `id` - The external document identifier to delete.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL write, lexical deletion, or vector
    /// deletion fails for any matched document.
    pub async fn delete_documents(&self, id: &str) -> Result<()> {
        let doc_ids = self.lexical.find_doc_ids_by_term("_id", id)?;
        for doc_id in doc_ids {
            // 1. Write to log
            let seq = self.log.append_delete(doc_id, id)?;
            // 2. Delete from Lexical
            self.lexical.delete_document_by_internal_id(doc_id)?;
            // 3. Delete from Vector
            self.vector.delete_document_by_internal_id(doc_id).await?;
            // 4. Update trackers AFTER both deletes succeed.
            // This ensures failed deletes are retried on recovery.
            self.lexical.set_last_wal_seq(seq)?;
            self.vector.set_last_wal_seq(seq);
        }
        Ok(())
    }

    /// Commit changes to both stores and truncate the WAL.
    ///
    /// Persists state in a fixed order — the **commit durability ladder** — that
    /// makes a crash at any step recoverable (Issue #821):
    ///
    /// 1. `flush_wal()` — force the WAL durable (the hard barrier).
    /// 2. `lexical.commit()` — materialize + fsync the lexical store. This is
    ///    where the lexical `last_wal_seq` checkpoint is persisted.
    /// 3. `vector.commit()` — materialize + fsync the vector store.
    /// 4. `commit_documents()` — materialize + fsync the document store.
    /// 5. `truncate()` — replace the WAL with an empty, fsync'd file.
    ///
    /// This order upholds two invariants. First, `last_wal_seq` is persisted
    /// only in step 2+, always *after* the step-1 barrier, so a committed index
    /// can never reference a WAL record that is not yet durable. Second, every
    /// store is fully fsync'd (steps 2–4) before the WAL is truncated (step 5),
    /// so the WAL is discarded only once the data it described is durable. A
    /// crash between any two steps therefore leaves enough in the WAL for the
    /// idempotent replay in [`Self::recover`] to reconstruct a consistent state.
    /// After a successful commit, the WAL is empty and all data is durable.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing the WAL, committing the lexical store,
    /// vector store, document store, or truncating the WAL fails.
    pub async fn commit(&self) -> Result<()> {
        // Hard durability barrier: force the WAL durable before any store
        // materializes its state, so the WAL is never less durable than the
        // committed lexical/vector indexes. A near-no-op under the per-record
        // default (each append already synced, so the dirty guard skips the
        // fsync); the load-bearing step once group commit defers per-append
        // fsync (#542 Phase 3).
        self.log.flush_wal()?;
        self.lexical.commit()?;
        self.vector.commit().await?;
        self.log.commit_documents()?;
        // After successful commit to all stores, truncate the log
        self.log.truncate()?;
        Ok(())
    }

    /// Force every appended-but-unsynced WAL record durable, without a full
    /// [`commit`](Self::commit) (Issue #542).
    ///
    /// Under the default [`WalSyncPolicy::PerRecord`] this is a near-no-op: each
    /// `add`/`delete` already fsyncs, so there is nothing pending. Under
    /// [`WalSyncPolicy::Group`] appends defer their fsync, so this is the way to
    /// bound the crash-loss window at an application-chosen point — analogous to
    /// SQLite's manual WAL checkpoint — without paying the cost of materializing
    /// the lexical/vector indexes that [`commit`](Self::commit) entails.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or fsyncing the open WAL writer fails.
    pub fn flush_wal(&self) -> Result<()> {
        self.log.flush_wal()
    }

    /// Get combined index statistics from both the lexical and vector stores.
    ///
    /// Returns an [`EngineStats`] containing:
    /// - `document_count` from the lexical index (authoritative source).
    /// - Per-field vector statistics from the vector store (empty when no
    ///   vector fields are defined in the schema).
    ///
    /// # Errors
    ///
    /// Returns an error if the lexical index statistics cannot be retrieved.
    pub fn stats(&self) -> Result<EngineStats> {
        let lexical_stats = self.lexical.stats()?;

        let vector_fields = match self.vector.stats() {
            Ok(vs) => vs.fields,
            Err(_) => std::collections::HashMap::new(),
        };

        // doc_count includes deleted documents (soft-deleted, pending merge).
        // Subtract deleted_count for the live document count.
        let live_count = lexical_stats
            .doc_count
            .saturating_sub(lexical_stats.deleted_count);

        Ok(EngineStats {
            document_count: live_count,
            vector_fields,
        })
    }

    /// Return a clone of the current schema.
    ///
    /// This can be used to inspect the schema after dynamic field additions
    /// or to persist it to storage (e.g., `schema.toml`).
    pub fn schema(&self) -> Schema {
        self.schema.read().clone()
    }

    /// Returns the embedder used by the vector store.
    ///
    /// This is useful for constructing a [`VectorQueryParser`] or
    /// [`UnifiedQueryParser`] that shares the same embedder configuration
    /// as the engine.
    pub fn embedder(&self) -> Arc<dyn Embedder> {
        self.vector.embedder()
    }

    /// Create a [`UnifiedQueryParser`] configured for this engine.
    ///
    /// The returned parser uses the engine's analyzer for lexical queries
    /// and the engine's embedder for vector queries. Default fields are
    /// derived from the schema: `default_fields` for lexical queries, and
    /// all vector fields for vector queries.
    ///
    /// # Errors
    ///
    /// Returns an error if the lexical query parser cannot be constructed
    /// (e.g. the analyzer is misconfigured).
    pub fn unified_query_parser(&self) -> Result<self::query::UnifiedQueryParser> {
        let lexical_parser = self.lexical.query_parser()?;
        let embedder = self.embedder();

        let schema = self.schema.read();
        let vector_fields: Vec<String> = schema
            .fields
            .iter()
            .filter(|(_, opt)| opt.is_vector())
            .map(|(name, _)| name.clone())
            .collect();

        let vector_field_set: std::collections::HashSet<String> =
            vector_fields.iter().cloned().collect();

        // All declared field names (lexical + vector), used by the parser to
        // reject typo'd field references at parse time.
        let known_fields: std::collections::HashSet<String> =
            schema.fields.keys().cloned().collect();

        let mut vector_parser = crate::vector::query::parser::VectorQueryParser::new(embedder);
        if !vector_fields.is_empty() {
            vector_parser = vector_parser.with_default_fields(vector_fields);
        }
        if let Some(cache) = &self.embedding_cache {
            // Share the engine's cache so DSL queries hit the same entries
            // as the direct Payloads path (Issue #678).
            vector_parser = vector_parser.with_embedding_cache(cache.clone());
        }

        Ok(
            self::query::UnifiedQueryParser::new(lexical_parser, vector_parser, vector_field_set)
                .with_known_fields(known_fields),
        )
    }

    /// Dynamically add a new field to the engine at runtime.
    ///
    /// This method registers the field in both the engine schema and the
    /// appropriate underlying store (lexical or vector). Only field addition
    /// is supported; removal or type changes are not allowed.
    ///
    /// After adding a field, new documents can include values for this field
    /// and searches can target it. Existing documents are unaffected (they
    /// simply do not have a value for the new field).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name. Must not collide with an existing field.
    /// * `option` - The field configuration (e.g., `FieldOption::Text`,
    ///   `FieldOption::Hnsw`, etc.).
    ///
    /// # Returns
    ///
    /// Returns the updated [`Schema`] on success. The caller is responsible
    /// for persisting it (e.g., writing `schema.toml`).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A field with the same name already exists.
    /// - The field references an unknown analyzer or embedder.
    /// - The underlying store rejects the field.
    pub async fn add_field(&self, name: &str, option: schema::FieldOption) -> Result<Schema> {
        // 1a. Reject reserved field names (e.g. `_`-prefixed except `_id`).
        schema::validate_field_name(name)?;

        // 1. Check for duplicates.
        {
            let schema = self.schema.read();
            if schema.fields.contains_key(name) {
                return Err(crate::error::LaurusError::invalid_argument(format!(
                    "Field '{name}' already exists in the schema"
                )));
            }
        }

        // 2. Register in the appropriate store.
        if option.is_lexical() {
            // Resolve the per-field analyzer if configured.
            let field_analyzer = if let schema::FieldOption::Text(ref text_opt) = option
                && let Some(ref analyzer_spec) = text_opt.analyzer
            {
                let schema = self.schema.read();
                let analyzer = crate::analysis::analyzer::registry::create_analyzer_from_spec(
                    analyzer_spec,
                    &schema.analyzers,
                    &self.runtime_analyzers,
                )
                .map_err(|e| {
                    crate::error::LaurusError::invalid_argument(format!(
                        "Failed to resolve analyzer for field '{name}': {e}"
                    ))
                })?;
                Some(analyzer)
            } else {
                None
            };

            let lexical_opt = option
                .to_lexical()
                .expect("is_lexical() was true but to_lexical() returned None");
            self.lexical.add_field(name, lexical_opt, field_analyzer)?;
        }

        if option.is_vector() {
            // Resolve the per-field embedder if configured.
            // Clone the embedder definition out of the schema lock before
            // calling the async factory so that the non-Send parking_lot
            // guard is not held across an await point.
            let field_embedder = if let Some(embedder_name) = option.embedder_name() {
                let embedder_def = {
                    let schema = self.schema.read();
                    schema.embedders.get(embedder_name).cloned()
                };
                if let Some(def) = embedder_def {
                    Some(
                        crate::embedding::registry::create_embedder_from_definition(
                            embedder_name,
                            &def,
                        )
                        .await?,
                    )
                } else {
                    None
                }
            } else {
                None
            };

            self.vector.add_field(name, field_embedder).await;
        }

        // 3. Update the schema.
        {
            let mut schema = self.schema.write();
            schema.fields.insert(name.to_string(), option);
        }

        Ok(self.schema.read().clone())
    }

    /// Dynamically remove a field from the engine schema at runtime.
    ///
    /// This removes the field definition from the schema so that it is no longer
    /// available for indexing or searching. Existing data already stored in the
    /// index is **not** deleted; it simply becomes inaccessible through the
    /// normal query path.
    ///
    /// For lexical fields, the field is also removed from the underlying
    /// [`LexicalStore`] (if it was dynamically added) and any per-field analyzer
    /// is unregistered. For vector fields, the per-field embedder is
    /// unregistered and writer/searcher caches are invalidated.
    ///
    /// If the deleted field appears in [`Schema::default_fields`], it is removed
    /// from that list as well.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the field to delete
    ///
    /// # Returns
    ///
    /// The updated [`Schema`] after the field has been removed.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No field with the given name exists in the schema.
    /// - The underlying store rejects the deletion.
    pub async fn delete_field(&self, name: &str) -> Result<Schema> {
        // 1. Check that the field exists.
        let option = {
            let schema = self.schema.read();
            schema.fields.get(name).cloned().ok_or_else(|| {
                crate::error::LaurusError::invalid_argument(format!(
                    "Field '{name}' does not exist in the schema"
                ))
            })?
        };

        // 2. Remove from the appropriate store.
        if option.is_lexical() {
            self.lexical.delete_field(name)?;
        }

        if option.is_vector() {
            self.vector.delete_field(name).await;
        }

        // 3. Update the schema.
        {
            let mut schema = self.schema.write();
            schema.fields.remove(name);
            schema.default_fields.retain(|f| f != name);
        }

        Ok(self.schema.read().clone())
    }

    /// Resolve a [`LexicalSearchQuery`] into a concrete [`Query`] object.
    ///
    /// If the query is already an `Obj` variant, it is returned as-is.
    /// If it is a `Dsl` string, it is parsed using the lexical store's
    /// query parser (which includes the configured analyzer and default fields).
    ///
    /// # Arguments
    ///
    /// * `query` - The query to resolve.
    ///
    /// # Errors
    ///
    /// Returns an error if the DSL string cannot be parsed.
    fn resolve_query(
        &self,
        query: crate::lexical::search::searcher::LexicalSearchQuery,
    ) -> Result<Box<dyn crate::lexical::query::Query>> {
        match query {
            crate::lexical::search::searcher::LexicalSearchQuery::Obj(q) => Ok(q),
            crate::lexical::search::searcher::LexicalSearchQuery::Dsl(dsl) => {
                let parser = self.lexical.query_parser()?;
                parser.parse(&dsl)
            }
        }
    }

    /// Resolve a [`SearchQuery`](self::search::SearchQuery) into internal
    /// search request types for the lexical and vector stores.
    ///
    /// This method converts the public query enum variants into the
    /// internal `LexicalSearchRequest` and `VectorSearchRequest` types,
    /// applying the relevant options.
    ///
    /// # Parameters
    ///
    /// * `query` - The search query to resolve.
    /// * `offset` - The pagination offset from the search request.
    /// * `limit` - The result limit from the search request.
    /// * `fusion_algorithm` - The caller-specified fusion algorithm, if any.
    /// * `lexical_options` - Lexical search options.
    /// * `vector_options` - Vector search options.
    ///
    /// # Errors
    ///
    /// Panics (via `unreachable!`) if called with `SearchQuery::Dsl`, which
    /// must be resolved before calling this method.
    #[allow(clippy::type_complexity)]
    fn resolve_search_query_from_parts(
        &self,
        query: self::search::SearchQuery,
        offset: usize,
        limit: usize,
        fusion_algorithm: Option<FusionAlgorithm>,
        lexical_options: &self::search::LexicalSearchOptions,
        vector_options: &self::search::VectorSearchOptions,
    ) -> Result<(
        Option<crate::lexical::search::searcher::LexicalSearchRequest>,
        Option<crate::vector::store::request::VectorSearchRequest>,
        Option<FusionAlgorithm>,
        self::search::HybridMode,
    )> {
        let fetch_count = offset.saturating_add(limit);

        match query {
            self::search::SearchQuery::Dsl(_) => {
                // DSL should be parsed by UnifiedQueryParser before calling this
                unreachable!("DSL should be resolved before resolve_search_query_from_parts")
            }
            self::search::SearchQuery::Lexical(lexical_query) => {
                let lex_req = crate::lexical::search::searcher::LexicalSearchRequest {
                    query: lexical_query,
                    params: crate::lexical::search::searcher::LexicalSearchParams {
                        limit: 0, // Controlled by engine
                        min_score: lexical_options.min_score,
                        load_documents: true,
                        timeout_ms: lexical_options.timeout_ms,
                        parallel: lexical_options.parallel,
                        sort_by: lexical_options.sort_by.clone(),
                    },
                    field_boosts: lexical_options.field_boosts.clone(),
                };
                Ok((Some(lex_req), None, None, self::search::HybridMode::Union))
            }
            self::search::SearchQuery::Vector(vector_query) => {
                let vec_req = self.build_vector_request(vector_query, vector_options, fetch_count);
                Ok((None, Some(vec_req), None, self::search::HybridMode::Union))
            }
            self::search::SearchQuery::Hybrid {
                lexical,
                vector,
                mode,
            } => {
                let lex_req = crate::lexical::search::searcher::LexicalSearchRequest {
                    query: lexical,
                    params: crate::lexical::search::searcher::LexicalSearchParams {
                        limit: 0, // Controlled by engine
                        min_score: lexical_options.min_score,
                        load_documents: true,
                        timeout_ms: lexical_options.timeout_ms,
                        parallel: lexical_options.parallel,
                        sort_by: lexical_options.sort_by.clone(),
                    },
                    field_boosts: lexical_options.field_boosts.clone(),
                };
                let vec_req = self.build_vector_request(vector, vector_options, fetch_count);
                let fusion = fusion_algorithm.or(Some(FusionAlgorithm::RRF { k: 60.0 }));
                Ok((Some(lex_req), Some(vec_req), fusion, mode))
            }
        }
    }

    /// Build a [`VectorSearchRequest`](crate::vector::store::request::VectorSearchRequest)
    /// from a [`VectorSearchQuery`](self::search::VectorSearchQuery) and options.
    ///
    /// # Parameters
    ///
    /// * `query` - The vector search query (payloads or pre-embedded vectors).
    /// * `opts` - Vector search options (score mode, min score).
    /// * `limit` - Maximum number of results to fetch.
    fn build_vector_request(
        &self,
        query: self::search::VectorSearchQuery,
        opts: &self::search::VectorSearchOptions,
        limit: usize,
    ) -> crate::vector::store::request::VectorSearchRequest {
        crate::vector::store::request::VectorSearchRequest {
            query,
            params: crate::vector::search::searcher::VectorSearchParams {
                fields: None,
                limit,
                score_mode: opts.score_mode,
                overfetch: 2.0,
                min_score: opts.min_score,
                allowed_ids: None,
                allowed_filter: None,
                rerank_factor: opts.rerank_factor,
                ef_search: opts.ef_search,
            },
        }
    }

    /// Get all documents (including chunks) by external ID.
    ///
    /// Only fields marked as stored in the schema are included in the
    /// returned documents. If no documents match the given ID, an empty
    /// `Vec` is returned (not an error).
    ///
    /// # Parameters
    ///
    /// - `id` - The external document identifier to look up.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal ID lookup or document retrieval fails.
    pub async fn get_documents(&self, id: &str) -> Result<Vec<Document>> {
        let doc_ids = self.lexical.find_doc_ids_by_term("_id", id)?;
        let mut docs = Vec::with_capacity(doc_ids.len());
        for doc_id in doc_ids {
            if let Some(doc) = self.get_document_by_internal_id(doc_id)? {
                docs.push(doc);
            }
        }
        Ok(docs)
    }

    /// Check if a field should be stored based on the schema.
    ///
    /// - `_id`: always stored (system field)
    /// - Lexical fields: stored only if `stored=true`
    /// - Vector fields: always stored
    /// - Unknown fields: not stored
    fn is_field_stored(&self, name: &str) -> bool {
        use crate::engine::schema::FieldOption;

        if name == "_id" {
            return true;
        }
        let schema = self.schema.read();
        if let Some(field_opt) = schema.fields.get(name) {
            match field_opt {
                FieldOption::Text(o) => o.stored,
                FieldOption::Integer(o) => o.stored,
                FieldOption::Float(o) => o.stored,
                FieldOption::Boolean(o) => o.stored,
                FieldOption::DateTime(o) => o.stored,
                FieldOption::Geo(o) => o.stored,
                FieldOption::Geo3d(o) => o.stored,
                FieldOption::Bytes(o) => o.stored,
                // Vector fields are always stored
                FieldOption::Hnsw(_) | FieldOption::Flat(_) | FieldOption::Ivf(_) => true,
            }
        } else {
            false
        }
    }

    /// Filter a document to only include fields that should be stored.
    ///
    /// The document log (WAL) stores ALL fields for recovery, but the
    /// document store only keeps stored fields to save space.
    fn filter_stored_fields(&self, doc: &Document) -> Document {
        let mut stored_doc = Document::new();
        for (name, val) in &doc.fields {
            if self.is_field_stored(name) {
                stored_doc.fields.insert(name.clone(), val.clone());
            }
        }
        stored_doc
    }

    /// Get a document by its internal ID (private helper).
    ///
    /// Retrieves from the document log and filters out non-stored fields.
    fn get_document_by_internal_id(&self, doc_id: u64) -> Result<Option<Document>> {
        let doc = self.log.get_document(doc_id)?;

        if let Some(doc) = doc {
            Ok(Some(self.filter_stored_fields(&doc)))
        } else {
            Ok(None)
        }
    }

    /// Batch-resolve external IDs and documents for multiple internal IDs.
    ///
    /// Fetches all documents in one pass through the document store,
    /// reducing per-document lock acquisition overhead.
    ///
    /// # Arguments
    ///
    /// * `internal_ids` - Slice of internal document IDs.
    ///
    /// # Returns
    ///
    /// A map from internal ID to `(external_id, Option<Document>)`.
    fn resolve_ids_and_documents_batch(
        &self,
        internal_ids: &[u64],
    ) -> Result<HashMap<u64, (String, Option<Document>)>> {
        let mut results = HashMap::with_capacity(internal_ids.len());
        for &id in internal_ids {
            if let Some(doc) = self.log.get_document(id)? {
                let external_id = doc
                    .fields
                    .get("_id")
                    .and_then(|v| v.as_text())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("unknown_{}", id));
                let filtered = self.filter_stored_fields(&doc);
                results.insert(id, (external_id, Some(filtered)));
            } else {
                results.insert(id, (format!("unknown_{}", id), None));
            }
        }
        Ok(results)
    }

    /// Split the unified schema into specialized configs.
    async fn split_schema(
        schema: &Schema,
        analyzer: Option<Arc<dyn Analyzer>>,
        embedder: Option<Arc<dyn Embedder>>,
        runtime_analyzers: &HashMap<String, Arc<dyn Analyzer>>,
    ) -> Result<(LexicalIndexConfig, VectorIndexConfig)> {
        // Construct Lexical Config
        let analyzer = match analyzer {
            Some(a) => a,
            None => Arc::new(StandardAnalyzer::new()?),
        };

        // If the user passed a PerFieldAnalyzer, clone it and ensure _id uses KeywordAnalyzer.
        // Otherwise, wrap the simple analyzer in a new PerFieldAnalyzer.
        let per_field_analyzer =
            if let Some(existing) = analyzer.as_any().downcast_ref::<PerFieldAnalyzer>() {
                let pfa = existing.clone();
                pfa.add_analyzer("_id", Arc::new(KeywordAnalyzer::new()));
                pfa
            } else {
                let pfa = PerFieldAnalyzer::new(analyzer);
                pfa.add_analyzer("_id", Arc::new(KeywordAnalyzer::new()));
                pfa
            };

        // Register per-field analyzers declared in the schema.
        // Resolution order: parameterized built-in → built-in name → custom
        // definition in schema.analyzers.
        for (name, field_option) in &schema.fields {
            if let schema::FieldOption::Text(text_opt) = field_option
                && let Some(spec) = &text_opt.analyzer
            {
                let field_analyzer =
                    crate::analysis::analyzer::registry::create_analyzer_from_spec(
                        spec,
                        &schema.analyzers,
                        runtime_analyzers,
                    )
                    .map_err(|e| {
                        crate::error::LaurusError::invalid_argument(format!(
                            "Failed to resolve analyzer for field '{name}': {e}"
                        ))
                    })?;
                per_field_analyzer.add_analyzer(name, field_analyzer);
            }
        }

        let mut lexical_builder =
            LexicalIndexConfig::builder().analyzer(Arc::new(per_field_analyzer));

        if !schema.default_fields.is_empty() {
            lexical_builder = lexical_builder.default_fields(schema.default_fields.clone());
        }

        for (name, field_option) in &schema.fields {
            if let Some(lexical_opt) = field_option.to_lexical() {
                lexical_builder = lexical_builder.add_field(name, lexical_opt);
            }
        }

        let lexical_config = lexical_builder.build();

        // Construct Vector Config — resolve embedder from schema if not explicitly provided.
        let embedder = if embedder.is_some() {
            embedder
        } else if !schema.embedders.is_empty() {
            // Build a PerFieldEmbedder from schema.embedders declarations.
            let mut embedder_cache: HashMap<String, Arc<dyn crate::embedding::embedder::Embedder>> =
                HashMap::new();
            let default_embedder: Arc<dyn crate::embedding::embedder::Embedder> =
                Arc::new(crate::embedding::precomputed::PrecomputedEmbedder::new());
            let per_field = crate::embedding::per_field::PerFieldEmbedder::new(default_embedder);

            for (name, field_option) in &schema.fields {
                if let Some(embedder_name) = field_option.embedder_name() {
                    let emb = if let Some(cached) = embedder_cache.get(embedder_name) {
                        cached.clone()
                    } else {
                        let def = schema.embedders.get(embedder_name).ok_or_else(|| {
                            crate::error::LaurusError::invalid_argument(format!(
                                "Unknown embedder '{embedder_name}' for field '{name}': \
                                 not defined in schema.embedders"
                            ))
                        })?;
                        let emb = crate::embedding::registry::create_embedder_from_definition(
                            embedder_name,
                            def,
                        )
                        .await?;
                        embedder_cache.insert(embedder_name.to_string(), emb.clone());
                        emb
                    };
                    per_field.add_embedder(name, emb);
                }
            }

            let emb: Arc<dyn crate::embedding::embedder::Embedder> = Arc::new(per_field);
            Some(emb)
        } else {
            None
        };

        let mut vector_builder = VectorIndexConfig::builder();
        if let Some(embedder) = &embedder {
            vector_builder = vector_builder.embedder(embedder.clone());
        }

        for (name, field_option) in &schema.fields {
            if let Some(vector_opt) = field_option.to_vector() {
                vector_builder = vector_builder.add_field(name, vector_opt)?;
            }
        }

        let vector_config = vector_builder.build()?;

        Ok((lexical_config, vector_config))
    }

    /// Warm the vector searcher so the first vector / hybrid query does not pay
    /// the searcher-construction and page-fault cost (Issue #677).
    ///
    /// Delegates to [`VectorStore::warmup`](crate::vector::VectorStore::warmup):
    /// it eagerly builds the cached searcher (loading the reader) and pre-faults
    /// on-disk vector data into the OS page cache where applicable (HNSW `Mmap`
    /// mode). Call once after building the engine, before serving traffic.
    /// Safe to call multiple times; lexical search needs no warming.
    ///
    /// # Errors
    ///
    /// Returns an error if building the vector searcher (reader load) fails.
    pub fn warmup(&self) -> Result<()> {
        self.vector.warmup()
    }

    /// Search the index.
    ///
    /// Supports three modes depending on how the
    /// [`SearchRequest`](self::search::SearchRequest) is configured:
    ///
    /// - **Unified query DSL** (via `query_dsl`): The query string is
    ///   parsed using [`UnifiedQueryParser`](self::query::UnifiedQueryParser)
    ///   to automatically extract lexical and/or vector components. This is
    ///   the recommended approach for external callers.
    /// - **Structured fields** (via `lexical_search_request` /
    ///   `vector_search_request`): Lower-level API for programmatic use.
    ///
    /// When `query_dsl` is set, it is parsed first, and the resulting
    /// lexical/vector components replace any explicitly set fields. The
    /// `fusion_algorithm`, `limit`, `offset`, and `filter_query` fields
    /// from the original request are preserved.
    ///
    /// After resolving the query source, the engine executes the
    /// appropriate search mode:
    ///
    /// - **Lexical only**: BM25-scored inverted index search.
    /// - **Vector only**: Nearest-neighbor vector search.
    /// - **Hybrid**: Both searches run and results are merged using the
    ///   configured `fusion_algorithm` (defaults to
    ///   [`RRF { k: 60.0 }`](FusionAlgorithm::RRF)).
    ///
    /// When a `filter_query` is present, it is evaluated first to determine
    /// the set of candidate documents. For lexical search, the filter is
    /// combined with the user query via a boolean `must` + `filter` clause.
    /// For vector search, the filter produces an `allowed_ids` list that
    /// restricts candidate scoring. If the filter matches zero documents,
    /// an empty result is returned immediately.
    ///
    /// When both lexical and vector search requests are present, both fetch
    /// limits are doubled (2x overfetch) to improve fusion quality.
    ///
    /// Results are paginated via `offset` and `limit` on the
    /// [`SearchRequest`](self::search::SearchRequest).
    ///
    /// # Parameters
    ///
    /// - `request` - The unified search request.
    ///
    /// # Errors
    ///
    /// Returns an error if the unified query parsing, filter query
    /// execution, lexical search, vector search, embedding, or document
    /// retrieval fails.
    pub async fn search(
        &self,
        request: self::search::SearchRequest,
    ) -> Result<Vec<self::search::SearchResult>> {
        // 0a. Resolve query to internal search components
        //
        // When the query is a DSL string, parse it with UnifiedQueryParser to
        // extract both lexical and vector components. For other variants,
        // construct the internal request types from the query + options.
        //
        // Destructure the request upfront so that `query` can be moved
        // independently while the remaining fields stay available.
        let self::search::SearchRequest {
            query: request_query,
            limit: request_limit,
            offset: request_offset,
            fusion_algorithm: request_fusion,
            filter_query: request_filter,
            lexical_options,
            vector_options,
        } = request;

        let (lexical_search_request, vector_search_request, fusion_algorithm, hybrid_mode) =
            match request_query {
                self::search::SearchQuery::Dsl(ref dsl) => {
                    let parser = self.unified_query_parser()?;
                    let parser = if let Some(fusion) = request_fusion {
                        parser.with_fusion(fusion)
                    } else {
                        parser
                    };
                    let parsed = parser.parse(dsl).await?;
                    // UnifiedQueryParser now returns Lexical/Vector/Hybrid variants
                    self.resolve_search_query_from_parts(
                        parsed.query,
                        request_offset,
                        request_limit,
                        request_fusion,
                        &lexical_options,
                        &vector_options,
                    )?
                }
                other => self.resolve_search_query_from_parts(
                    other,
                    request_offset,
                    request_limit,
                    request_fusion,
                    &lexical_options,
                    &vector_options,
                )?,
            };

        // 0b. Pre-process Filter
        let (allowed_filter, lexical_query_override) = if let Some(filter_query) = &request_filter {
            // Evaluate the filter through the snapshot-scoped query/filter cache
            // (Issue #578): a repeated filter is served as a cached doc-id set
            // instead of re-walking posting lists. Unlike the previous path,
            // this is not capped at 1M matches. The resulting `Arc<RoaringTreemap>`
            // is handed to the vector side as-is (Issue #739) — no `Vec<u64>` /
            // `AHashSet` round trip.
            let allowed = self.lexical.matching_doc_ids(filter_query.clone_box())?;

            if allowed.is_empty() {
                return Ok(Vec::new());
            }

            let new_lexical_query: Option<Box<dyn crate::lexical::query::Query>> =
                if let Some(lex_req) = &lexical_search_request {
                    use crate::lexical::query::boolean::BooleanQueryBuilder;
                    let user_query = self.resolve_query(lex_req.query.clone())?;
                    let bool_query = BooleanQueryBuilder::new()
                        .must(user_query)
                        .filter(filter_query.clone_box())
                        .build();
                    Some(Box::new(bool_query))
                } else {
                    None
                };

            (Some(allowed), new_lexical_query)
        } else {
            (None, None)
        };

        // 1. Execute Lexical Search
        let mut lexical_query_to_use = if lexical_query_override.is_some() {
            lexical_query_override
        } else if let Some(lex_req) = &lexical_search_request {
            Some(self.resolve_query(lex_req.query.clone())?)
        } else {
            None
        };

        if let Some(query) = &mut lexical_query_to_use
            && let Some(lex_req) = &lexical_search_request
            && !lex_req.field_boosts.is_empty()
        {
            query.apply_field_boosts(&lex_req.field_boosts);
        }

        let fetch_count = request_offset.saturating_add(request_limit);

        // Build the lexical request; the search itself runs in parallel below.
        let lex_req = if let Some(query) = &lexical_query_to_use {
            let q = query.clone_box();
            let overfetch_limit = if vector_search_request.is_some() {
                fetch_count.saturating_mul(2)
            } else {
                fetch_count
            };
            Some(
                crate::lexical::search::searcher::LexicalSearchRequest::new(q)
                    .limit(overfetch_limit)
                    .load_documents(false),
            )
        } else {
            None
        };

        // 2. Build the vector request — including the async payload embedding,
        // which must complete before the (synchronous) search runs in parallel
        // below.
        let vec_req = if let Some(vector_req) = &vector_search_request {
            let mut vreq = vector_req.clone();
            if lexical_search_request.is_some() && vreq.params.limit < fetch_count.saturating_mul(2)
            {
                vreq.params.limit = fetch_count.saturating_mul(2);
            }
            if let Some(filter) = &allowed_filter {
                vreq.params.allowed_filter = Some(filter.clone());
            }
            // Embed Payloads into Vectors before searching.
            // NOTE: When using VectorQueryParser, query is already Vectors
            // at parse time, so this block is skipped. This fallback remains for
            // VectorSearchRequestBuilder users who populate Payloads directly.
            if let crate::vector::search::searcher::VectorSearchQuery::Payloads(ref payloads) =
                vreq.query
            {
                use crate::data::DataValue;
                use crate::embedding::embedder::EmbedInput;
                use crate::vector::store::request::QueryVector;

                // Owned payload data for the embeddable (Text / Bytes) payloads,
                // keeping each one's field and weight. Non-text / non-bytes
                // payloads are skipped, as before. Owned buffers must outlive the
                // borrowed `EmbedInput`s handed to the batch call below.
                enum Owned {
                    Text(String),
                    Bytes(Vec<u8>, Option<String>),
                }
                let mut owned: Vec<(String, f32, Owned)> = Vec::new();
                for payload in payloads {
                    let data = match &payload.payload {
                        DataValue::Text(t) => Owned::Text(t.clone()),
                        DataValue::Bytes(b, m) => Owned::Bytes(b.clone(), m.clone()),
                        _ => continue,
                    };
                    owned.push((payload.field.clone(), payload.weight, data));
                }

                // Embed every payload in one batch (Issue #671) so a
                // batch-capable embedder pays one round trip instead of one per
                // payload, while preserving cache and per-field routing.
                let items: Vec<(String, EmbedInput<'_>)> = owned
                    .iter()
                    .map(|(field, _, data)| {
                        let input = match data {
                            Owned::Text(t) => EmbedInput::Text(t),
                            Owned::Bytes(b, m) => EmbedInput::Bytes(b, m.as_deref()),
                        };
                        (field.clone(), input)
                    })
                    .collect();
                let embedder = self.vector.embedder();
                let vectors =
                    embed_batch_with_cache(self.embedding_cache.as_ref(), &embedder, &items)
                        .await?;

                let query_vectors: Vec<QueryVector> = owned
                    .iter()
                    .zip(vectors)
                    .map(|((field, weight, _), vector)| QueryVector {
                        vector,
                        weight: *weight,
                        fields: Some(vec![field.clone()]),
                    })
                    .collect();
                vreq.query =
                    crate::vector::search::searcher::VectorSearchQuery::Vectors(query_vectors);
            }
            Some(vreq)
        } else {
            None
        };

        // Run the independent lexical and vector searches (#659). On native
        // builds both synchronous searches overlap via `rayon::join`, so the
        // hybrid latency drops from `lex + vec` toward `max(lex, vec)`. The
        // closures take disjoint immutable borrows of `self.lexical` /
        // `self.vector` plus the moved requests, so they are `Send`. On wasm32
        // (no rayon) they run sequentially. Fusion below is order-independent,
        // so the result set is identical either way.
        let run_lexical = || lex_req.map(|r| self.lexical.search(r)).transpose();
        let run_vector = || vec_req.map(|r| self.vector.search(r)).transpose();
        #[cfg(feature = "native")]
        let (lex_res, vec_res) = rayon::join(run_lexical, run_vector);
        #[cfg(not(feature = "native"))]
        let (lex_res, vec_res) = (run_lexical(), run_vector());
        let lexical_hits = lex_res?.map(|r| r.hits).unwrap_or_default();
        let vector_hits = vec_res?.map(|r| r.hits).unwrap_or_default();

        // 3. Fusion
        if lexical_search_request.is_some() && vector_search_request.is_some() {
            let algorithm = fusion_algorithm.unwrap_or(FusionAlgorithm::RRF { k: 60.0 });
            let mut results = self.fuse_results(
                lexical_hits,
                vector_hits,
                algorithm,
                hybrid_mode,
                fetch_count,
            )?;
            if request_offset > 0 {
                results = results.into_iter().skip(request_offset).collect();
            }
            results.truncate(request_limit);
            Ok(results)
        } else if !vector_hits.is_empty() {
            // Only vector results — batch-resolve external IDs and documents.
            let paginated: Vec<_> = vector_hits
                .into_iter()
                .skip(request_offset)
                .take(request_limit)
                .collect();
            let ids: Vec<u64> = paginated.iter().map(|h| h.doc_id).collect();
            let resolved = self.resolve_ids_and_documents_batch(&ids)?;
            let mut results = Vec::with_capacity(paginated.len());
            for hit in paginated {
                if let Some((external_id, document)) = resolved.get(&hit.doc_id) {
                    results.push(SearchResult {
                        id: external_id.clone(),
                        score: hit.score,
                        document: document.clone(),
                    });
                }
            }
            Ok(results)
        } else {
            // Only lexical results (or both empty)
            let paginated: Vec<_> = lexical_hits
                .into_iter()
                .skip(request_offset)
                .take(request_limit)
                .collect();
            let ids: Vec<u64> = paginated.iter().map(|h| h.doc_id).collect();
            let resolved = self.resolve_ids_and_documents_batch(&ids)?;
            let mut results = Vec::with_capacity(paginated.len());
            for hit in paginated {
                if let Some((external_id, document)) = resolved.get(&hit.doc_id) {
                    results.push(SearchResult {
                        id: external_id.clone(),
                        score: hit.score,
                        document: document.clone(),
                    });
                }
            }
            Ok(results)
        }
    }

    /// Combine results from lexical and vector engines.
    fn fuse_results(
        &self,
        lexical_hits: Vec<crate::lexical::query::SearchHit>,
        vector_hits: Vec<crate::vector::store::response::VectorHit>,
        fusion: FusionAlgorithm,
        mode: self::search::HybridMode,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        // Collect doc_id sets upfront for intersection filtering.
        let lexical_ids: HashSet<u64> = lexical_hits.iter().map(|h| h.doc_id).collect();
        let vector_ids: HashSet<u64> = vector_hits.iter().map(|h| h.doc_id).collect();

        let mut fused_scores: HashMap<u64, (f32, Option<crate::data::Document>)> = HashMap::new();

        match fusion {
            FusionAlgorithm::RRF { k } => {
                for (rank, hit) in lexical_hits.into_iter().enumerate() {
                    let rrf_score = 1.0 / (k + (rank + 1) as f64);
                    let entry = fused_scores
                        .entry(hit.doc_id)
                        .or_insert((0.0, hit.document));
                    entry.0 += rrf_score as f32;
                }
                for (rank, hit) in vector_hits.into_iter().enumerate() {
                    let rrf_score = 1.0 / (k + (rank + 1) as f64);
                    let entry = fused_scores.entry(hit.doc_id).or_insert((0.0, None));
                    entry.0 += rrf_score as f32;
                }
            }
            FusionAlgorithm::WeightedSum {
                lexical_weight,
                vector_weight,
            } => {
                let lexical_min = lexical_hits
                    .iter()
                    .map(|h| h.score)
                    .fold(f32::INFINITY, f32::min);
                let lexical_max = lexical_hits
                    .iter()
                    .map(|h| h.score)
                    .fold(f32::NEG_INFINITY, f32::max);

                for hit in lexical_hits {
                    let norm_score = if lexical_max > lexical_min {
                        (hit.score - lexical_min) / (lexical_max - lexical_min)
                    } else {
                        1.0
                    };
                    let entry = fused_scores
                        .entry(hit.doc_id)
                        .or_insert((0.0, hit.document));
                    entry.0 += norm_score * lexical_weight;
                }

                let vector_min = vector_hits
                    .iter()
                    .map(|h| h.score)
                    .fold(f32::INFINITY, f32::min);
                let vector_max = vector_hits
                    .iter()
                    .map(|h| h.score)
                    .fold(f32::NEG_INFINITY, f32::max);

                for hit in vector_hits {
                    let norm_score = if vector_max > vector_min {
                        (hit.score - vector_min) / (vector_max - vector_min)
                    } else {
                        1.0
                    };
                    let entry = fused_scores.entry(hit.doc_id).or_insert((0.0, None));
                    entry.0 += norm_score * vector_weight;
                }
            }
        }

        // Intersection mode: keep only documents appearing in BOTH result sets.
        if mode == self::search::HybridMode::Intersection {
            fused_scores.retain(|id, _| lexical_ids.contains(id) && vector_ids.contains(id));
        }

        let mut intermediate: Vec<(u64, f32, Option<crate::data::Document>)> = fused_scores
            .into_iter()
            .map(|(doc_id, (score, document))| (doc_id, score, document))
            .collect();

        // Sort by fused score descending
        intermediate.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Limit results
        if intermediate.len() > limit {
            intermediate.truncate(limit);
        }

        // Batch-resolve external IDs and fill missing documents.
        // Collect IDs that need resolution (either missing external ID or
        // missing document).
        let ids_to_resolve: Vec<u64> = intermediate.iter().map(|(doc_id, _, _)| *doc_id).collect();
        let resolved = self.resolve_ids_and_documents_batch(&ids_to_resolve)?;

        let mut results = Vec::with_capacity(intermediate.len());
        for (doc_id, score, document) in intermediate {
            if let Some((external_id, resolved_doc)) = resolved.get(&doc_id) {
                // Prefer the document already fetched by the lexical search;
                // fall back to the batch-resolved copy.
                let final_doc = if document.is_some() {
                    document
                } else {
                    resolved_doc.clone()
                };
                results.push(SearchResult {
                    id: external_id.clone(),
                    score,
                    document: final_doc,
                });
            }
        }

        Ok(results)
    }

    /// Execute multiple independent search requests in parallel.
    ///
    /// Batched form of [`Self::search`] that runs each request
    /// concurrently on the tokio runtime via
    /// [`futures::future::try_join_all`]. Internal vector-search work
    /// additionally parallelises per-request via rayon (Phase 1 of
    /// issue [#648](https://github.com/mosuka/laurus/issues/648), PR
    /// [#711](https://github.com/mosuka/laurus/pull/711)), so a batch
    /// of `B` requests benefits from two-level parallelism: `B`
    /// requests in parallel on tokio, each request's multi-vector
    /// path in parallel on rayon.
    ///
    /// External callers (gRPC service, REST gateway, language
    /// bindings) invoke this method to amortise IPC and serialisation
    /// overhead across multiple queries, in addition to the per-query
    /// parallelism already provided by Phases 1 and 2 of
    /// [#648](https://github.com/mosuka/laurus/issues/648).
    ///
    /// # Parameters
    ///
    /// - `requests` - The list of independent search requests. Order
    ///   is preserved in the output.
    ///
    /// # Returns
    ///
    /// A `Vec<Vec<SearchResult>>` where `results[i]` is the result of
    /// `requests[i]`. Empty input returns an empty `Vec` without
    /// invoking [`Self::search`] at all.
    ///
    /// # Errors
    ///
    /// Short-circuits with the first error encountered; the other
    /// in-flight requests are dropped per
    /// [`futures::future::try_join_all`] semantics.
    ///
    /// Issue [#715](https://github.com/mosuka/laurus/issues/715)
    /// (Phase 3 prerequisite of
    /// [#648](https://github.com/mosuka/laurus/issues/648)).
    pub async fn search_batch(
        &self,
        requests: Vec<self::search::SearchRequest>,
    ) -> Result<Vec<Vec<self::search::SearchResult>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        futures::future::try_join_all(requests.into_iter().map(|r| self.search(r))).await
    }
}

/// Builder for constructing an [`Engine`] with custom configuration.
///
/// Use this when you need to specify a custom text analyzer or embedding
/// model. For simple cases with default settings (StandardAnalyzer, no
/// embedder), use [`Engine::new`] directly.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
///
/// let schema = Schema::builder()
///     .add_field("content", FieldOption::Text(TextOption::default()))
///     .add_field("content_vec", FieldOption::Flat(FlatOption { dimension: 384, ..Default::default() }))
///     .build();
///
/// let engine = Engine::builder(storage, schema)
///     .analyzer(Arc::new(StandardAnalyzer::default()))
///     .embedder(Arc::new(MyEmbedder))
///     .build()
///     .await?;
/// ```
pub struct EngineBuilder {
    storage: Arc<dyn Storage>,
    schema: Schema,
    analyzer: Option<Arc<dyn Analyzer>>,
    embedder: Option<Arc<dyn Embedder>>,
    runtime_analyzers: HashMap<String, Arc<dyn Analyzer>>,
    embedding_cache_capacity: Option<usize>,
    wal_sync_policy: WalSyncPolicy,
}

impl EngineBuilder {
    /// Create a new builder with the given storage and schema.
    pub fn new(storage: Arc<dyn Storage>, schema: Schema) -> Self {
        Self {
            storage,
            schema,
            analyzer: None,
            embedder: None,
            runtime_analyzers: HashMap::new(),
            embedding_cache_capacity: None,
            wal_sync_policy: WalSyncPolicy::default(),
        }
    }

    /// Set the analyzer for text fields.
    ///
    /// Both simple analyzers (e.g., [`StandardAnalyzer`]) and [`PerFieldAnalyzer`] are
    /// supported. When a `PerFieldAnalyzer` is passed, it is used directly (with `_id`
    /// automatically set to `KeywordAnalyzer` if not already configured).
    ///
    /// If not set, [`StandardAnalyzer`] is used as the default.
    pub fn analyzer(mut self, analyzer: Arc<dyn Analyzer>) -> Self {
        self.analyzer = Some(analyzer);
        self
    }

    /// Register a pre-constructed analyzer under a name, resolved at
    /// build time before built-in names and `schema.analyzers`.
    ///
    /// Useful when an analyzer cannot be expressed as a serializable
    /// [`crate::AnalyzerSpec`] — for example, a Japanese analyzer
    /// constructed from raw dictionary bytes loaded from OPFS in a
    /// browser WASM context. Schema text fields can refer to the
    /// runtime-registered analyzer by its `Named` form.
    ///
    /// # Arguments
    ///
    /// * `name` - The name used in `TextOption.analyzer` (e.g.
    ///   `"ja-ipadic"`).
    /// * `analyzer` - The pre-built analyzer instance.
    pub fn register_runtime_analyzer(
        mut self,
        name: impl Into<String>,
        analyzer: Arc<dyn Analyzer>,
    ) -> Self {
        self.runtime_analyzers.insert(name.into(), analyzer);
        self
    }

    /// Set the embedder for vector fields.
    ///
    /// Both simple embedders and [`PerFieldEmbedder`](crate::embedding::per_field::PerFieldEmbedder)
    /// are supported. When a `PerFieldEmbedder` is passed, each vector field will use
    /// the embedder registered for that field name, falling back to the default.
    ///
    /// If not set, no embedder is configured.
    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Enable an LRU cache for query-time embeddings, holding up to
    /// `capacity` entries (Issue #678).
    ///
    /// When set, identical query payloads embedded by the same field /
    /// embedder are produced only once and reused on subsequent searches,
    /// avoiding repeated model inference (or network round trips for remote
    /// embedders). Disabled by default; `capacity = 0` is treated as
    /// disabled.
    ///
    /// The cache only affects query-time embedding in [`Engine::search`];
    /// document-ingestion embedding is unaffected.
    pub fn embedding_cache_capacity(mut self, capacity: usize) -> Self {
        self.embedding_cache_capacity = Some(capacity);
        self
    }

    /// Set the WAL durability policy (Issue #542).
    ///
    /// Defaults to [`WalSyncPolicy::PerRecord`], where every `add`/`delete`
    /// fsyncs the WAL before returning, so a successful write can never be lost
    /// to a crash. Switch to [`WalSyncPolicy::Group`] to defer and batch the
    /// fsync — much higher ingest throughput at the cost of losing up to the
    /// last unsynced batch on a crash. [`Engine::commit`] is a hard durability
    /// barrier under both policies, and [`Engine::flush_wal`] forces a flush on
    /// demand.
    ///
    /// # Arguments
    ///
    /// * `policy` - The durability policy. Use
    ///   [`WalSyncPolicy::group_with_defaults`] for group commit with the
    ///   default batch thresholds.
    pub fn wal_sync_policy(mut self, policy: WalSyncPolicy) -> Self {
        self.wal_sync_policy = policy;
        self
    }

    /// Build the [`Engine`].
    ///
    /// Creates the lexical store, vector store, and document log (WAL),
    /// then runs WAL recovery to replay any uncommitted changes from a
    /// previous session.
    ///
    /// # Errors
    ///
    /// Returns an error if storage initialization, index creation, WAL
    /// opening, or recovery replay fails.
    pub async fn build(self) -> Result<Engine> {
        let (lexical_config, vector_config) = Engine::split_schema(
            &self.schema,
            self.analyzer,
            self.embedder,
            &self.runtime_analyzers,
        )
        .await?;

        let lexical_storage = Arc::new(PrefixedStorage::new("lexical", self.storage.clone()));
        let vector_storage = Arc::new(PrefixedStorage::new("vector", self.storage.clone()));
        let document_storage: Arc<dyn Storage> =
            Arc::new(PrefixedStorage::new("documents", self.storage.clone()));

        let lexical = LexicalStore::new(lexical_storage, lexical_config)?;
        let vector = VectorStore::new(vector_storage, vector_config)?;

        let log = Arc::new(DocumentLog::with_sync_policy(
            self.storage,
            "engine.wal",
            document_storage,
            self.wal_sync_policy,
        )?);

        let embedding_cache = self
            .embedding_cache_capacity
            .and_then(NonZeroUsize::new)
            .map(|cap| Arc::new(EmbeddingCache::new(cap)));

        // Start the periodic WAL flush timer when the policy is a group commit
        // with an interval. Native only; on wasm32 the interval is ignored.
        #[cfg(not(target_arch = "wasm32"))]
        let wal_flush_timer = match self.wal_sync_policy.flush_interval() {
            Some(interval) => Some(WalFlushTimer::spawn(Arc::clone(&log), interval)?),
            None => None,
        };

        let engine = Engine {
            schema: RwLock::new(self.schema),
            lexical,
            vector,
            log,
            runtime_analyzers: self.runtime_analyzers,
            embedding_cache,
            #[cfg(not(target_arch = "wasm32"))]
            _wal_flush_timer: wal_flush_timer,
        };

        engine.recover().await?;

        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::per_field::PerFieldEmbedder;
    use crate::embedding::precomputed::PrecomputedEmbedder;
    use crate::storage::memory::MemoryStorage;

    #[tokio::test]
    async fn test_accepts_per_field_analyzer() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::default()));

        let result = Engine::builder(storage, schema)
            .analyzer(Arc::new(per_field))
            .build()
            .await;

        assert!(result.is_ok(), "Should accept PerFieldAnalyzer");
    }

    #[tokio::test]
    async fn test_accepts_per_field_embedder() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let dummy_embedder = Arc::new(PrecomputedEmbedder::new());
        let per_field = PerFieldEmbedder::new(dummy_embedder);

        let result = Engine::builder(storage, schema)
            .embedder(Arc::new(per_field))
            .build()
            .await;

        assert!(result.is_ok(), "Should accept PerFieldEmbedder");
    }

    #[tokio::test]
    async fn test_accepts_simple_analyzer() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let result = Engine::builder(storage, schema)
            .analyzer(Arc::new(StandardAnalyzer::default()))
            .build()
            .await;

        assert!(result.is_ok(), "Should accept StandardAnalyzer");
    }

    #[tokio::test]
    async fn test_accepts_simple_embedder() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let dummy_embedder = Arc::new(PrecomputedEmbedder::new());

        let result = Engine::builder(storage, schema)
            .embedder(dummy_embedder)
            .build()
            .await;

        assert!(result.is_ok(), "Should accept simple embedder");
    }

    #[tokio::test]
    async fn test_schema_per_field_analyzer() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        // "category" uses keyword analyzer (no tokenization).
        // "body" uses default (standard) analyzer.
        let schema = Schema::builder()
            .add_field(
                "category",
                FieldOption::Text(TextOption::default().analyzer("keyword")),
            )
            .add_field("body", FieldOption::Text(TextOption::default()))
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        let mut doc = crate::data::Document::new();
        doc.fields
            .insert("category".into(), DataValue::Text("Rust Lang".into()));
        doc.fields.insert(
            "body".into(),
            DataValue::Text("Rust is a systems programming language".into()),
        );
        engine.put_document("doc1", doc).await.unwrap();
        engine.commit().await.unwrap();

        // "Rust Lang" as keyword — exact match required.
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("category:\"Rust Lang\""))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert_eq!(
            results.len(),
            1,
            "Keyword analyzer should match exact phrase"
        );

        // Partial token "Rust" should NOT match keyword-analyzed category.
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("category:Rust"))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert!(
            results.is_empty(),
            "Keyword analyzer should not match partial tokens"
        );

        // Standard-analyzed "body" field should match single token "rust".
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("body:rust"))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert_eq!(
            results.len(),
            1,
            "Standard analyzer should tokenize and match"
        );
    }

    #[tokio::test]
    async fn test_custom_analyzer_definition_in_schema() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::engine::schema::analyzer::{
            AnalyzerDefinition, CharFilterConfig, TokenFilterConfig, TokenizerConfig,
        };
        use crate::lexical::core::field::TextOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        // Define a custom analyzer: whitespace + NFKC normalization + lowercase.
        let schema = Schema::builder()
            .add_analyzer(
                "my_custom",
                AnalyzerDefinition {
                    char_filters: vec![CharFilterConfig::UnicodeNormalization {
                        form: "nfkc".into(),
                    }],
                    tokenizer: TokenizerConfig::Whitespace,
                    token_filters: vec![TokenFilterConfig::Lowercase],
                },
            )
            .add_field(
                "content",
                FieldOption::Text(TextOption::default().analyzer("my_custom")),
            )
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        let mut doc = crate::data::Document::new();
        // Fullwidth "ＨＥＬＬＯ" should be normalized to "HELLO", then lowercased.
        doc.fields.insert(
            "content".into(),
            DataValue::Text("\u{ff28}\u{ff25}\u{ff2c}\u{ff2c}\u{ff2f} world".into()),
        );
        engine.put_document("doc1", doc).await.unwrap();
        engine.commit().await.unwrap();

        // Search for "hello" should match (NFKC + lowercase).
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("content:hello"))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert_eq!(
            results.len(),
            1,
            "Custom analyzer (NFKC + lowercase) should match normalized text"
        );
    }

    #[tokio::test]
    async fn test_add_lexical_field() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        // Start with a schema containing only "title".
        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        // Dynamically add a "category" field.
        let updated = engine
            .add_field(
                "category",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .await
            .unwrap();

        assert!(updated.fields.contains_key("category"));
        assert!(updated.fields.contains_key("title"));

        // Index a document that uses the new field.
        engine
            .add_document(
                "doc1",
                Document::builder()
                    .add_text("title", "Rust Programming")
                    .add_text("category", "programming")
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();

        // Search on the dynamically added field.
        use crate::lexical::search::searcher::LexicalSearchQuery;
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("category:programming"))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert_eq!(
            results.len(),
            1,
            "Should find doc via dynamically added field"
        );
    }

    /// An engine built with [`WalSyncPolicy::Group`] plumbs the policy through to
    /// the WAL, accepts [`Engine::flush_wal`] as an on-demand durability barrier,
    /// and commits searchable results (Issue #542, Phase 4).
    #[tokio::test]
    async fn test_group_commit_policy_is_wired_and_searchable() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let engine = Engine::builder(storage, schema)
            .wal_sync_policy(WalSyncPolicy::group_with_defaults())
            .build()
            .await
            .unwrap();

        engine
            .add_document(
                "doc1",
                Document::builder()
                    .add_text("title", "group commit")
                    .build(),
            )
            .await
            .unwrap();

        // Under the group policy the append defers its fsync; an on-demand
        // flush_wal (no full commit) must succeed as a durability barrier.
        engine.flush_wal().unwrap();

        engine.commit().await.unwrap();

        use crate::lexical::search::searcher::LexicalSearchQuery;
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("title:group"))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert_eq!(results.len(), 1, "group-committed doc must be searchable");
    }

    /// The background flush timer forces a dirty (deferred) WAL writer durable
    /// within its interval, then stops cleanly on drop (Issue #542, Phase 4b).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn wal_flush_timer_flushes_dirty_writer_and_stops_on_drop() {
        use std::time::Duration;

        let wal_storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let doc_storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        // Thresholds high enough that a single append never trips them, so only
        // the timer can flush the writer.
        let log = Arc::new(
            DocumentLog::with_sync_policy(
                wal_storage,
                "engine.wal",
                doc_storage,
                WalSyncPolicy::Group {
                    max_records: usize::MAX,
                    max_bytes: usize::MAX,
                    max_interval: Some(Duration::from_millis(20)),
                },
            )
            .unwrap(),
        );

        log.append("doc1", Document::builder().add_text("title", "x").build())
            .unwrap();
        assert!(
            log.wal_is_dirty(),
            "the deferred append leaves the writer dirty"
        );

        let timer = WalFlushTimer::spawn(Arc::clone(&log), Duration::from_millis(20)).unwrap();

        // Poll up to ~2s for the timer to flush the writer.
        let mut flushed = false;
        for _ in 0..200 {
            if !log.wal_is_dirty() {
                flushed = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            flushed,
            "the timer should flush the dirty writer within its interval"
        );

        // Dropping the timer must return promptly (clean shutdown / thread join).
        drop(timer);
    }

    /// An engine built with a group-commit policy that includes a flush interval
    /// starts and stops its background timer without hanging on drop (Issue
    /// #542, Phase 4b).
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_group_commit_with_interval_builds_and_drops_cleanly() {
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let engine = Engine::builder(storage, schema)
            .wal_sync_policy(WalSyncPolicy::group_with_interval(Duration::from_millis(
                20,
            )))
            .build()
            .await
            .unwrap();

        engine
            .add_document(
                "doc1",
                Document::builder().add_text("title", "timer").build(),
            )
            .await
            .unwrap();

        // Dropping the engine must stop the background timer without hanging.
        drop(engine);
    }

    #[tokio::test]
    async fn test_add_field_duplicate_rejected() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        // Adding a field with the same name should fail.
        let result = engine
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .await;
        assert!(result.is_err(), "Duplicate field should be rejected");
    }

    #[tokio::test]
    async fn test_add_vector_field() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let dummy_embedder = Arc::new(PrecomputedEmbedder::new());
        let per_field = PerFieldEmbedder::new(dummy_embedder);

        let engine = Engine::builder(storage, schema)
            .embedder(Arc::new(per_field))
            .build()
            .await
            .unwrap();

        // Dynamically add a vector field with dimension 128 (matching PrecomputedEmbedder default).
        let updated = engine
            .add_field(
                "embedding",
                schema::FieldOption::Flat(
                    crate::vector::core::field::FlatOption::default().dimension(128),
                ),
            )
            .await
            .unwrap();

        assert!(updated.fields.contains_key("embedding"));

        // Index a document with the vector field.
        let vec_data: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        engine
            .add_document(
                "doc1",
                Document::builder()
                    .add_text("title", "Hello")
                    .add_vector("embedding", vec_data)
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();

        // Verify document was indexed successfully.
        let docs = engine.get_documents("doc1").await.unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[tokio::test]
    async fn test_schema_returns_current_state() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let engine = Engine::new(storage, schema).await.unwrap();

        // Initially empty (no user fields).
        assert!(engine.schema().fields.is_empty());

        // Add a field.
        engine
            .add_field(
                "body",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .await
            .unwrap();

        // schema() should reflect the addition.
        let current = engine.schema();
        assert!(current.fields.contains_key("body"));
    }

    #[tokio::test]
    async fn test_delete_lexical_field() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        // Dynamically add a "category" field, then delete it.
        engine
            .add_field(
                "category",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .await
            .unwrap();
        assert!(engine.schema().fields.contains_key("category"));

        let updated = engine.delete_field("category").await.unwrap();
        assert!(!updated.fields.contains_key("category"));
        assert!(updated.fields.contains_key("title"));
    }

    #[tokio::test]
    async fn test_delete_field_removes_from_default_fields() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .add_default_field("title")
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        let updated = engine.delete_field("title").await.unwrap();
        assert!(!updated.fields.contains_key("title"));
        assert!(!updated.default_fields.contains(&"title".to_string()));
    }

    #[tokio::test]
    async fn test_delete_field_nonexistent_rejected() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let engine = Engine::new(storage, schema).await.unwrap();

        let result = engine.delete_field("nonexistent").await;
        assert!(result.is_err(), "Deleting a nonexistent field should fail");
    }

    #[tokio::test]
    async fn test_delete_vector_field() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let dummy_embedder = Arc::new(PrecomputedEmbedder::new());
        let per_field = PerFieldEmbedder::new(dummy_embedder);

        let engine = Engine::builder(storage, schema)
            .embedder(Arc::new(per_field))
            .build()
            .await
            .unwrap();

        // Add then delete a vector field.
        engine
            .add_field(
                "embedding",
                schema::FieldOption::Hnsw(crate::vector::core::field::HnswOption {
                    dimension: 4,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        assert!(engine.schema().fields.contains_key("embedding"));

        let updated = engine.delete_field("embedding").await.unwrap();
        assert!(!updated.fields.contains_key("embedding"));
    }

    /// Regression test for the InvertedIndexWriter `delete_document` bug
    /// where the in-memory inverted index and DocValues were not rebuilt
    /// after a buffered doc was retained out, leaving ghost postings that
    /// survived into the next flushed segment.
    ///
    /// Symptom in callers: `put_document(id, doc1)` then
    /// `put_document(id, doc2)` in the same uncommitted batch ended up
    /// with two live docs sharing the same external `_id` after commit,
    /// and `get_documents(id)` / `engine.search(`_id:id`)` returned both.
    ///
    /// Fix lives in `lexical/index/inverted/writer.rs::delete_document`:
    /// it now calls `remove_pending_document` (which rebuilds the
    /// in-memory inverted index and DocValues) instead of doing a bare
    /// `buffered_docs.retain`.
    #[tokio::test]
    async fn test_put_document_replaces_within_uncommitted_batch() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        let mut doc1 = crate::data::Document::new();
        doc1.fields
            .insert("title".into(), DataValue::Text("first".into()));
        engine.put_document("X", doc1).await.unwrap();

        // Second put for the same external id, BEFORE commit. The first
        // doc must be fully replaced — not appended.
        let mut doc2 = crate::data::Document::new();
        doc2.fields
            .insert("title".into(), DataValue::Text("second".into()));
        engine.put_document("X", doc2).await.unwrap();

        engine.commit().await.unwrap();

        let docs = engine.get_documents("X").await.unwrap();
        assert_eq!(
            docs.len(),
            1,
            "exactly one doc should exist for id=X after two puts in the \
             same uncommitted batch (got {} docs: {:?})",
            docs.len(),
            docs.iter()
                .map(|d| d.fields.get("title").cloned())
                .collect::<Vec<_>>(),
        );

        let title = docs[0]
            .fields
            .get("title")
            .and_then(|v| v.as_text())
            .map(String::from);
        assert_eq!(
            title.as_deref(),
            Some("second"),
            "the surviving doc must be the latest put"
        );

        let stats = engine.stats().unwrap();
        assert_eq!(
            stats.document_count, 1,
            "engine.stats().document_count must agree with get_documents",
        );
    }

    /// Regression test for the same bug under a heavier put-pattern that
    /// mirrors the `laurus-wasm/examples/geo3d/` workload: put many docs,
    /// many of them carrying the same external id, in a single
    /// uncommitted batch.
    ///
    /// Before the fix, the engine reported `document_count` equal to the
    /// raw put count (with duplicates) instead of the unique-id count.
    #[tokio::test]
    async fn test_put_document_dedupes_duplicate_ids_in_batch() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        // 10 unique ids, each put 3 times in a row before commit.
        for i in 0..10 {
            for rev in 0..3 {
                let mut doc = crate::data::Document::new();
                doc.fields
                    .insert("title".into(), DataValue::Text(format!("id{i}-rev{rev}")));
                engine.put_document(&format!("id{i}"), doc).await.unwrap();
            }
        }

        engine.commit().await.unwrap();

        let stats = engine.stats().unwrap();
        assert_eq!(
            stats.document_count, 10,
            "exactly 10 unique docs should be live; the 20 redundant puts \
             must have been replaced, not accumulated"
        );

        // Each id should resolve to exactly one doc — the last put wins.
        for i in 0..10 {
            let docs = engine.get_documents(&format!("id{i}")).await.unwrap();
            assert_eq!(docs.len(), 1, "id{i} should resolve to a single doc");
            let title = docs[0]
                .fields
                .get("title")
                .and_then(|v| v.as_text())
                .map(String::from);
            assert_eq!(
                title.as_deref(),
                Some(format!("id{i}-rev2").as_str()),
                "id{i} should retain the last put's title"
            );
        }
    }

    /// #828: updating many docs that are still in the uncommitted buffer must
    /// stay correct under the deferred in-memory-index rebuild. Each update goes
    /// through `delete_documents` → `delete_document(old_buffered_id)` →
    /// `remove_pending_document`, which now defers the rebuild to flush. After
    /// commit, every external id must resolve to exactly its latest version and
    /// the live doc count must equal the number of unique ids (no ghosts from
    /// the superseded buffered versions).
    #[tokio::test]
    async fn test_put_document_update_many_before_commit() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        let n = 200usize;

        // Phase 1: add N distinct docs into one uncommitted buffer.
        for i in 0..n {
            let mut doc = crate::data::Document::new();
            doc.fields
                .insert("title".into(), DataValue::Text(format!("v0-{i}")));
            engine.put_document(&format!("id{i}"), doc).await.unwrap();
        }

        // Phase 2: update every one of them BEFORE committing. Each update hits
        // the deferred-rebuild path (the old version is still buffered).
        for i in 0..n {
            let mut doc = crate::data::Document::new();
            doc.fields
                .insert("title".into(), DataValue::Text(format!("v1-{i}")));
            engine.put_document(&format!("id{i}"), doc).await.unwrap();
        }

        engine.commit().await.unwrap();

        let stats = engine.stats().unwrap();
        assert_eq!(
            stats.document_count, n as u64,
            "every external id must collapse to exactly one live doc after the \
             pre-commit updates (no ghost versions from the deferred rebuild)"
        );

        // Each id resolves to exactly one doc carrying the updated content.
        for i in [0usize, 1, n / 2, n - 1] {
            let docs = engine.get_documents(&format!("id{i}")).await.unwrap();
            assert_eq!(docs.len(), 1, "id{i} must resolve to a single doc");
            let title = docs[0]
                .fields
                .get("title")
                .and_then(|v| v.as_text())
                .map(String::from);
            assert_eq!(
                title.as_deref(),
                Some(format!("v1-{i}").as_str()),
                "id{i} must carry the updated (v1) content after commit"
            );
        }
    }

    /// Regression test for the geo3d demo's "departure + re-arrival"
    /// pattern: put → commit → delete → commit → put-with-same-id →
    /// commit. The post-commit search must return exactly one doc and
    /// `engine.stats().document_count` must agree.
    ///
    /// This exercises the path where the previous version of the
    /// document is in a *committed* segment (not the writer buffer)
    /// when the next put runs `delete_documents` internally — which
    /// is what happens when an aircraft drops out of an
    /// `airplanes.live` snapshot and re-enters on a later refresh.
    #[tokio::test]
    async fn test_put_document_replaces_after_delete_across_commits() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        // Round 1: put + commit so the doc lives in a flushed segment.
        let mut doc1 = crate::data::Document::new();
        doc1.fields
            .insert("title".into(), DataValue::Text("first".into()));
        engine.put_document("X", doc1).await.unwrap();
        engine.commit().await.unwrap();

        // Round 2: delete + commit so the previous version is
        // soft-deleted in the segment.
        engine.delete_documents("X").await.unwrap();
        engine.commit().await.unwrap();
        assert!(
            engine.get_documents("X").await.unwrap().is_empty(),
            "after delete + commit, get_documents must return empty"
        );

        // Round 3: re-arrival with the same external id. The new put
        // must produce a single live doc; the soft-deleted segment
        // version must not resurface.
        let mut doc2 = crate::data::Document::new();
        doc2.fields
            .insert("title".into(), DataValue::Text("second".into()));
        engine.put_document("X", doc2).await.unwrap();
        engine.commit().await.unwrap();

        let docs = engine.get_documents("X").await.unwrap();
        assert_eq!(
            docs.len(),
            1,
            "exactly one doc should exist for id=X after departure + re-arrival"
        );
        assert_eq!(
            docs[0]
                .fields
                .get("title")
                .and_then(|v| v.as_text())
                .map(String::from)
                .as_deref(),
            Some("second"),
            "the surviving doc must be the latest put"
        );

        let stats = engine.stats().unwrap();
        assert_eq!(
            stats.document_count, 1,
            "engine.stats().document_count must agree with get_documents \
             across the departure + re-arrival cycle"
        );
    }

    /// Regression test for the geo3d-side stale-id bug: spatial queries
    /// (BKD-backed: Geo / Geo3d) used to return soft-deleted docs
    /// because the BKD tree itself does not consult the segment
    /// deletion bitmap. The fix lives in the
    /// `lexical/query/geo3d.rs::*::find_matches` helpers (and the 2D
    /// counterparts in `lexical/query/geo.rs`), which now skip
    /// `reader.is_deleted(doc_id)` hits.
    ///
    /// Steps:
    ///   1. Put a doc with a geo3d position, commit.
    ///   2. Run `geo3d_bbox(...)` over a region containing the point —
    ///      should find 1 hit (sanity check).
    ///   3. Delete the doc, commit. The BKD tree still contains the
    ///      point until a merge, but the deletion bitmap is set.
    ///   4. Run the same `geo3d_bbox(...)` query — must find 0 hits.
    #[tokio::test]
    async fn test_geo3d_query_filters_soft_deleted_docs() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::Geo3dOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("position", FieldOption::Geo3d(Geo3dOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        // Tokyo Tower in ECEF (approx).
        let mut doc = crate::data::Document::new();
        doc.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3955182.0, 3350553.0, 3700276.0,
            )),
        );
        engine.put_document("FW52", doc).await.unwrap();
        engine.commit().await.unwrap();

        // Sanity check: a wide bbox around the point matches it.
        let bbox_dsl = "position:geo3d_bbox(-3956000.0, 3349000.0, 3699000.0, \
                       -3954000.0, 3352000.0, 3702000.0)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(bbox_dsl))
            .limit(10)
            .build();
        let before = engine.search(request).await.unwrap();
        assert_eq!(before.len(), 1, "live doc should match the bbox");

        // Delete + commit. The doc is soft-deleted in the segment but
        // its BKD entry stays in place until merge.
        engine.delete_documents("FW52").await.unwrap();
        engine.commit().await.unwrap();

        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(bbox_dsl))
            .limit(10)
            .build();
        let after = engine.search(request).await.unwrap();
        assert_eq!(
            after.len(),
            0,
            "soft-deleted doc must NOT be returned by geo3d_bbox \
             (BKD entry survives in-tree until merge)",
        );

        // Same expectation for geo3d_nearest.
        let nearest_dsl = "position:geo3d_nearest(-3955182.0, 3350553.0, 3700276.0, 5)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(nearest_dsl))
            .limit(10)
            .build();
        let after_nearest = engine.search(request).await.unwrap();
        assert_eq!(
            after_nearest.len(),
            0,
            "soft-deleted doc must NOT be returned by geo3d_nearest",
        );

        // And for geo3d_distance.
        let distance_dsl = "position:geo3d_distance(-3955182.0, 3350553.0, 3700276.0, 100000.0)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(distance_dsl))
            .limit(10)
            .build();
        let after_distance = engine.search(request).await.unwrap();
        assert_eq!(
            after_distance.len(),
            0,
            "soft-deleted doc must NOT be returned by geo3d_distance",
        );
    }

    /// Regression test for #480 (`e7c206ad`): the per-segment fanout
    /// path in [`InvertedIndexSearcher::search_with_collector_parallel`]
    /// wraps each segment in a [`PerSegmentReaderView`] that did not
    /// override `get_bkd_tree`, so the trait default (`Ok(None)`)
    /// silently disabled every BKD-backed query (geo / geo3d / numeric
    /// range) once an index accumulated two or more segments. Reported
    /// in production via the `laurus-wasm/examples/geo3d` demo, where
    /// the second auto-refresh commit added a second segment and every
    /// subsequent `geo3d_nearest` returned 0 hits.
    ///
    /// Steps:
    ///   1. Put one doc, commit — segment 0.
    ///   2. Put another doc, commit — segment 1. Reader now has
    ///      `segment_count() == 2`, the fanout condition triggers.
    ///   3. Run `geo3d_distance(...)` over a sphere covering both
    ///      points — must find 2 hits, NOT 0.
    #[tokio::test]
    async fn test_geo3d_distance_multi_segment_returns_hits() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::Geo3dOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("position", FieldOption::Geo3d(Geo3dOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        // Two distinct ECEF points roughly 50 km apart near Tokyo.
        let mut doc_a = crate::data::Document::new();
        doc_a.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3955182.0, 3350553.0, 3700276.0,
            )),
        );
        engine.put_document("A", doc_a).await.unwrap();
        engine.commit().await.unwrap();

        let mut doc_b = crate::data::Document::new();
        doc_b.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3960000.0, 3350000.0, 3700000.0,
            )),
        );
        engine.put_document("B", doc_b).await.unwrap();
        engine.commit().await.unwrap();

        // 100 km sphere covers both points. With the bug, fanout makes
        // this return 0 because PerSegmentReaderView.get_bkd_tree falls
        // through to the trait default `Ok(None)`.
        let dsl = "position:geo3d_distance(-3957000.0, 3350000.0, 3700000.0, 100000.0)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(dsl))
            .limit(10)
            .build();
        let hits = engine.search(request).await.unwrap();
        assert_eq!(
            hits.len(),
            2,
            "geo3d_distance must find both docs across two segments; \
             got {} (bug: per-segment fanout drops BKD-backed queries)",
            hits.len()
        );
    }

    /// Regression test for #480: same as
    /// [`test_geo3d_distance_multi_segment_returns_hits`] but for
    /// `geo3d_nearest`, which is the query path the demo exercises.
    #[tokio::test]
    async fn test_geo3d_nearest_multi_segment_returns_hits() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::Geo3dOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("position", FieldOption::Geo3d(Geo3dOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        let mut doc_a = crate::data::Document::new();
        doc_a.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3955182.0, 3350553.0, 3700276.0,
            )),
        );
        engine.put_document("A", doc_a).await.unwrap();
        engine.commit().await.unwrap();

        let mut doc_b = crate::data::Document::new();
        doc_b.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3960000.0, 3350000.0, 3700000.0,
            )),
        );
        engine.put_document("B", doc_b).await.unwrap();
        engine.commit().await.unwrap();

        let dsl = "position:geo3d_nearest(-3957000.0, 3350000.0, 3700000.0, 5)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(dsl))
            .limit(10)
            .build();
        let hits = engine.search(request).await.unwrap();
        assert_eq!(
            hits.len(),
            2,
            "geo3d_nearest must find both docs across two segments; \
             got {} (bug: per-segment fanout returns no BKD tree)",
            hits.len()
        );
    }

    /// Regression test for #480 on numeric range queries. Same
    /// underlying cause — BKD-backed query through the fanout view.
    #[tokio::test]
    async fn test_numeric_range_multi_segment_returns_hits() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::IntegerOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("score", FieldOption::Integer(IntegerOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        let mut doc_a = crate::data::Document::new();
        doc_a.fields.insert("score".into(), DataValue::Int64(10));
        engine.put_document("A", doc_a).await.unwrap();
        engine.commit().await.unwrap();

        let mut doc_b = crate::data::Document::new();
        doc_b.fields.insert("score".into(), DataValue::Int64(20));
        engine.put_document("B", doc_b).await.unwrap();
        engine.commit().await.unwrap();

        let dsl = "score:[5 TO 25]";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(dsl))
            .limit(10)
            .build();
        let hits = engine.search(request).await.unwrap();
        assert_eq!(
            hits.len(),
            2,
            "numeric range query must find both docs across two \
             segments; got {} (bug: per-segment fanout drops BKD tree)",
            hits.len()
        );
    }

    /// Combined regression test: per-segment fanout must restore BKD
    /// query hits (#480 fix) AND continue to filter out soft-deleted
    /// hits within each segment (#400 fix). Without per-segment
    /// deletion filtering in `PerSegmentReaderView::get_bkd_tree`,
    /// the #480 fix would re-introduce the #400 ghost-hit regression.
    #[tokio::test]
    async fn test_geo3d_distance_multi_segment_filters_deleted() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::Geo3dOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("position", FieldOption::Geo3d(Geo3dOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        // Segment 0: two docs near Tokyo.
        for id in &["A", "B"] {
            let mut doc = crate::data::Document::new();
            doc.fields.insert(
                "position".into(),
                DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                    -3955182.0, 3350553.0, 3700276.0,
                )),
            );
            engine.put_document(id, doc).await.unwrap();
        }
        engine.commit().await.unwrap();

        // Soft-delete A in segment 0 and commit so segment 0 carries
        // a deletion bitmap when the fanout view consults it.
        engine.delete_documents("A").await.unwrap();
        engine.commit().await.unwrap();

        // Segment 1: one new doc.
        let mut doc_c = crate::data::Document::new();
        doc_c.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3960000.0, 3350000.0, 3700000.0,
            )),
        );
        engine.put_document("C", doc_c).await.unwrap();
        engine.commit().await.unwrap();

        let dsl = "position:geo3d_distance(-3957000.0, 3350000.0, 3700000.0, 100000.0)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(dsl))
            .limit(10)
            .build();
        let hits = engine.search(request).await.unwrap();
        assert_eq!(
            hits.len(),
            2,
            "must return live docs B and C across two segments; \
             got {} (regression: either #480 fanout BKD or #400 \
             per-segment deletion filter)",
            hits.len()
        );
    }
}
