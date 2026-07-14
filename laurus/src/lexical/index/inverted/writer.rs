//! Inverted index writer implementation.
//!
//! This module provides the writer for building inverted indexes in schema-less mode.

use std::collections::HashMap;
use std::sync::Arc;

use ahash::{AHashMap, AHashSet};

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::per_field::PerFieldAnalyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::analysis::token::Token;
use crate::error::{LaurusError, Result};
use crate::lexical::core::analyzed::{AnalyzedDocument, AnalyzedTerm};
use crate::lexical::core::document::Document;

use crate::lexical::core::field::FieldOption;
use crate::lexical::index::inverted::IndexMetadata;
use crate::lexical::index::inverted::core::posting::{Posting, TermPostingIndex};
use crate::lexical::index::inverted::segment::SegmentInfo;
use crate::lexical::index::structures::bkd_tree::BKDWriter;
use crate::lexical::index::structures::dictionary::{TermDictionaryBuilder, TermInfo};
use crate::lexical::index::structures::doc_values::DocValuesWriter;
use crate::lexical::writer::LexicalIndexWriter;

use crate::storage::Storage;
use crate::storage::structured::StructWriter;

// ============================================================================
// Inverted index writer implementation
// ============================================================================

/// Inverted index writer configuration.
#[derive(Clone)]
pub struct InvertedIndexWriterConfig {
    /// Maximum number of documents to buffer before flushing to disk.
    pub max_buffered_docs: usize,

    /// Maximum memory usage for buffering (in bytes).
    pub max_buffer_memory: usize,

    /// Segment name prefix.
    pub segment_prefix: String,

    /// Whether to store term positions for phrase queries.
    pub store_term_positions: bool,

    /// Whether to optimize segments after writing.
    pub optimize_segments: bool,

    /// Analyzer for text fields (can be PerFieldAnalyzer for field-specific analysis).
    pub analyzer: Arc<dyn Analyzer>,

    /// Shard ID for this writer.
    pub shard_id: u16,

    /// Field-specific configurations.
    pub fields: HashMap<String, FieldOption>,
}

impl std::fmt::Debug for InvertedIndexWriterConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndexWriterConfig")
            .field("max_buffered_docs", &self.max_buffered_docs)
            .field("max_buffer_memory", &self.max_buffer_memory)
            .field("segment_prefix", &self.segment_prefix)
            .field("store_term_positions", &self.store_term_positions)
            .field("optimize_segments", &self.optimize_segments)
            .field("analyzer", &self.analyzer.name())
            .finish()
    }
}

impl Default for InvertedIndexWriterConfig {
    fn default() -> Self {
        InvertedIndexWriterConfig {
            max_buffered_docs: 10000,
            max_buffer_memory: 64 * 1024 * 1024, // 64MB
            segment_prefix: "segment".to_string(),
            store_term_positions: true,
            optimize_segments: false,
            analyzer: Arc::new(StandardAnalyzer::new().unwrap()),
            shard_id: 0,
            fields: HashMap::new(),
        }
    }
}

/// Statistics about the writing process.
#[derive(Debug, Clone)]
pub struct WriterStats {
    /// Number of documents added.
    pub docs_added: u64,
    /// Number of unique terms indexed.
    pub unique_terms: u64,
    /// Total postings created.
    pub total_postings: u64,
    /// Memory currently used.
    pub memory_used: usize,
    /// Number of segments created.
    pub segments_created: u32,
    /// Number of deleted documents (from persisted segments).
    pub deleted_count: u64,
}

/// Inverted index writer implementation (schema-less mode).
pub struct InvertedIndexWriter {
    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// Writer configuration.
    config: InvertedIndexWriterConfig,

    /// In-memory inverted index being built.
    inverted_index: TermPostingIndex,

    /// Buffered analyzed documents with their assigned doc IDs.
    buffered_docs: Vec<(u64, AnalyzedDocument)>,

    /// Whether [`Self::inverted_index`] / [`Self::doc_values_writer`] are stale
    /// relative to [`Self::buffered_docs`] and need a rebuild before flush.
    ///
    /// Set by [`Self::remove_pending_document`], which drops the doc from
    /// `buffered_docs` (cheap, order-preserving `retain`) but **defers** the
    /// expensive `rebuild_in_memory_index` rather than running it per removal.
    /// The rebuild runs once at flush time (and eagerly on the same-id re-upsert
    /// path), turning an `update × M` over an `N`-doc uncommitted buffer from
    /// `O(M·N)` into `O(M) + O(N)` (Issue #828). While dirty, the in-memory
    /// index still holds the removed doc's postings, so the NRT lookups
    /// ([`Self::find_doc_id_by_term`] / [`Self::find_doc_ids_by_term`]) filter
    /// their results by [`Self::buffered_doc_ids`] (the physically-correct live
    /// set).
    index_dirty: bool,

    /// Membership index of the doc IDs currently in [`Self::buffered_docs`].
    ///
    /// Kept perfectly in sync with `buffered_docs` (every push inserts, every
    /// clear empties it) so [`Self::remove_pending_document`] can answer
    /// "is this id buffered?" in O(1). Without it the upsert path scanned the
    /// whole buffer on every call — `O(N)` per upsert, `O(N²)` over an
    /// `add × N` ingest, since newly assigned doc IDs are never already
    /// buffered yet still paid the full scan (Issue #570).
    buffered_doc_ids: AHashSet<u64>,

    /// DocValues writer for the current segment.
    doc_values_writer: DocValuesWriter,

    /// Document ID counter.
    next_doc_id: u64,

    /// Current segment number.
    current_segment: u32,

    /// Whether the writer is closed.
    closed: bool,

    /// Writer statistics.
    stats: WriterStats,

    /// Base metadata read at startup.
    base_metadata: IndexMetadata,

    /// Last processed WAL sequence number.
    last_wal_seq: u64,

    /// Pending deletions that are not yet reflected in the reader (NRT).
    pending_deletions: std::collections::HashSet<u64>,

    /// Cached `(segment_id, min_doc_id, max_doc_id)` for every committed
    /// segment, mirroring the `*.meta` files on storage (Issue #559 / #864).
    ///
    /// Built from the constructor's existing recovery scan (no extra I/O),
    /// extended in place whenever this writer flushes a segment, and rebuilt
    /// by [`Self::invalidate_segment_cache`] after an external segment
    /// rewrite ([`LexicalStore::optimize`](crate::lexical::store::LexicalStore::optimize)
    /// force-merge — the only path that rewrites segments behind a live
    /// writer; `LexicalStore::commit` drops the writer before it merges).
    /// Lets [`Self::find_segments_for_doc`] answer from memory instead of
    /// listing + JSON-parsing every `.meta` file per upsert.
    segment_ranges: Vec<(String, u64, u64)>,

    /// Highest `max_doc_id` across [`Self::segment_ranges`] (0 when no
    /// segments exist). Fresh doc IDs handed out by the WAL are strictly
    /// greater, so the steady-state ingest path rejects the "is this doc in a
    /// committed segment?" question with one integer compare.
    max_committed_doc_id: u64,

    /// Lazily created [`DeletionManager`](crate::maintenance::deletion::DeletionManager)
    /// reused across upserts of already-committed documents (Issue #571).
    /// Construction loads every `.delmap` bitmap from storage, so building it
    /// once per writer — and only when an overwrite actually needs it —
    /// replaces a full bitmap reload per call. Dropped together with
    /// [`Self::segment_ranges`] on [`Self::invalidate_segment_cache`] because
    /// a force-merge deletes the segments (and bitmaps) it holds in memory.
    deletion_manager: Option<crate::maintenance::deletion::DeletionManager>,
}

impl std::fmt::Debug for InvertedIndexWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndexWriter")
            .field("config", &self.config)
            .field("next_doc_id", &self.next_doc_id)
            .field("current_segment", &self.current_segment)
            .field("closed", &self.closed)
            .field("buffered_docs_count", &self.buffered_docs.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl InvertedIndexWriter {
    /// Create a new inverted index writer (schema-less mode).
    pub fn new(storage: Arc<dyn Storage>, config: InvertedIndexWriterConfig) -> Result<Self> {
        // Recover state from existing segments. The same scan also seeds the
        // segment-range cache (Issue #559 / #864), so the cache is warm from
        // birth at no extra I/O.
        let mut next_doc_id = 0;
        let mut max_segment_id = -1i32;
        let mut segment_ranges = Vec::new();
        let mut max_committed_doc_id = 0u64;

        if let Ok(files) = storage.list_files() {
            for file in files {
                if file.ends_with(".meta") && file != "index.meta" {
                    // unexpected error handling: ignore malformed files
                    if let Ok(input) = storage.open_input(&file)
                        && let Ok(meta) = serde_json::from_reader::<_, SegmentInfo>(input)
                    {
                        // Only consider segments from the same shard for next_doc_id (local counter part)
                        if meta.shard_id == config.shard_id {
                            let local_id = crate::util::id::get_local_id(meta.max_doc_id);
                            next_doc_id = next_doc_id.max(local_id + 1);
                        }
                        max_segment_id = max_segment_id.max(meta.generation as i32);
                        max_committed_doc_id = max_committed_doc_id.max(meta.max_doc_id);
                        segment_ranges.push((meta.segment_id, meta.min_doc_id, meta.max_doc_id));
                    }
                }
            }
        }

        let current_segment = (max_segment_id + 1) as u32;

        // Create initial DocValuesWriter (will be reset per segment)
        let initial_segment_name = format!("{}_{:06}", config.segment_prefix, current_segment);
        let doc_values_writer = DocValuesWriter::new(storage.clone(), initial_segment_name);

        // Read existing metadata or use default
        let base_metadata =
            crate::lexical::index::inverted::InvertedIndex::read_metadata(storage.as_ref())
                .unwrap_or_else(|_| IndexMetadata::default());

        Ok(InvertedIndexWriter {
            storage,
            config,
            inverted_index: TermPostingIndex::new(),
            buffered_docs: Vec::new(),
            index_dirty: false,
            buffered_doc_ids: AHashSet::new(),
            doc_values_writer,
            next_doc_id,
            current_segment,
            closed: false,
            stats: WriterStats {
                docs_added: 0,
                unique_terms: 0,
                total_postings: 0,
                memory_used: 0,
                segments_created: 0,
                deleted_count: 0,
            },
            last_wal_seq: base_metadata.last_wal_seq,
            base_metadata,
            pending_deletions: std::collections::HashSet::new(),
            segment_ranges,
            max_committed_doc_id,
            deletion_manager: None,
        })
    }

    /// Add a document to the index with automatic ID assignment.
    /// Returns the assigned document ID.
    pub fn add_document(&mut self, doc: Document) -> Result<u64> {
        self.check_closed()?;

        // Schema-less mode: no validation needed
        // Analyze the document
        let analyzed_doc = self.analyze_document(doc)?;

        // Add the analyzed document and return the assigned ID
        self.add_analyzed_document(analyzed_doc)
    }

    /// Upsert a document to the index with a specific document ID.
    pub fn upsert_document(&mut self, doc_id: u64, doc: Document) -> Result<()> {
        self.check_closed()?;

        // Analyze the document
        let analyzed_doc = self.analyze_document(doc)?;

        // Same-id re-upsert detection: if this exact id is already buffered, the
        // deferred-rebuild scheme cannot distinguish the old version's stale
        // postings from the new version's (both carry this id), so we must purge
        // the old version from the in-memory index *before* re-indexing. This
        // never happens on the production engine path (every add gets a fresh
        // monotonic doc_id), so the eager rebuild here costs nothing in
        // production and only restores exact NRT state for direct re-upserts.
        let was_buffered = self.buffered_doc_ids.contains(&doc_id);

        // Upsert: remove any pending document with the same ID before adding
        self.remove_pending_document(doc_id)?;
        // Upsert: mark persisted occurrences as deleted (flushed segments)
        self.mark_persisted_doc_deleted(doc_id)?;

        if was_buffered {
            // Purge the just-removed old version's postings now (eager), so the
            // re-added version below is the only one in the in-memory index.
            self.rebuild_in_memory_index()?;
            self.index_dirty = false;
        }

        // Add the analyzed document with the specified ID
        self.upsert_analyzed_document(doc_id, analyzed_doc)
    }

    /// Add an already analyzed document to the index with a specific document ID.
    pub fn upsert_analyzed_document(
        &mut self,
        doc_id: u64,
        analyzed_doc: AnalyzedDocument,
    ) -> Result<()> {
        self.check_closed()?;

        // Update next_doc_id if necessary to avoid ID collisions
        if doc_id >= self.next_doc_id {
            self.next_doc_id = doc_id + 1;
        }

        // Add field values to DocValues
        for (field_name, value) in &analyzed_doc.stored_fields {
            self.doc_values_writer
                .add_value(doc_id, field_name, value.clone());
        }

        // Add to inverted index
        self.add_analyzed_document_to_index(doc_id, &analyzed_doc)?;

        // Buffer the document with its assigned ID
        self.buffered_docs.push((doc_id, analyzed_doc));
        self.buffered_doc_ids.insert(doc_id);
        self.stats.docs_added += 1;

        // Check if we need to flush
        if self.should_flush() {
            self.flush_segment()?;
        }

        Ok(())
    }

    /// Add an already analyzed document to the index with automatic ID assignment.
    /// Returns the assigned document ID.
    ///
    /// This method allows you to add pre-analyzed documents directly,
    /// bypassing the internal document analysis step. This is useful when:
    /// - You want to use DocumentParser explicitly for better control
    /// - You have pre-tokenized documents from external systems
    /// - You need to customize the analysis process
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use laurus::lexical::core::document::Document;
    /// use laurus::lexical::core::parser::DocumentParser;
    /// use laurus::analysis::analyzer::per_field::PerFieldAnalyzer;
    /// use laurus::analysis::analyzer::standard::StandardAnalyzer;
    /// use laurus::lexical::index::inverted::writer::{InvertedIndexWriter, InvertedIndexWriterConfig};
    /// use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use laurus::storage::StorageConfig;
    /// use std::sync::Arc;
    ///
    /// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    /// let per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
    /// let config = InvertedIndexWriterConfig {
    ///     analyzer: Arc::new(per_field.clone()),
    ///     ..Default::default()
    /// };
    /// let mut writer = InvertedIndexWriter::new(storage, config).unwrap();
    ///
    /// use laurus::lexical::core::field::TextOption;
    /// let doc = Document::builder()
    ///     .add_text("title", "Rust Programming")
    ///     .build();
    ///
    /// let doc_parser = DocumentParser::new(Arc::new(per_field));
    /// let analyzed = doc_parser.parse(doc).unwrap();
    /// let doc_id = writer.add_analyzed_document(analyzed).unwrap();
    /// ```
    pub fn add_analyzed_document(&mut self, analyzed_doc: AnalyzedDocument) -> Result<u64> {
        self.check_closed()?;

        // Assign document ID using shard-prefixed strategy
        let local_id = self.next_doc_id;
        self.next_doc_id += 1;
        let doc_id = crate::util::id::create_doc_id(self.config.shard_id, local_id);

        // Add the analyzed document with the assigned ID
        self.upsert_analyzed_document(doc_id, analyzed_doc)?;

        Ok(doc_id)
    }

    /// Find the internal document ID for a given term (field:value).
    ///
    /// This searches both the in-memory buffer (uncommitted) and, in the future,
    /// persisted segments (committed).
    ///
    /// Currently only searches the in-memory buffer for NRT (Near Real-Time) lookups.
    pub fn find_doc_id_by_term(&self, field: &str, term: &str) -> Result<Option<u64>> {
        let full_term = format!("{field}:{term}");

        // 1. Check in-memory inverted index. The index may hold stale postings
        // for docs removed since the last rebuild (deferred under #828), so keep
        // only ids still in the live buffer and return the highest (latest).
        if let Some(posting_list) = self.inverted_index.get_posting_list(&full_term)
            && let Some(doc_id) = posting_list
                .postings
                .iter()
                .map(|p| p.doc_id)
                .filter(|id| self.buffered_doc_ids.contains(id))
                .max()
        {
            return Ok(Some(doc_id));
        }

        // 2. TODO: Check persisted segments
        // This requires opening readers for existing segments, which is expensive if done on every write.
        // For now, we rely on the upper layer (LexicalStore/HybridEngine) to check committed segments via Readers,
        // and use this method specifically for the "In-Memory / NRT" part of the check.
        // Or we implement a BloomFilter cache for segments here.

        Ok(None)
    }

    /// Find all internal document IDs for a given term (field:value).
    ///
    /// This searches both the in-memory buffer (uncommitted) and, in the future,
    /// persisted segments (committed).
    fn find_doc_ids_by_term(&self, field: &str, term: &str) -> Result<Option<Vec<u64>>> {
        let full_term = format!("{field}:{term}");

        // 1. Check in-memory inverted index. The index may hold stale postings
        // for docs removed since the last rebuild (deferred under #828), so keep
        // only ids still in the live buffer; a same-id re-upsert can also leave
        // two postings for one id, so dedup.
        if let Some(posting_list) = self.inverted_index.get_posting_list(&full_term) {
            let mut seen = AHashSet::new();
            let ids: Vec<u64> = posting_list
                .postings
                .iter()
                .map(|p| p.doc_id)
                .filter(|id| self.buffered_doc_ids.contains(id) && seen.insert(*id))
                .collect();
            if !ids.is_empty() {
                return Ok(Some(ids));
            }
        }

        Ok(None)
    }

    /// Analyze a document into terms.
    fn analyze_document(&mut self, doc: Document) -> Result<AnalyzedDocument> {
        let mut field_terms = AHashMap::new();
        let mut stored_fields = AHashMap::new();
        let mut point_values = AHashMap::new();

        // Process each field in the document
        for (field_name, val) in &doc.fields {
            use crate::data::DataValue;

            // 1. Get field option (Schema-aware)
            // If the field is not in schema, we check if it starts with "_" (internal)
            // Internal fields are indexed/stored by default unless explicitly disabled?
            // For now, let's say if NOT in schema, we follow a "relaxed" schema-less mode:
            // - If config.fields is empty, we index/store everything (backward compat)
            // - If config.fields is NOT empty, we ONLY index/store what's in schema,
            //   plus reserved fields like "_id".
            let option = if let Some(opt) = self.config.fields.get(field_name) {
                Some(opt)
            } else if field_name.starts_with('_') || self.config.fields.is_empty() {
                // If it's an internal field or we are in schema-less mode (empty config)
                None // Uses default behavior below
            } else {
                continue; // Skip fields not in schema
            };

            let (should_index, should_store) = match option {
                Some(FieldOption::Text(opt)) => (opt.indexed, opt.stored),
                Some(FieldOption::Integer(opt)) => (opt.indexed, opt.stored),
                Some(FieldOption::Float(opt)) => (opt.indexed, opt.stored),
                Some(FieldOption::Boolean(opt)) => (opt.indexed, opt.stored),
                Some(FieldOption::DateTime(opt)) => (opt.indexed, opt.stored),
                Some(FieldOption::Geo(opt)) => (opt.indexed, opt.stored),
                Some(FieldOption::Geo3d(opt)) => (opt.indexed, opt.stored),
                Some(FieldOption::Bytes(opt)) => (false, opt.stored), // Bytes are not lexically indexed
                None => (true, true), // Internal or schema-less default
            };

            // Index the field if enabled
            if should_index {
                match val {
                    DataValue::Text(text) => {
                        // Use analyzer from config (can be PerFieldAnalyzer for field-specific analysis)
                        let tokens = if let Some(per_field) = self
                            .config
                            .analyzer
                            .as_any()
                            .downcast_ref::<PerFieldAnalyzer>()
                        {
                            per_field.analyze_field(field_name, text)?
                        } else {
                            self.config.analyzer.analyze(text)?
                        };
                        let token_vec: Vec<Token> = tokens.collect();
                        let analyzed_terms = self.tokens_to_analyzed_terms(token_vec);

                        field_terms.insert(field_name.clone(), analyzed_terms);
                    }

                    DataValue::Int64(num) => {
                        // Convert integer to text for indexing
                        let text = num.to_string();

                        let analyzed_term = AnalyzedTerm {
                            term: text.clone(),
                            position: 0,
                            frequency: 1,
                            offset: (0, text.len()),
                        };

                        field_terms.insert(field_name.clone(), vec![analyzed_term]);
                        point_values.insert(field_name.clone(), vec![vec![*num as f64]]);
                    }
                    DataValue::Float64(num) => {
                        // ...
                        field_terms.insert(
                            field_name.clone(),
                            vec![AnalyzedTerm {
                                term: num.to_string(),
                                position: 0,
                                frequency: 1,
                                offset: (0, num.to_string().len()),
                            }],
                        );
                        point_values.insert(field_name.clone(), vec![vec![*num]]);
                    }
                    DataValue::DateTime(dt) => {
                        let ts = dt.timestamp() as f64;
                        field_terms.insert(
                            field_name.clone(),
                            vec![AnalyzedTerm {
                                term: ts.to_string(),
                                position: 0,
                                frequency: 1,
                                offset: (0, ts.to_string().len()),
                            }],
                        );
                        point_values.insert(field_name.clone(), vec![vec![ts]]);
                    }
                    DataValue::Bool(b) => {
                        // bool is indexed as "true"/"false" text for lexical queries,
                        // and also stored as a point value (1.0/0.0) for numeric range queries
                        let text = if *b { "true" } else { "false" };
                        field_terms.insert(
                            field_name.clone(),
                            vec![AnalyzedTerm {
                                term: text.to_string(),
                                position: 0,
                                frequency: 1,
                                offset: (0, text.len()),
                            }],
                        );
                    }
                    DataValue::Geo(p) => {
                        // Geo points are indexed as 2D points in BKD
                        field_terms.insert(
                            field_name.clone(),
                            vec![AnalyzedTerm {
                                term: format!("{},{}", p.lat, p.lon),
                                position: 0,
                                frequency: 1,
                                offset: (0, format!("{},{}", p.lat, p.lon).len()),
                            }],
                        );
                        point_values.insert(field_name.clone(), vec![vec![p.lat, p.lon]]);
                    }
                    DataValue::GeoEcef(p) => {
                        // 3D ECEF point: emitting a 3-element point here is
                        // what tells `BKDWriter::new` to build a 3D BKD when
                        // the per-field tree is materialized. Schema-side,
                        // `FieldOption::Geo3d` (#298) classifies the field
                        // as lexical and respects the (indexed, stored)
                        // flags exactly like 2D Geo.
                        field_terms.insert(
                            field_name.clone(),
                            vec![AnalyzedTerm {
                                term: format!("{},{},{}", p.x, p.y, p.z),
                                position: 0,
                                frequency: 1,
                                offset: (0, format!("{},{},{}", p.x, p.y, p.z).len()),
                            }],
                        );
                        point_values.insert(field_name.clone(), vec![vec![p.x, p.y, p.z]]);
                    }
                    DataValue::Int64Array(arr) => {
                        // Multi-valued integer field. Each element is a
                        // distinct analyzed term and a distinct 1D BKD
                        // point so range queries match if any value falls
                        // in range (Lucene "any match" semantics).
                        let mut terms: Vec<AnalyzedTerm> = Vec::with_capacity(arr.len());
                        let mut points: Vec<Vec<f64>> = Vec::with_capacity(arr.len());
                        let mut offset = 0usize;
                        for (idx, num) in arr.iter().enumerate() {
                            let text = num.to_string();
                            let len = text.len();
                            terms.push(AnalyzedTerm {
                                term: text,
                                position: idx as u32,
                                frequency: 1,
                                offset: (offset, offset + len),
                            });
                            offset += len + 1;
                            points.push(vec![*num as f64]);
                        }
                        field_terms.insert(field_name.clone(), terms);
                        point_values.insert(field_name.clone(), points);
                    }
                    DataValue::Float64Array(arr) => {
                        // Multi-valued float field. Same shape as Int64Array.
                        let mut terms: Vec<AnalyzedTerm> = Vec::with_capacity(arr.len());
                        let mut points: Vec<Vec<f64>> = Vec::with_capacity(arr.len());
                        let mut offset = 0usize;
                        for (idx, num) in arr.iter().enumerate() {
                            let text = num.to_string();
                            let len = text.len();
                            terms.push(AnalyzedTerm {
                                term: text,
                                position: idx as u32,
                                frequency: 1,
                                offset: (offset, offset + len),
                            });
                            offset += len + 1;
                            points.push(vec![*num]);
                        }
                        field_terms.insert(field_name.clone(), terms);
                        point_values.insert(field_name.clone(), points);
                    }
                    // Handle other variants (Bytes, Vector, Null) — not lexically indexed.
                    _ => {}
                }
            }

            // Store the field if enabled
            if should_store {
                stored_fields.insert(field_name.clone(), val.clone());
            }
        }

        // Calculate field lengths (number of tokens per field)
        let mut field_lengths = AHashMap::new();
        for (field_name, terms) in &field_terms {
            field_lengths.insert(field_name.clone(), terms.len() as u32);
        }

        Ok(AnalyzedDocument {
            field_terms,
            stored_fields,
            field_lengths,
            point_values,
        })
    }

    /// Convert tokens to analyzed terms.
    fn tokens_to_analyzed_terms(&self, tokens: Vec<Token>) -> Vec<AnalyzedTerm> {
        let mut term_frequencies = AHashMap::new();
        let mut analyzed_terms = Vec::new();

        for (position, token) in tokens.into_iter().enumerate() {
            let term = token.text;
            let frequency = term_frequencies.entry(term.clone()).or_insert(0);
            *frequency += 1;

            analyzed_terms.push(AnalyzedTerm {
                term: term.clone(),
                position: position as u32,
                frequency: *frequency,
                offset: (token.start_offset, token.end_offset),
            });
        }

        analyzed_terms
    }

    /// Add an analyzed document to the inverted index.
    fn add_analyzed_document_to_index(
        &mut self,
        doc_id: u64,
        doc: &AnalyzedDocument,
    ) -> Result<()> {
        for (field_name, terms) in &doc.field_terms {
            for analyzed_term in terms {
                let full_term = format!("{field_name}:{}", analyzed_term.term);

                let posting = if self.config.store_term_positions {
                    Posting::with_positions(doc_id, vec![analyzed_term.position])
                } else {
                    Posting::with_frequency(doc_id, analyzed_term.frequency)
                };

                self.inverted_index.add_posting(full_term, posting);
                self.stats.total_postings += 1;
            }
        }

        self.stats.unique_terms = self.inverted_index.term_count();
        Ok(())
    }

    /// Check if we should flush the current segment.
    fn should_flush(&self) -> bool {
        self.buffered_docs.len() >= self.config.max_buffered_docs
            || self.estimate_memory_usage() >= self.config.max_buffer_memory
    }

    /// Estimate current memory usage.
    fn estimate_memory_usage(&self) -> usize {
        // Rough estimation
        let doc_memory = self.buffered_docs.len() * 1024; // 1KB per doc estimate
        let index_memory = self.inverted_index.term_count() as usize * 256; // 256 bytes per term estimate
        doc_memory + index_memory
    }

    /// Flush the current segment to disk.
    fn flush_segment(&mut self) -> Result<()> {
        if self.buffered_docs.is_empty() {
            return Ok(());
        }

        // Materialize any deferred removals (#828) so the in-memory index +
        // DocValues match `buffered_docs` before they are written to disk.
        if self.index_dirty {
            self.rebuild_in_memory_index()?;
            self.index_dirty = false;
        }

        let segment_name = format!("{}_{:06}", self.config.segment_prefix, self.current_segment);

        self.write_segment_files(&segment_name)?;

        // Mirror the freshly written `.meta` into the segment-range cache
        // (Issue #559 / #864) so an overwrite of one of these docs later in
        // this writer's life resolves without rescanning storage.
        self.extend_segment_cache(&segment_name);

        // Clear buffers
        self.buffered_docs.clear();
        self.buffered_doc_ids.clear();
        self.index_dirty = false;
        self.inverted_index = TermPostingIndex::new();

        // Reset DocValuesWriter for next segment
        let next_segment_name = format!(
            "{}_{:06}",
            self.config.segment_prefix,
            self.current_segment + 1
        );
        self.doc_values_writer = DocValuesWriter::new(self.storage.clone(), next_segment_name);

        self.current_segment += 1;
        self.stats.segments_created += 1;

        Ok(())
    }

    /// Write all per-segment files for the currently buffered documents under
    /// `segment_name`, returning the written file paths.
    ///
    /// Shared by the normal flush path ([`Self::flush_segment`]) and the merge
    /// path ([`Self::flush_buffered_to_segment`], Issue #753) so both produce an
    /// identical, complete, typed segment. Does not touch buffers or the
    /// auto-naming counter — callers manage those.
    ///
    /// # Arguments
    ///
    /// * `segment_name` - Name the segment's files are written under.
    fn write_segment_files(&self, segment_name: &str) -> Result<Vec<String>> {
        self.write_inverted_index(segment_name)?;
        self.write_stored_documents(segment_name)?;
        self.write_field_lengths(segment_name)?;
        self.write_field_stats(segment_name)?;
        self.write_doc_values(segment_name)?;
        self.write_segment_metadata(segment_name)?;
        self.write_bkd_trees(segment_name)?;
        // The redundant `.json` stored-field mirror is no longer written
        // (Issue #756); stored fields are read from the typed `.docs` file.

        // Collect every file written under this segment prefix (the BKD step
        // writes one `.{field}.bkd` per numeric/geo field, so enumerate rather
        // than hard-code the set).
        let prefix = format!("{segment_name}.");
        let paths = self
            .storage
            .list_files()?
            .into_iter()
            .filter(|f| f.starts_with(&prefix))
            .collect();
        Ok(paths)
    }

    /// Flush the currently buffered documents to a caller-named segment,
    /// returning the written file paths (Issue #753).
    ///
    /// Unlike [`Self::flush_segment`], the segment name is supplied by the
    /// caller (the merge engine names the merged segment) and the auto-naming
    /// counter is left untouched. Buffers are cleared afterwards so the writer
    /// can be dropped. Returns an empty vector when nothing is buffered.
    ///
    /// # Arguments
    ///
    /// * `segment_name` - Name the merged segment's files are written under.
    pub fn flush_buffered_to_segment(&mut self, segment_name: &str) -> Result<Vec<String>> {
        if self.buffered_docs.is_empty() {
            return Ok(Vec::new());
        }
        // Materialize any deferred removals (#828) before writing. The merge
        // path only adds distinct, pre-deduped docs so this is normally a no-op,
        // but the guard keeps the invariant if that ever changes.
        if self.index_dirty {
            self.rebuild_in_memory_index()?;
            self.index_dirty = false;
        }
        let paths = self.write_segment_files(segment_name)?;
        self.extend_segment_cache(segment_name);
        self.buffered_docs.clear();
        self.buffered_doc_ids.clear();
        self.index_dirty = false;
        self.inverted_index = TermPostingIndex::new();
        self.stats.segments_created += 1;
        Ok(paths)
    }

    /// Write the inverted index to storage.
    fn write_inverted_index(&self, segment_name: &str) -> Result<()> {
        // Write posting lists
        let posting_file = format!("{segment_name}.post");
        let posting_output = self.storage.create_output(&posting_file)?;
        let mut posting_writer = StructWriter::new(posting_output);

        let mut term_dict_builder = TermDictionaryBuilder::new();

        // Compute per-field average length and a `doc_id → field_lengths`
        // lookup once per flush. Both are needed to precompute the
        // tightened BM25 TF-component upper bound stored as
        // `TermInfo::max_score_factor` (#403 PR-B2). The factor uses
        // the default BM25 parameters (`k1 = 1.2`, `b = 0.75`) — at
        // search time, scorers fall back to the loose `k1 + 1` ceiling
        // when the caller overrides `(k1, b)`.
        let field_avg_lengths = self.compute_field_avg_lengths();
        let field_lengths_by_doc: AHashMap<u64, &AHashMap<String, u32>> = self
            .buffered_docs
            .iter()
            .map(|(doc_id, doc)| (*doc_id, &doc.field_lengths))
            .collect();

        // Collect and sort terms for deterministic output
        let mut terms: Vec<_> = self.inverted_index.terms().collect();
        terms.sort();

        for term in terms {
            if let Some(posting_list) = self.inverted_index.get_posting_list(term) {
                let start_offset = posting_writer.position();

                // Write posting list (v2: includes on-disk multi-level
                // skip table for O(log_8 N) `skip_to`, #503).
                posting_list.encode_v2(&mut posting_writer)?;

                let end_offset = posting_writer.position();
                let length = end_offset - start_offset;

                // The internal term key is `"<field>:<term>"`; split on
                // the first `:` to recover the field name. Terms that
                // somehow lack a field prefix get a `0.0` factor and
                // the BM25 scorer will fall back to the loose bound.
                let (max_score_factor, block_max) = match term.split_once(':') {
                    Some((field_name, _)) => {
                        let avg_len = field_avg_lengths
                            .get(field_name)
                            .copied()
                            .unwrap_or(0.0_f32);
                        let term_factor = Self::compute_term_max_score_factor(
                            posting_list,
                            field_name,
                            avg_len,
                            &field_lengths_by_doc,
                        );
                        // Compute per-block max-impact metadata for
                        // Block-Max-WAND (#403 PR-C). A term with no
                        // postings naturally produces no blocks.
                        let blocks = Self::compute_term_block_max(
                            posting_list,
                            field_name,
                            avg_len,
                            &field_lengths_by_doc,
                        );
                        (term_factor, blocks)
                    }
                    None => (0.0_f32, Vec::new()),
                };

                // Add to term dictionary
                let term_info = TermInfo::with_block_max(
                    start_offset,
                    length,
                    posting_list.doc_frequency,
                    posting_list.total_frequency,
                    max_score_factor,
                    block_max,
                );
                term_dict_builder.add_term(term.clone(), term_info);
            }
        }

        posting_writer.close()?;

        // Write term dictionary
        let dict_file = format!("{segment_name}.dict");
        let dict_output = self.storage.create_output(&dict_file)?;
        let mut dict_writer = StructWriter::new(dict_output);

        let term_dict = term_dict_builder.build()?;
        term_dict.write_to_storage(&mut dict_writer)?;
        dict_writer.close()?;

        Ok(())
    }

    /// Compute per-field average field length over the currently
    /// buffered documents. Used by [`Self::write_inverted_index`] to
    /// precompute the tightened BM25 TF-component bound that lands in
    /// each term's `TermInfo::max_score_factor` (#403 PR-B2).
    fn compute_field_avg_lengths(&self) -> AHashMap<String, f32> {
        let mut totals: AHashMap<String, (u64, u64)> = AHashMap::new();
        for (_doc_id, doc) in &self.buffered_docs {
            for (field_name, &length) in &doc.field_lengths {
                let entry = totals.entry(field_name.clone()).or_insert((0, 0));
                entry.0 += 1; // doc count contributing to this field
                entry.1 += length as u64; // total length for this field
            }
        }
        totals
            .into_iter()
            .map(|(field, (count, total))| {
                let avg = if count > 0 {
                    total as f32 / count as f32
                } else {
                    0.0
                };
                (field, avg)
            })
            .collect()
    }

    /// Compute the per-term tightened BM25 TF-component upper bound
    /// using the default `k1 = 1.2`, `b = 0.75` parameters and the
    /// segment's average field length (#403 PR-B2).
    ///
    /// Returns the maximum of
    /// `(tf · (k1 + 1)) / (tf + k1 · (1 - b + b · (L / avg_L)))`
    /// taken over every posting in `posting_list`. `0.0` if the
    /// posting list is empty or no doc lengths are resolvable.
    fn compute_term_max_score_factor(
        posting_list: &crate::lexical::index::inverted::core::posting::PostingList,
        field_name: &str,
        avg_field_length: f32,
        field_lengths_by_doc: &AHashMap<u64, &AHashMap<String, u32>>,
    ) -> f32 {
        const K1: f32 = 1.2;
        const B: f32 = 0.75;

        let mut max_factor: f32 = 0.0;
        for posting in &posting_list.postings {
            let tf = posting.frequency as f32;
            if tf == 0.0 {
                continue;
            }
            let field_len = field_lengths_by_doc
                .get(&posting.doc_id)
                .and_then(|fls| fls.get(field_name))
                .copied()
                .unwrap_or(0) as f32;
            let len_ratio = if avg_field_length > 0.0 {
                field_len / avg_field_length
            } else {
                1.0
            };
            let denom = tf + K1 * (1.0 - B + B * len_ratio);
            if denom == 0.0 {
                continue;
            }
            let factor = (tf * (K1 + 1.0)) / denom;
            if factor > max_factor {
                max_factor = factor;
            }
        }
        max_factor
    }

    /// Compute the per-block max-impact metadata used by Block-Max-WAND
    /// (#403 PR-C). Walks the posting list in
    /// [`BLOCK_SIZE`](crate::lexical::index::structures::dictionary::BLOCK_SIZE)-wide
    /// chunks and records `(last_doc_id, max_factor)` for each block.
    ///
    /// `max_factor` is computed with the same formula as
    /// [`Self::compute_term_max_score_factor`] but restricted to the
    /// block's postings; the per-term factor is the max over all per-
    /// block factors.
    fn compute_term_block_max(
        posting_list: &crate::lexical::index::inverted::core::posting::PostingList,
        field_name: &str,
        avg_field_length: f32,
        field_lengths_by_doc: &AHashMap<u64, &AHashMap<String, u32>>,
    ) -> Vec<crate::lexical::index::structures::dictionary::BlockMax> {
        use crate::lexical::index::structures::dictionary::{BLOCK_SIZE, BlockMax};

        const K1: f32 = 1.2;
        const B: f32 = 0.75;

        if posting_list.postings.is_empty() {
            return Vec::new();
        }

        let mut blocks = Vec::with_capacity(posting_list.postings.len().div_ceil(BLOCK_SIZE));
        for chunk in posting_list.postings.chunks(BLOCK_SIZE) {
            let mut block_max: f32 = 0.0;
            for posting in chunk {
                let tf = posting.frequency as f32;
                if tf == 0.0 {
                    continue;
                }
                let field_len = field_lengths_by_doc
                    .get(&posting.doc_id)
                    .and_then(|fls| fls.get(field_name))
                    .copied()
                    .unwrap_or(0) as f32;
                let len_ratio = if avg_field_length > 0.0 {
                    field_len / avg_field_length
                } else {
                    1.0
                };
                let denom = tf + K1 * (1.0 - B + B * len_ratio);
                if denom == 0.0 {
                    continue;
                }
                let factor = (tf * (K1 + 1.0)) / denom;
                if factor > block_max {
                    block_max = factor;
                }
            }
            // `chunks` always yields at least one element (we early-
            // returned on empty above), so unwrap is safe.
            let last_doc_id = chunk.last().unwrap().doc_id;
            blocks.push(BlockMax {
                last_doc_id,
                max_factor: block_max,
            });
        }
        blocks
    }

    /// Write stored documents to storage with type information preserved.
    fn write_stored_documents(&self, segment_name: &str) -> Result<()> {
        let stored_file = format!("{segment_name}.docs");
        let stored_output = self.storage.create_output(&stored_file)?;
        let mut stored_writer = StructWriter::new(stored_output);

        // Write document count
        stored_writer.write_varint(self.buffered_docs.len() as u64)?;

        // Write each document
        for (doc_id, doc) in &self.buffered_docs {
            stored_writer.write_u64(*doc_id)?;
            stored_writer.write_varint(doc.stored_fields.len() as u64)?;

            for (field_name, field_value) in &doc.stored_fields {
                stored_writer.write_string(field_name)?;

                // Write type tag and value
                match field_value {
                    crate::data::DataValue::Text(text) => {
                        stored_writer.write_u8(0)?; // Type tag for Text
                        stored_writer.write_string(text)?;
                    }
                    crate::data::DataValue::Int64(num) => {
                        stored_writer.write_u8(1)?; // Type tag for Integer
                        stored_writer.write_u64(*num as u64)?; // Store as u64, preserving bit pattern
                    }
                    crate::data::DataValue::Float64(num) => {
                        stored_writer.write_u8(2)?; // Type tag for Float
                        stored_writer.write_f64(*num)?;
                    }
                    crate::data::DataValue::Bool(b) => {
                        stored_writer.write_u8(3)?; // Type tag for Boolean
                        stored_writer.write_u8(if *b { 1 } else { 0 })?;
                    }
                    crate::data::DataValue::DateTime(dt) => {
                        stored_writer.write_u8(5)?; // Type tag for DateTime
                        stored_writer.write_string(&dt.to_rfc3339())?;
                    }
                    crate::data::DataValue::Geo(p) => {
                        stored_writer.write_u8(6)?; // Type tag for Geo
                        stored_writer.write_f64(p.lat)?;
                        stored_writer.write_f64(p.lon)?;
                    }
                    crate::data::DataValue::GeoEcef(p) => {
                        // Type tag 12 = 3D ECEF point. Tag 11 was originally
                        // claimed for ECEF in #297 but collided with the
                        // pre-existing Float64Array tag (also 11); #299
                        // moves ECEF to tag 12 and wires reader support.
                        stored_writer.write_u8(12)?;
                        stored_writer.write_f64(p.x)?;
                        stored_writer.write_f64(p.y)?;
                        stored_writer.write_f64(p.z)?;
                    }
                    crate::data::DataValue::Bytes(bytes, mime) => {
                        stored_writer.write_u8(4)?; // Type tag for Bytes
                        stored_writer.write_string(mime.as_deref().unwrap_or(""))?;
                        stored_writer.write_varint(bytes.len() as u64)?;
                        stored_writer.write_bytes(bytes)?;
                    }
                    crate::data::DataValue::Null => {
                        stored_writer.write_u8(7)?; // Type tag for Null
                    }
                    crate::data::DataValue::Vector(v) => {
                        stored_writer.write_u8(9)?; // Type tag for Vector
                        stored_writer.write_varint(v.len() as u64)?;
                        for &f in v {
                            stored_writer.write_f32(f)?;
                        }
                    }
                    crate::data::DataValue::Int64Array(arr) => {
                        stored_writer.write_u8(10)?; // Type tag for Int64Array
                        stored_writer.write_varint(arr.len() as u64)?;
                        for &v in arr {
                            stored_writer.write_u64(v as u64)?;
                        }
                    }
                    crate::data::DataValue::Float64Array(arr) => {
                        stored_writer.write_u8(11)?; // Type tag for Float64Array
                        stored_writer.write_varint(arr.len() as u64)?;
                        for &v in arr {
                            stored_writer.write_f64(v)?;
                        }
                    }
                }
            }
        }

        stored_writer.close()?;
        Ok(())
    }

    /// Calculate field statistics from buffered documents.
    fn calculate_field_stats(&self) -> AHashMap<String, (u64, f64, u64, u64)> {
        // field_name -> (doc_count, total_length, min_length, max_length)
        let mut field_stats: AHashMap<String, (u64, u64, u64, u64)> = AHashMap::new();

        for (_doc_id, doc) in &self.buffered_docs {
            for (field_name, &length) in &doc.field_lengths {
                let stats = field_stats
                    .entry(field_name.clone())
                    .or_insert((0, 0, u64::MAX, 0));
                stats.0 += 1; // doc_count
                stats.1 += length as u64; // total_length
                stats.2 = stats.2.min(length as u64); // min_length
                stats.3 = stats.3.max(length as u64); // max_length
            }
        }

        // Convert to (doc_count, avg_length, min_length, max_length)
        field_stats
            .into_iter()
            .map(
                |(field, (doc_count, total_length, min_length, max_length))| {
                    let avg_length = if doc_count > 0 {
                        total_length as f64 / doc_count as f64
                    } else {
                        0.0
                    };
                    (field, (doc_count, avg_length, min_length, max_length))
                },
            )
            .collect()
    }

    /// Write field lengths to storage.
    fn write_field_lengths(&self, segment_name: &str) -> Result<()> {
        let lens_file = format!("{segment_name}.lens");
        let lens_output = self.storage.create_output(&lens_file)?;
        let mut lens_writer = StructWriter::new(lens_output);

        // Write document count
        lens_writer.write_varint(self.buffered_docs.len() as u64)?;

        // Write field lengths for each document
        for (doc_id, doc) in &self.buffered_docs {
            lens_writer.write_u64(*doc_id)?;
            lens_writer.write_varint(doc.field_lengths.len() as u64)?;

            for (field_name, length) in &doc.field_lengths {
                lens_writer.write_string(field_name)?;
                lens_writer.write_u32(*length)?;
            }
        }

        lens_writer.close()?;
        Ok(())
    }

    /// Write field statistics to storage.
    fn write_field_stats(&self, segment_name: &str) -> Result<()> {
        let fstats_file = format!("{segment_name}.fstats");
        let fstats_output = self.storage.create_output(&fstats_file)?;
        let mut fstats_writer = StructWriter::new(fstats_output);

        let field_stats = self.calculate_field_stats();

        // Write number of fields
        fstats_writer.write_varint(field_stats.len() as u64)?;

        for (field_name, (doc_count, avg_length, min_length, max_length)) in field_stats {
            fstats_writer.write_string(&field_name)?;
            fstats_writer.write_u64(doc_count)?;
            fstats_writer.write_f64(avg_length)?;
            fstats_writer.write_u64(min_length)?;
            fstats_writer.write_u64(max_length)?;
        }

        fstats_writer.close()?;
        Ok(())
    }

    /// Write DocValues to storage.
    fn write_doc_values(&self, segment_name: &str) -> Result<()> {
        // Write under the caller-supplied segment name. On the normal flush
        // path this equals the writer's own `segment_name`; the merge path
        // (Issue #753) passes the merged segment's name so accumulated values
        // land in the right `.dv` file.
        self.doc_values_writer.write_to(segment_name)?;
        Ok(())
    }

    /// Write BKD trees for numeric and geo fields.
    ///
    /// Per-field state is accumulated into flat row-major coordinate buffers
    /// (`points`) and parallel doc-id buffers (`doc_ids`) so the BKD writer
    /// can be fed without re-allocating per point. The dimensionality is
    /// captured from the first point seen for each field.
    fn write_bkd_trees(&self, segment_name: &str) -> Result<()> {
        // (flat points, doc_ids, num_dims)
        let mut field_buckets: AHashMap<String, (Vec<f64>, Vec<u64>, usize)> = AHashMap::new();

        for (doc_id, doc) in &self.buffered_docs {
            for (field, points) in &doc.point_values {
                // Each `point` is one BKD entry. A single-valued field
                // contributes one entry; a multi-valued field contributes
                // one per element. The BKD reader's `range_search` already
                // deduplicates `doc_id`s, so a multi-valued document is
                // reported at most once per query.
                for point in points {
                    let bucket = field_buckets
                        .entry(field.clone())
                        .or_insert_with(|| (Vec::new(), Vec::new(), point.len()));
                    bucket.0.extend_from_slice(point);
                    bucket.1.push(*doc_id);
                }
            }
        }

        for (field, (points, doc_ids, num_dims)) in field_buckets {
            if doc_ids.is_empty() {
                continue;
            }

            let file_name = format!("{segment_name}.{field}.bkd");
            let output = self.storage.create_output(&file_name)?;
            let mut writer = BKDWriter::new(output, num_dims as u32);
            writer.write(&points, &doc_ids)?;
            writer.finish()?;
        }
        Ok(())
    }

    /// Doc-ID range `(min, max)` of the currently buffered documents,
    /// `(0, 0)` when the buffer is empty. Shared by
    /// [`Self::write_segment_metadata`] and [`Self::extend_segment_cache`] so
    /// the on-storage `.meta` and the in-memory segment-range cache can never
    /// disagree.
    fn buffered_doc_id_range(&self) -> (u64, u64) {
        let min_id = self
            .buffered_docs
            .iter()
            .map(|(id, _)| *id)
            .min()
            .unwrap_or(0);
        let max_id = self
            .buffered_docs
            .iter()
            .map(|(id, _)| *id)
            .max()
            .unwrap_or(0);
        (min_id, max_id)
    }

    /// Append the segment just written from the current buffer to
    /// [`Self::segment_ranges`] and raise [`Self::max_committed_doc_id`]
    /// (Issue #559 / #864). Callers must invoke this after
    /// [`Self::write_segment_files`] and **before** clearing the buffers the
    /// range is computed from.
    ///
    /// # Parameters
    ///
    /// - `segment_name` - Name the segment's files were written under.
    fn extend_segment_cache(&mut self, segment_name: &str) {
        let (min_id, max_id) = self.buffered_doc_id_range();
        self.max_committed_doc_id = self.max_committed_doc_id.max(max_id);
        self.segment_ranges
            .push((segment_name.to_string(), min_id, max_id));
    }

    /// Write segment metadata.
    fn write_segment_metadata(&self, segment_name: &str) -> Result<()> {
        let (min_id, max_id) = self.buffered_doc_id_range();

        // Create SegmentInfo
        let info = SegmentInfo {
            segment_id: segment_name.to_string(),
            doc_count: self.buffered_docs.len() as u64,
            min_doc_id: min_id,
            max_doc_id: max_id,
            generation: self.current_segment as u64,
            has_deletions: false, // New segments initially have no deletions
            shard_id: self.config.shard_id,
        };

        // Write as JSON for compatibility with InvertedIndex::load_segments()
        let meta_file = format!("{segment_name}.meta");
        let json_data = serde_json::to_string_pretty(&info).map_err(|e| {
            LaurusError::index(format!("Failed to serialize segment metadata: {e}"))
        })?;

        let mut output = self.storage.create_output(&meta_file)?;
        std::io::Write::write_all(&mut output, json_data.as_bytes())?;
        output.close()?;

        Ok(())
    }

    /// Commit all pending changes.
    pub fn commit(&mut self) -> Result<()> {
        self.check_closed()?;

        // Flush any remaining documents
        if !self.buffered_docs.is_empty() {
            self.flush_segment()?;
        }

        // Write index metadata
        self.write_index_metadata()?;
        self.write_metadata_json()?;

        Ok(())
    }

    /// Write global index metadata.
    fn write_index_metadata(&self) -> Result<()> {
        let meta_output = self.storage.create_output("index.meta")?;
        let mut meta_writer = StructWriter::new(meta_output);

        meta_writer.write_u32(0x494D4554)?; // Magic "IMET"
        meta_writer.write_u32(1)?; // Version
        meta_writer.write_u64(crate::util::time::now_secs())?; // Timestamp
        meta_writer.write_u64(self.stats.docs_added)?;
        meta_writer.write_u32(self.stats.segments_created)?;

        meta_writer.close()?;
        Ok(())
    }

    /// Write metadata.json (used by InvertedIndex).
    fn write_metadata_json(&self) -> Result<()> {
        let mut meta = self.base_metadata.clone();
        meta.doc_count += self.stats.docs_added;
        meta.deleted_count += self.stats.deleted_count;
        meta.modified = crate::util::time::now_secs();
        meta.generation += 1; // Increment generation
        meta.last_wal_seq = self.last_wal_seq;

        let metadata_json = serde_json::to_string_pretty(&meta)
            .map_err(|e| LaurusError::index(format!("Failed to serialize metadata: {e}")))?;

        let mut output = self.storage.create_output("metadata.json")?;
        std::io::Write::write_all(&mut output, metadata_json.as_bytes())?;
        output.close()?;
        Ok(())
    }

    /// Rollback all pending changes.
    pub fn rollback(&mut self) -> Result<()> {
        self.check_closed()?;

        // Clear all buffers
        self.buffered_docs.clear();
        self.buffered_doc_ids.clear();
        self.index_dirty = false;
        self.inverted_index = TermPostingIndex::new();

        Ok(())
    }

    /// Get writer statistics.
    pub fn stats(&self) -> &WriterStats {
        &self.stats
    }

    /// Close the writer.
    pub fn close(&mut self) -> Result<()> {
        if !self.closed {
            self.commit()?;
            self.closed = true;
        }
        Ok(())
    }

    /// Check if the writer is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed {
            Err(LaurusError::index("Writer is closed"))
        } else {
            Ok(())
        }
    }

    /// Get the number of pending documents.
    pub fn pending_docs(&self) -> usize {
        self.buffered_docs.len()
    }

    /// Check if the writer is closed.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Remove a pending document with the given ID from in-memory buffers and rebuild indices.
    fn remove_pending_document(&mut self, doc_id: u64) -> Result<()> {
        // Fast path: nothing buffered.
        if self.buffered_docs.is_empty() {
            return Ok(());
        }

        // O(1) membership probe. Newly assigned doc IDs (the common `add`
        // case) are never already buffered, so this returns early and skips
        // the full-buffer scan + index rebuild below — turning the per-upsert
        // cost from O(N) into O(1) and the `add × N` ingest from O(N²) into
        // O(N) (Issue #570). `remove` also drops the id, keeping the set in
        // sync with the `retain` that follows.
        if !self.buffered_doc_ids.remove(&doc_id) {
            return Ok(());
        }

        // The id was buffered: drop it from the buffer (cheap, order-preserving
        // so postings stay doc-id-ascending for the skip-table encode) and
        // **defer** the expensive in-memory index / DocValues rebuild. The
        // rebuild runs once at flush time (and eagerly on the same-id re-upsert
        // path in `upsert_document`), so updating M docs in an N-doc uncommitted
        // buffer is O(M)+O(N) instead of O(M·N) (Issue #828). Until the rebuild,
        // the in-memory index still holds this doc's postings; the NRT lookups
        // filter them out via `buffered_doc_ids`.
        self.buffered_docs.retain(|(id, _)| *id != doc_id);
        self.index_dirty = true;

        // Decrement docs_added for the removed (un-done) document.
        if self.stats.docs_added > 0 {
            self.stats.docs_added -= 1;
        }
        Ok(())
    }

    /// Rebuild the in-memory index and DocValues from buffered docs (used after removals).
    fn rebuild_in_memory_index(&mut self) -> Result<()> {
        // Reset structures
        self.inverted_index = TermPostingIndex::new();
        let segment_name = format!("{}_{:06}", self.config.segment_prefix, self.current_segment);
        self.doc_values_writer = DocValuesWriter::new(self.storage.clone(), segment_name);

        // Reset stats counters that depend on buffered content
        // Do NOT reset docs_added here, as it includes flushed docs.
        // docs_added is adjusted in remove_pending_document directly.
        self.stats.unique_terms = 0;
        self.stats.total_postings = 0;

        // Re-add all buffered analyzed docs
        let buffered_snapshot = self.buffered_docs.clone();
        for (id, analyzed_doc) in buffered_snapshot {
            // Re-add stored fields to DocValues
            for (field_name, value) in &analyzed_doc.stored_fields {
                self.doc_values_writer
                    .add_value(id, field_name, value.clone());
            }

            // Re-add postings
            self.add_analyzed_document_to_index(id, &analyzed_doc)?;
            // stats.docs_added is ALREADY accounting for these docs (except the one removed)
        }

        Ok(())
    }

    /// Mark a persisted document as deleted.
    ///
    /// This updates the deletion bitmap for the segment containing the document.
    fn mark_persisted_doc_deleted(&mut self, doc_id: u64) -> Result<()> {
        let segments = self.find_segments_for_doc(doc_id)?;

        if !segments.is_empty() {
            // Create the deletion manager on first use and keep it for the
            // writer's lifetime (Issue #571): construction reloads every
            // `.delmap` bitmap from storage, and the fresh-id ingest path
            // (empty `segments`) never needs it at all.
            if self.deletion_manager.is_none() {
                self.deletion_manager = Some(crate::maintenance::deletion::DeletionManager::new(
                    Default::default(), // Use default config for now
                    self.storage.clone(),
                )?);
            }
            let mut deleted = 0;
            for (segment_id, min_doc_id, max_doc_id) in &segments {
                let manager = self.deletion_manager.as_ref().ok_or_else(|| {
                    LaurusError::internal("deletion manager missing after initialization")
                })?;

                manager.initialize_segment(segment_id, *min_doc_id, *max_doc_id)?;

                let delete_result = manager.delete_document(segment_id, doc_id, "upsert");
                if delete_result.is_err() {
                    // If initializing failed (e.g. bitmap corrupted), try force re-init
                    // In production code we should be more careful, but here we prioritize consistency
                    manager.initialize_segment(segment_id, *min_doc_id, *max_doc_id)?;
                    manager.delete_document(segment_id, doc_id, "upsert")?;
                }

                // Update segment metadata to reflect deletions
                self.update_segment_meta_deletions(segment_id)?;

                deleted += 1;
            }
            // Track globally
            self.stats.deleted_count += deleted;
        }

        // Add to pending deletions for NRT visibility
        self.pending_deletions.insert(doc_id);

        Ok(())
    }

    /// Check if a document is marked as deleted in the pending set.
    pub fn is_updated_deleted(&self, doc_id: u64) -> bool {
        self.pending_deletions.contains(&doc_id)
    }

    /// Find all segments containing the global doc_id.
    /// Returns a list of (segment_id, min_doc_id, max_doc_id).
    ///
    /// Served from [`Self::segment_ranges`] (Issue #559 / #864) — the cache
    /// mirrors the on-storage `*.meta` files, so this no longer lists and
    /// JSON-parses them per call. Fresh doc IDs (the steady-state ingest
    /// path) are rejected with a single compare against
    /// [`Self::max_committed_doc_id`].
    fn find_segments_for_doc(&self, doc_id: u64) -> Result<Vec<(String, u64, u64)>> {
        // Fast path: WAL doc IDs are monotonic, so an ID above every
        // committed segment's max cannot be in any of them.
        if doc_id > self.max_committed_doc_id {
            return Ok(Vec::new());
        }
        // In Stable ID mode, we check if the ID is within the min/max range.
        // Note: This might match multiple segments if ranges overlap across shards,
        // or if we have multiple versions of the same document (upserts).
        // To be 100% sure, we should check if the document actually exists in
        // the segment. For now, assume the range is specific enough.
        Ok(self
            .segment_ranges
            .iter()
            .filter(|(_, min_doc_id, max_doc_id)| doc_id >= *min_doc_id && doc_id <= *max_doc_id)
            .cloned()
            .collect())
    }

    /// Rebuild [`Self::segment_ranges`] / [`Self::max_committed_doc_id`] from
    /// the `*.meta` files on storage and drop the cached
    /// [`Self::deletion_manager`].
    ///
    /// Must be called after an external segment rewrite that this writer did
    /// not perform itself — today that is only
    /// [`LexicalStore::optimize`](crate::lexical::store::LexicalStore::optimize)'s
    /// force-merge, which replaces every segment (and its deletion bitmap)
    /// behind a live writer. `LexicalStore::commit` needs no call: it drops
    /// the cached writer before merging, so the next writer rebuilds the
    /// cache in its constructor.
    ///
    /// # Errors
    ///
    /// Returns an error if listing the storage fails; malformed `.meta` files
    /// are skipped, matching the constructor's recovery scan.
    pub fn invalidate_segment_cache(&mut self) -> Result<()> {
        // Build the new view first and swap it in only on success: clearing
        // eagerly and then failing (e.g. on `list_files`) would leave the
        // writer alive with an EMPTY cache and `max_committed_doc_id == 0`,
        // silently skipping every subsequent overwrite's deletion via the
        // fast path — worse than the error itself.
        let mut segment_ranges = Vec::new();
        let mut max_committed_doc_id = 0u64;
        for file in self.storage.list_files()? {
            if !file.ends_with(".meta") || file == "index.meta" {
                continue;
            }
            let Ok(input) = self.storage.open_input(&file) else {
                continue;
            };
            let Ok(meta) = serde_json::from_reader::<_, SegmentInfo>(input) else {
                continue;
            };
            max_committed_doc_id = max_committed_doc_id.max(meta.max_doc_id);
            segment_ranges.push((meta.segment_id, meta.min_doc_id, meta.max_doc_id));
        }
        self.segment_ranges = segment_ranges;
        self.max_committed_doc_id = max_committed_doc_id;
        self.deletion_manager = None;
        Ok(())
    }

    /// Rewrite segment metadata to mark `has_deletions = true`.
    fn update_segment_meta_deletions(&self, segment_id: &str) -> Result<()> {
        let meta_file = format!("{segment_id}.meta");
        let input = self.storage.open_input(&meta_file)?;
        let mut meta: SegmentInfo = serde_json::from_reader(input)
            .map_err(|e| LaurusError::index(format!("Failed to read segment meta: {e}")))?;

        if !meta.has_deletions {
            meta.has_deletions = true;
            let json = serde_json::to_string_pretty(&meta).map_err(|e| {
                LaurusError::index(format!("Failed to serialize segment meta: {e}"))
            })?;
            let mut output = self.storage.create_output(&meta_file)?;
            std::io::Write::write_all(&mut output, json.as_bytes())?;
            output.close()?;
        }

        Ok(())
    }

    /// Delete a document by ID.
    ///
    /// Removes the document from the buffered documents if it exists. The
    /// in-memory inverted index and DocValues are rebuilt from the remaining
    /// buffered docs so the deleted doc's postings and stored fields do not
    /// leak into the next flushed segment. For documents that have already
    /// been committed to disk, the deletion is recorded through the
    /// `DeletionManager` (i.e. soft-deleted via the segment's deletion
    /// bitmap).
    ///
    /// This mirrors the pattern used by [`Self::upsert_document`], which also
    /// runs `remove_pending_document` + `mark_persisted_doc_deleted` before
    /// indexing the new version. Deleting via `buffered_docs.retain` alone
    /// (the previous behaviour) left the doc's postings in the in-memory
    /// inverted index, so subsequent puts of the same external id within the
    /// same uncommitted batch would accumulate ghost postings — see the
    /// regression test `delete_document_clears_inverted_index_postings`.
    pub fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        self.remove_pending_document(doc_id)?;
        self.mark_persisted_doc_deleted(doc_id)?;
        Ok(())
    }
}

impl Drop for InvertedIndexWriter {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

// Implement LexicalIndexWriter trait for compatibility with existing code
impl LexicalIndexWriter for InvertedIndexWriter {
    fn add_document(&mut self, doc: Document) -> Result<u64> {
        InvertedIndexWriter::add_document(self, doc)
    }

    fn invalidate_segment_cache(&mut self) -> Result<()> {
        InvertedIndexWriter::invalidate_segment_cache(self)
    }

    fn upsert_document(&mut self, doc_id: u64, doc: Document) -> Result<()> {
        InvertedIndexWriter::upsert_document(self, doc_id, doc)
    }

    fn add_analyzed_document(&mut self, doc: AnalyzedDocument) -> Result<u64> {
        InvertedIndexWriter::add_analyzed_document(self, doc)
    }

    fn upsert_analyzed_document(&mut self, doc_id: u64, doc: AnalyzedDocument) -> Result<()> {
        InvertedIndexWriter::upsert_analyzed_document(self, doc_id, doc)
    }

    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        InvertedIndexWriter::delete_document(self, doc_id)
    }

    fn commit(&mut self) -> Result<()> {
        InvertedIndexWriter::commit(self)
    }

    fn rollback(&mut self) -> Result<()> {
        InvertedIndexWriter::rollback(self)
    }

    fn pending_docs(&self) -> u64 {
        InvertedIndexWriter::pending_docs(self) as u64
    }

    fn close(&mut self) -> Result<()> {
        InvertedIndexWriter::close(self)
    }

    fn is_closed(&self) -> bool {
        InvertedIndexWriter::is_closed(self)
    }

    fn set_last_wal_seq(&mut self, seq: u64) -> Result<()> {
        self.last_wal_seq = seq;
        Ok(())
    }

    fn is_updated_deleted(&self, doc_id: u64) -> bool {
        InvertedIndexWriter::is_updated_deleted(self, doc_id)
    }

    /// Builds an InvertedIndexReader from the current state of the writer's storage.
    /// This method is intended to be called by the LexicalIndexWriter trait implementation.
    fn build_reader(
        &self,
    ) -> Result<std::sync::Arc<dyn crate::lexical::reader::LexicalIndexReader>> {
        use crate::lexical::index::inverted::reader::{
            InvertedIndexReader, InvertedIndexReaderConfig,
        };
        use crate::lexical::index::inverted::segment::SegmentInfo;

        // List all segments from storage
        // This assumes standard segment naming: segment_XXXXXX.meta
        let mut segments = Vec::new();
        let mut segment_id = 0;

        loop {
            let segment_name = format!("{}_{:06}", self.config.segment_prefix, segment_id);
            let meta_file = format!("{}.meta", segment_name);

            if self.storage.file_exists(&meta_file) {
                // Read segment metadata
                let input = self.storage.open_input(&meta_file)?;
                let mut json_data = String::new();
                std::io::Read::read_to_string(&mut std::io::BufReader::new(input), &mut json_data)?;

                let segment_info: SegmentInfo = serde_json::from_str(&json_data).map_err(|e| {
                    LaurusError::index(format!("Failed to parse segment metadata: {e}"))
                })?;

                segments.push(segment_info);
                segment_id += 1;
            } else {
                break;
            }
        }

        let config = InvertedIndexReaderConfig {
            analyzer: self.config.analyzer.clone(),
            ..Default::default()
        };

        // Note: InvertedIndexReader::new expects Vec<SegmentInfo> and Arc<dyn Storage>
        // We use the same storage as the writer
        let reader = InvertedIndexReader::new(segments, self.storage.clone(), config)?;
        Ok(Arc::new(reader))
    }

    fn next_doc_id(&self) -> u64 {
        self.next_doc_id
    }

    fn find_doc_id_by_term(&self, field: &str, term: &str) -> Result<Option<u64>> {
        InvertedIndexWriter::find_doc_id_by_term(self, field, term)
    }

    fn find_doc_ids_by_term(&self, field: &str, term: &str) -> Result<Option<Vec<u64>>> {
        InvertedIndexWriter::find_doc_ids_by_term(self, field, term)
    }
}
