//! Inverted index writer implementation.
//!
//! This module provides the writer for building inverted indexes in schema-less mode.

use std::collections::HashMap;
use std::sync::Arc;

use ahash::AHashMap;

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
        // Recover state from existing segments
        let mut next_doc_id = 0;
        let mut max_segment_id = -1i32;

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

        // Upsert: remove any pending document with the same ID before adding
        self.remove_pending_document(doc_id)?;
        // Upsert: mark persisted occurrences as deleted (flushed segments)
        self.mark_persisted_doc_deleted(doc_id)?;

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

        // 1. Check in-memory inverted index
        if let Some(posting_list) = self.inverted_index.get_posting_list(&full_term) {
            // Get the last document ID (most recent version if multiple exist, though upsert usually handles that)
            // Posting list is sorted by doc_id? Usually yes for inverted index.
            // But TermPostingIndex might just append.
            // Let's assume the last one is the latest.
            if let Some(last_posting) = posting_list.postings.last() {
                return Ok(Some(last_posting.doc_id));
            }
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

        // 1. Check in-memory inverted index
        if let Some(posting_list) = self.inverted_index.get_posting_list(&full_term) {
            let ids: Vec<u64> = posting_list.postings.iter().map(|p| p.doc_id).collect();
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
                        point_values.insert(field_name.clone(), vec![*num as f64]);
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
                        point_values.insert(field_name.clone(), vec![*num]);
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
                        point_values.insert(field_name.clone(), vec![ts]);
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
                    DataValue::Geo(lat, lon) => {
                        // Geo points are indexed as 2D points in BKD
                        field_terms.insert(
                            field_name.clone(),
                            vec![AnalyzedTerm {
                                term: format!("{},{}", lat, lon),
                                position: 0,
                                frequency: 1,
                                offset: (0, format!("{},{}", lat, lon).len()),
                            }],
                        );
                        point_values.insert(field_name.clone(), vec![*lat, *lon]);
                    }
                    // Handle other variants...
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

        let segment_name = format!("{}_{:06}", self.config.segment_prefix, self.current_segment);

        // Write inverted index
        self.write_inverted_index(&segment_name)?;

        // Write stored documents
        self.write_stored_documents(&segment_name)?;

        // Write field lengths
        self.write_field_lengths(&segment_name)?;

        // Write field statistics
        self.write_field_stats(&segment_name)?;

        // Write DocValues
        self.write_doc_values(&segment_name)?;

        // Write segment metadata
        self.write_segment_metadata(&segment_name)?;

        // Write BKD trees for numeric fields
        self.write_bkd_trees(&segment_name)?;

        // COMPATIBILITY: Also write documents as JSON for BasicIndexReader
        self.write_json_documents(&segment_name)?;

        // Clear buffers
        self.buffered_docs.clear();
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

    /// Write the inverted index to storage.
    fn write_inverted_index(&self, segment_name: &str) -> Result<()> {
        // Write posting lists
        let posting_file = format!("{segment_name}.post");
        let posting_output = self.storage.create_output(&posting_file)?;
        let mut posting_writer = StructWriter::new(posting_output);

        let mut term_dict_builder = TermDictionaryBuilder::new();

        // Collect and sort terms for deterministic output
        let mut terms: Vec<_> = self.inverted_index.terms().collect();
        terms.sort();

        for term in terms {
            if let Some(posting_list) = self.inverted_index.get_posting_list(term) {
                let start_offset = posting_writer.position();

                // Write posting list
                posting_list.encode(&mut posting_writer)?;

                let end_offset = posting_writer.position();
                let length = end_offset - start_offset;

                // Add to term dictionary
                let term_info = TermInfo::new(
                    start_offset,
                    length,
                    posting_list.doc_frequency,
                    posting_list.total_frequency,
                );
                term_dict_builder.add_term(term.clone(), term_info);
            }
        }

        posting_writer.close()?;

        // Write term dictionary
        let dict_file = format!("{segment_name}.dict");
        let dict_output = self.storage.create_output(&dict_file)?;
        let mut dict_writer = StructWriter::new(dict_output);

        let term_dict = term_dict_builder.build_hybrid();
        term_dict.write_to_storage(&mut dict_writer)?;
        dict_writer.close()?;

        Ok(())
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
                    crate::data::DataValue::Geo(lat, lon) => {
                        stored_writer.write_u8(6)?; // Type tag for Geo
                        stored_writer.write_f64(*lat)?;
                        stored_writer.write_f64(*lon)?;
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
    fn write_doc_values(&self, _segment_name: &str) -> Result<()> {
        // DocValues are written using local filesystem approach
        // since Storage trait doesn't directly support it yet.
        // We'll write to a temporary location and then upload if needed.

        // For now, write directly using the doc_values_writer's write method
        self.doc_values_writer.write()?;

        // If using remote storage, we would need to upload the .dv file here
        // For filesystem-based storage, the file is already in the right place

        Ok(())
    }

    /// Write documents as JSON for compatibility with BasicIndexReader.
    fn write_json_documents(&self, segment_name: &str) -> Result<()> {
        // Convert analyzed documents back to Document format with preserved types
        let mut documents = Vec::new();
        for (_doc_id, analyzed_doc) in &self.buffered_docs {
            let mut doc = Document::new();
            for (field_name, field_value) in &analyzed_doc.stored_fields {
                doc.fields.insert(field_name.clone(), field_value.clone());
            }
            documents.push(doc);
        }

        // Write as JSON
        let json_file = format!("{segment_name}.json");
        let mut output = self.storage.create_output(&json_file)?;
        let segment_data = serde_json::to_string_pretty(&documents)
            .map_err(|e| LaurusError::index(format!("Failed to serialize segment: {e}")))?;
        std::io::Write::write_all(&mut output, segment_data.as_bytes())?;
        output.close()?;

        Ok(())
    }

    /// Write BKD trees for numeric and geo fields.
    fn write_bkd_trees(&self, segment_name: &str) -> Result<()> {
        let mut field_points: AHashMap<String, Vec<(Vec<f64>, u64)>> = AHashMap::new();

        for (doc_id, doc) in &self.buffered_docs {
            for (field, values) in &doc.point_values {
                field_points
                    .entry(field.clone())
                    .or_default()
                    .push((values.clone(), *doc_id));
            }
        }

        for (field, points) in field_points {
            if points.is_empty() {
                continue;
            }

            let num_dims = points[0].0.len() as u32;

            let file_name = format!("{segment_name}.{field}.bkd");
            let output = self.storage.create_output(&file_name)?;
            let mut writer = BKDWriter::new(output, num_dims);
            writer.write(&points)?;
            writer.finish()?;
        }
        Ok(())
    }

    /// Write segment metadata.
    fn write_segment_metadata(&self, segment_name: &str) -> Result<()> {
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
        // Fast path: nothing buffered
        if self.buffered_docs.is_empty() {
            return Ok(());
        }

        // Retain only docs with different IDs
        let mut changed = false;
        self.buffered_docs.retain(|(id, _)| {
            let keep = *id != doc_id;
            if !keep {
                changed = true;
            }
            keep
        });

        if !changed {
            return Ok(());
        }

        // Rebuild in-memory inverted index and DocValues from the remaining buffered docs
        self.rebuild_in_memory_index()?;

        if changed {
            // Decrement docs_added for the removed document
            // Note: remove_pending_document logic in upsert implies removing 1 old version
            // But if we just removed it from buffer, we un-did the add.
            if self.stats.docs_added > 0 {
                self.stats.docs_added -= 1;
            }
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

        for (segment_id, min_doc_id, max_doc_id) in segments {
            // Found the segment, update deletion bitmap
            let manager = crate::maintenance::deletion::DeletionManager::new(
                Default::default(), // Use default config for now
                self.storage.clone(),
            )?;

            manager.initialize_segment(&segment_id, min_doc_id, max_doc_id)?;

            let delete_result = manager.delete_document(&segment_id, doc_id, "upsert");
            if delete_result.is_err() {
                // If initializing failed (e.g. bitmap corrupted), try force re-init
                // In production code we should be more careful, but here we prioritize consistency
                manager.initialize_segment(&segment_id, min_doc_id, max_doc_id)?;
                manager.delete_document(&segment_id, doc_id, "upsert")?;
            }

            // Update segment metadata to reflect deletions
            self.update_segment_meta_deletions(&segment_id)?;

            // Track globally
            self.stats.deleted_count += 1;
        }

        // Add to pending deletions for NRT visibility
        self.pending_deletions.insert(doc_id);

        Ok(())
    }

    /// Check if a document is marked as deleted in the pending set.
    pub fn is_updated_deleted(&self, doc_id: u64) -> bool {
        self.pending_deletions.contains(&doc_id)
    }

    /// Find all segments containing the global doc_id by scanning segment metadata files.
    /// Returns a list of (segment_id, min_doc_id, max_doc_id).
    fn find_segments_for_doc(&self, doc_id: u64) -> Result<Vec<(String, u64, u64)>> {
        let mut segments = Vec::new();
        let files = self.storage.list_files()?;
        for file in files {
            if !file.ends_with(".meta") || file == "index.meta" {
                continue;
            }
            let input = match self.storage.open_input(&file) {
                Ok(input) => input,
                Err(_) => continue,
            };
            let meta: SegmentInfo = match serde_json::from_reader(input) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // In Stable ID mode, we check if the ID is within the min/max range.
            // Note: This might match multiple segments if ranges overlap across shards,
            // or if we have multiple versions of the same document (upserts).
            if doc_id >= meta.min_doc_id && doc_id <= meta.max_doc_id {
                // To be 100% sure, we should check if the document actually exists in this segment.
                // For now, assume this range is specific enough.
                segments.push((meta.segment_id.clone(), meta.min_doc_id, meta.max_doc_id));
            }
        }
        Ok(segments)
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
    /// Removes the document from the buffered documents if it exists.
    /// For committed documents, deletion is handled through the DeletionManager.
    pub fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        // Remove from buffered documents if present
        self.buffered_docs.retain(|(id, _)| *id != doc_id);

        // Also mark as deleted in persisted segments
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
