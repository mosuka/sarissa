//! Inverted index writer implementation.
//!
//! This module provides the writer for building inverted indexes in schema-less mode.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use ahash::AHashMap;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::per_field::PerFieldAnalyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::analysis::token::Token;
use crate::document::analyzed::{AnalyzedDocument, AnalyzedTerm};
use crate::document::document::Document;
use crate::document::field_value::FieldValue;
use crate::error::{Result, YatagarasuError};
use crate::lexical::core::dictionary::{TermDictionaryBuilder, TermInfo};
use crate::lexical::core::doc_values::DocValuesWriter;
use crate::lexical::index::inverted::core::posting::{Posting, TermPostingIndex};
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
        // Create initial DocValuesWriter (will be reset per segment)
        let initial_segment_name = format!("{}_{:06}", config.segment_prefix, 0);
        let doc_values_writer = DocValuesWriter::new(storage.clone(), initial_segment_name);

        Ok(InvertedIndexWriter {
            storage,
            config,
            inverted_index: TermPostingIndex::new(),
            buffered_docs: Vec::new(),
            doc_values_writer,
            next_doc_id: 0,
            current_segment: 0,
            closed: false,
            stats: WriterStats {
                docs_added: 0,
                unique_terms: 0,
                total_postings: 0,
                memory_used: 0,
                segments_created: 0,
            },
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

    /// Add a document to the index with a specific document ID.
    pub fn add_document_with_id(&mut self, doc_id: u64, doc: Document) -> Result<()> {
        self.check_closed()?;

        // Analyze the document
        let analyzed_doc = self.analyze_document(doc)?;

        // Add the analyzed document with the specified ID
        self.add_analyzed_document_with_id(doc_id, analyzed_doc)
    }

    /// Add an already analyzed document to the index with a specific document ID.
    pub fn add_analyzed_document_with_id(
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
    /// use yatagarasu::document::document::Document;
    /// use yatagarasu::document::parser::DocumentParser;
    /// use yatagarasu::analysis::analyzer::per_field::PerFieldAnalyzer;
    /// use yatagarasu::analysis::analyzer::standard::StandardAnalyzer;
    /// use yatagarasu::lexical::index::inverted::writer::{InvertedIndexWriter, InvertedIndexWriterConfig};
    /// use yatagarasu::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use yatagarasu::storage::StorageConfig;
    /// use std::sync::Arc;
    ///
    /// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    /// let mut per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
    /// let config = InvertedIndexWriterConfig {
    ///     analyzer: Arc::new(per_field.clone()),
    ///     ..Default::default()
    /// };
    /// let mut writer = InvertedIndexWriter::new(storage, config).unwrap();
    ///
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

        // Assign document ID
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        // Add the analyzed document with the assigned ID
        self.add_analyzed_document_with_id(doc_id, analyzed_doc)?;

        Ok(doc_id)
    }

    /// Analyze a document into terms.
    fn analyze_document(&mut self, doc: Document) -> Result<AnalyzedDocument> {
        let mut field_terms = AHashMap::new();
        let mut stored_fields = AHashMap::new();

        // Process each field in the document (schema-less mode)
        for (field_name, field_value) in doc.fields() {
            use crate::document::field_value::FieldValue;

            match field_value {
                FieldValue::Text(text) => {
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
                    stored_fields.insert(field_name.clone(), FieldValue::Text(text.to_string()));
                }
                FieldValue::Integer(num) => {
                    // Convert integer to text for indexing
                    let text = num.to_string();

                    let analyzed_term = AnalyzedTerm {
                        term: text.clone(),
                        position: 0,
                        frequency: 1,
                        offset: (0, text.len()),
                    };

                    field_terms.insert(field_name.clone(), vec![analyzed_term]);
                    stored_fields.insert(field_name.clone(), FieldValue::Integer(*num));
                }
                FieldValue::Float(num) => {
                    // Convert float to text for indexing
                    let text = num.to_string();

                    let analyzed_term = AnalyzedTerm {
                        term: text.clone(),
                        position: 0,
                        frequency: 1,
                        offset: (0, text.len()),
                    };

                    field_terms.insert(field_name.clone(), vec![analyzed_term]);
                    stored_fields.insert(field_name.clone(), FieldValue::Float(*num));
                }
                FieldValue::Boolean(boolean) => {
                    // Convert boolean to text
                    let text = boolean.to_string();

                    let analyzed_term = AnalyzedTerm {
                        term: text.clone(),
                        position: 0,
                        frequency: 1,
                        offset: (0, text.len()),
                    };

                    field_terms.insert(field_name.clone(), vec![analyzed_term]);
                    stored_fields.insert(field_name.clone(), FieldValue::Boolean(*boolean));
                }
                FieldValue::DateTime(dt) => {
                    // Handle DateTime field
                    let text = dt.to_rfc3339();

                    let analyzed_term = AnalyzedTerm {
                        term: text.clone(),
                        position: 0,
                        frequency: 1,
                        offset: (0, text.len()),
                    };

                    field_terms.insert(field_name.clone(), vec![analyzed_term]);
                    stored_fields.insert(field_name.clone(), FieldValue::DateTime(*dt));
                }
                FieldValue::Binary(_) | FieldValue::Geo(_) | FieldValue::Null => {
                    // For Binary, Geo, and Null types, only store but don't index
                    stored_fields.insert(field_name.clone(), field_value.clone());
                }
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
    fn add_analyzed_document_to_index(&mut self, doc_id: u64, doc: &AnalyzedDocument) -> Result<()> {
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
                    FieldValue::Text(text) => {
                        stored_writer.write_u8(0)?; // Type tag for Text
                        stored_writer.write_string(text)?;
                    }
                    FieldValue::Integer(num) => {
                        stored_writer.write_u8(1)?; // Type tag for Integer
                        stored_writer.write_u64(*num as u64)?; // Store as u64, preserving bit pattern
                    }
                    FieldValue::Float(num) => {
                        stored_writer.write_u8(2)?; // Type tag for Float
                        stored_writer.write_f64(*num)?;
                    }
                    FieldValue::Boolean(b) => {
                        stored_writer.write_u8(3)?; // Type tag for Boolean
                        stored_writer.write_u8(if *b { 1 } else { 0 })?;
                    }
                    FieldValue::Binary(bytes) => {
                        stored_writer.write_u8(4)?; // Type tag for Binary
                        stored_writer.write_varint(bytes.len() as u64)?;
                        stored_writer.write_bytes(bytes)?;
                    }
                    FieldValue::DateTime(dt) => {
                        stored_writer.write_u8(5)?; // Type tag for DateTime
                        stored_writer.write_string(&dt.to_rfc3339())?;
                    }
                    FieldValue::Geo(geo) => {
                        stored_writer.write_u8(6)?; // Type tag for Geo
                        stored_writer.write_f64(geo.lat)?;
                        stored_writer.write_f64(geo.lon)?;
                    }
                    FieldValue::Null => {
                        stored_writer.write_u8(7)?; // Type tag for Null
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
                doc.add_field(field_name, field_value.clone());
            }
            documents.push(doc);
        }

        // Write as JSON
        let json_file = format!("{segment_name}.json");
        let mut output = self.storage.create_output(&json_file)?;
        let segment_data = serde_json::to_string_pretty(&documents)
            .map_err(|e| YatagarasuError::index(format!("Failed to serialize segment: {e}")))?;
        std::io::Write::write_all(&mut output, segment_data.as_bytes())?;
        output.close()?;

        Ok(())
    }

    /// Write segment metadata.
    fn write_segment_metadata(&self, segment_name: &str) -> Result<()> {
        use crate::lexical::index::inverted::segment::SegmentInfo;

        // Create SegmentInfo
        let segment_info = SegmentInfo {
            segment_id: segment_name.to_string(),
            doc_count: self.buffered_docs.len() as u64,
            doc_offset: self.next_doc_id - self.buffered_docs.len() as u64,
            generation: self.current_segment as u64,
            has_deletions: false,
        };

        // Write as JSON for compatibility with InvertedIndex::load_segments()
        let meta_file = format!("{segment_name}.meta");
        let json_data = serde_json::to_string_pretty(&segment_info).map_err(|e| {
            YatagarasuError::index(format!("Failed to serialize segment metadata: {e}"))
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

        Ok(())
    }

    /// Write global index metadata.
    fn write_index_metadata(&self) -> Result<()> {
        let meta_output = self.storage.create_output("index.meta")?;
        let mut meta_writer = StructWriter::new(meta_output);

        meta_writer.write_u32(0x494D4554)?; // Magic "IMET"
        meta_writer.write_u32(1)?; // Version
        meta_writer.write_u64(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        )?; // Timestamp
        meta_writer.write_u64(self.stats.docs_added)?;
        meta_writer.write_u32(self.stats.segments_created)?;

        meta_writer.close()?;
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
            Err(YatagarasuError::index("Writer is closed"))
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

    /// Delete documents matching the given term.
    /// Note: This is a simplified implementation for compatibility.
    pub fn delete_documents(&mut self, _field: &str, _value: &str) -> Result<u64> {
        // TODO: Implement proper deletion support
        Ok(0)
    }

    /// Update a document (delete old, add new).
    pub fn update_document(&mut self, field: &str, value: &str, doc: Document) -> Result<()> {
        self.delete_documents(field, value)?;
        self.add_document(doc)?;
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

    fn add_document_with_id(&mut self, doc_id: u64, doc: Document) -> Result<()> {
        InvertedIndexWriter::add_document_with_id(self, doc_id, doc)
    }

    fn add_analyzed_document(&mut self, doc: AnalyzedDocument) -> Result<u64> {
        InvertedIndexWriter::add_analyzed_document(self, doc)
    }

    fn add_analyzed_document_with_id(&mut self, doc_id: u64, doc: AnalyzedDocument) -> Result<()> {
        InvertedIndexWriter::add_analyzed_document_with_id(self, doc_id, doc)
    }

    fn delete_documents(&mut self, field: &str, value: &str) -> Result<u64> {
        InvertedIndexWriter::delete_documents(self, field, value)
    }

    fn update_document(&mut self, field: &str, value: &str, doc: Document) -> Result<()> {
        InvertedIndexWriter::update_document(self, field, value, doc)
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
}
