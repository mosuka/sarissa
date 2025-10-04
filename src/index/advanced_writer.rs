//! Advanced index writer with inverted index support.
//!
//! This module provides a production-ready index writer that builds
//! inverted indexes with term dictionaries and posting lists.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use ahash::AHashMap;

use crate::analysis::{Analyzer, Token};
use crate::document::Document;
use crate::error::{Result, SarissaError};
use crate::index::dictionary::{TermDictionaryBuilder, TermInfo};
use crate::index::{InvertedIndex, Posting};
use crate::storage::{Storage, StructWriter};

/// Advanced index writer configuration.
#[derive(Debug, Clone)]
pub struct AdvancedWriterConfig {
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

    /// Default analyzer for text fields.
    pub default_analyzer: String,
}

impl Default for AdvancedWriterConfig {
    fn default() -> Self {
        AdvancedWriterConfig {
            max_buffered_docs: 10000,
            max_buffer_memory: 64 * 1024 * 1024, // 64MB
            segment_prefix: "segment".to_string(),
            store_term_positions: true,
            optimize_segments: false,
            default_analyzer: "standard".to_string(),
        }
    }
}

/// A document with analyzed terms ready for indexing.
#[derive(Debug, Clone)]
struct AnalyzedDocument {
    /// Original document ID.
    doc_id: u64,
    /// Field name to analyzed terms mapping.
    field_terms: AHashMap<String, Vec<AnalyzedTerm>>,
    /// Stored field values.
    stored_fields: AHashMap<String, String>,
}

/// An analyzed term with position and metadata.
#[derive(Debug, Clone)]
struct AnalyzedTerm {
    /// The term text.
    term: String,
    /// Position in the field.
    position: u32,
    /// Term frequency in the document.
    frequency: u32,
    /// Offset in the original text.
    #[allow(dead_code)]
    offset: (usize, usize),
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

/// Advanced index writer implementation (schema-less mode).
pub struct AdvancedIndexWriter {
    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// Writer configuration.
    config: AdvancedWriterConfig,

    /// In-memory inverted index being built.
    inverted_index: InvertedIndex,

    /// Buffered analyzed documents.
    buffered_docs: Vec<AnalyzedDocument>,

    /// Document ID counter.
    next_doc_id: u64,

    /// Current segment number.
    current_segment: u32,

    /// Available analyzers.
    analyzers: AHashMap<String, Box<dyn Analyzer>>,

    /// Whether the writer is closed.
    closed: bool,

    /// Writer statistics.
    stats: WriterStats,
}

impl std::fmt::Debug for AdvancedIndexWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdvancedIndexWriter")
            .field("config", &self.config)
            .field("next_doc_id", &self.next_doc_id)
            .field("current_segment", &self.current_segment)
            .field("closed", &self.closed)
            .field("buffered_docs_count", &self.buffered_docs.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl AdvancedIndexWriter {
    /// Create a new advanced index writer (schema-less mode).
    pub fn new(storage: Arc<dyn Storage>, config: AdvancedWriterConfig) -> Result<Self> {
        let mut analyzers = AHashMap::new();

        // Add default analyzers
        analyzers.insert(
            "standard".to_string(),
            Box::new(crate::analysis::StandardAnalyzer::new()?) as Box<dyn Analyzer>,
        );

        // For SimpleAnalyzer, we need to provide a tokenizer
        let regex_tokenizer = Arc::new(crate::analysis::tokenizer::RegexTokenizer::new()?);
        analyzers.insert(
            "simple".to_string(),
            Box::new(crate::analysis::SimpleAnalyzer::new(regex_tokenizer)) as Box<dyn Analyzer>,
        );

        // For KeywordAnalyzer, use the existing implementation
        analyzers.insert(
            "keyword".to_string(),
            Box::new(crate::analysis::KeywordAnalyzer::new()) as Box<dyn Analyzer>,
        );

        Ok(AdvancedIndexWriter {
            storage,
            config,
            inverted_index: InvertedIndex::new(),
            buffered_docs: Vec::new(),
            next_doc_id: 0,
            current_segment: 0,
            analyzers,
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

    /// Add a document to the index.
    pub fn add_document(&mut self, doc: Document) -> Result<()> {
        self.check_closed()?;

        // Schema-less mode: no validation needed
        // Analyze the document
        let analyzed_doc = self.analyze_document(doc)?;

        // Add to inverted index
        self.add_analyzed_document_to_index(&analyzed_doc)?;

        // Buffer the document
        self.buffered_docs.push(analyzed_doc);
        self.stats.docs_added += 1;

        // Check if we need to flush
        if self.should_flush() {
            self.flush_segment()?;
        }

        Ok(())
    }

    /// Analyze a document into terms.
    fn analyze_document(&mut self, doc: Document) -> Result<AnalyzedDocument> {
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        let mut field_terms = AHashMap::new();
        let mut stored_fields = AHashMap::new();

        // Process each field in the document (schema-less mode)
        for (field_name, field_value) in doc.fields() {
            use crate::document::FieldValue;

            match field_value {
                FieldValue::Text(text) => {
                    // Analyze text field
                    let analyzer_name = &self.config.default_analyzer;
                    let analyzer = self.analyzers.get_mut(analyzer_name).ok_or_else(|| {
                        SarissaError::analysis(format!("Unknown analyzer: {analyzer_name}"))
                    })?;

                    let tokens = analyzer.analyze(text)?;
                    let token_vec: Vec<Token> = tokens.collect();
                    let analyzed_terms = self.tokens_to_analyzed_terms(token_vec);

                    field_terms.insert(field_name.clone(), analyzed_terms);
                    stored_fields.insert(field_name.clone(), text.to_string());
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
                    stored_fields.insert(field_name.clone(), text);
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
                    stored_fields.insert(field_name.clone(), text);
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
                    stored_fields.insert(field_name.clone(), text);
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
                    stored_fields.insert(field_name.clone(), text);
                }
                _ => {
                    // For other types (Binary, Geo, Null), skip indexing
                }
            }
        }

        Ok(AnalyzedDocument {
            doc_id,
            field_terms,
            stored_fields,
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
    fn add_analyzed_document_to_index(&mut self, doc: &AnalyzedDocument) -> Result<()> {
        for (field_name, terms) in &doc.field_terms {
            for analyzed_term in terms {
                let full_term = format!("{field_name}:{}", analyzed_term.term);

                let posting = if self.config.store_term_positions {
                    Posting::with_positions(doc.doc_id, vec![analyzed_term.position])
                } else {
                    Posting::with_frequency(doc.doc_id, analyzed_term.frequency)
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

        // Write segment metadata
        self.write_segment_metadata(&segment_name)?;

        // COMPATIBILITY: Also write documents as JSON for BasicIndexReader
        self.write_json_documents(&segment_name)?;

        // Clear buffers
        self.buffered_docs.clear();
        self.inverted_index = InvertedIndex::new();
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

    /// Write stored documents to storage.
    fn write_stored_documents(&self, segment_name: &str) -> Result<()> {
        let stored_file = format!("{segment_name}.docs");
        let stored_output = self.storage.create_output(&stored_file)?;
        let mut stored_writer = StructWriter::new(stored_output);

        // Write document count
        stored_writer.write_varint(self.buffered_docs.len() as u64)?;

        // Write each document
        for doc in &self.buffered_docs {
            stored_writer.write_u64(doc.doc_id)?;
            stored_writer.write_varint(doc.stored_fields.len() as u64)?;

            for (field_name, field_value) in &doc.stored_fields {
                stored_writer.write_string(field_name)?;
                stored_writer.write_string(field_value)?;
            }
        }

        stored_writer.close()?;
        Ok(())
    }

    /// Write documents as JSON for compatibility with BasicIndexReader.
    fn write_json_documents(&self, segment_name: &str) -> Result<()> {
        use crate::document::FieldValue;

        // Convert analyzed documents back to Document format
        let mut documents = Vec::new();
        for analyzed_doc in &self.buffered_docs {
            let mut doc = Document::new();
            for (field_name, field_value) in &analyzed_doc.stored_fields {
                doc.add_field(field_name, FieldValue::Text(field_value.clone()));
            }
            documents.push(doc);
        }

        // Write as JSON
        let json_file = format!("{segment_name}.json");
        let mut output = self.storage.create_output(&json_file)?;
        let segment_data = serde_json::to_string_pretty(&documents)
            .map_err(|e| SarissaError::index(format!("Failed to serialize segment: {e}")))?;
        std::io::Write::write_all(&mut output, segment_data.as_bytes())?;
        output.close()?;

        Ok(())
    }

    /// Write segment metadata.
    fn write_segment_metadata(&self, segment_name: &str) -> Result<()> {
        use crate::index::SegmentInfo;

        // Create SegmentInfo
        let segment_info = SegmentInfo {
            segment_id: segment_name.to_string(),
            doc_count: self.buffered_docs.len() as u64,
            doc_offset: self.next_doc_id - self.buffered_docs.len() as u64,
            generation: self.current_segment as u64,
            has_deletions: false,
        };

        // Write as JSON for compatibility with FileIndex::load_segments()
        let meta_file = format!("{segment_name}.meta");
        let json_data = serde_json::to_string_pretty(&segment_info)
            .map_err(|e| SarissaError::index(format!("Failed to serialize segment metadata: {e}")))?;

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
        self.inverted_index = InvertedIndex::new();

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
            Err(SarissaError::index("Writer is closed"))
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

impl Drop for AdvancedIndexWriter {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

// Implement IndexWriter trait for compatibility with existing code
impl crate::index::writer::IndexWriter for AdvancedIndexWriter {
    fn add_document(&mut self, doc: Document) -> Result<()> {
        AdvancedIndexWriter::add_document(self, doc)
    }

    fn delete_documents(&mut self, field: &str, value: &str) -> Result<u64> {
        AdvancedIndexWriter::delete_documents(self, field, value)
    }

    fn update_document(&mut self, field: &str, value: &str, doc: Document) -> Result<()> {
        AdvancedIndexWriter::update_document(self, field, value, doc)
    }

    fn commit(&mut self) -> Result<()> {
        AdvancedIndexWriter::commit(self)
    }

    fn rollback(&mut self) -> Result<()> {
        AdvancedIndexWriter::rollback(self)
    }

    fn pending_docs(&self) -> u64 {
        AdvancedIndexWriter::pending_docs(self) as u64
    }

    fn close(&mut self) -> Result<()> {
        AdvancedIndexWriter::close(self)
    }

    fn is_closed(&self) -> bool {
        AdvancedIndexWriter::is_closed(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::storage::{MemoryStorage, StorageConfig};
    use std::sync::Arc;

    #[allow(dead_code)]
    fn create_test_document(title: &str, body: &str) -> Document {
        Document::builder()
            .add_text("title", title)
            .add_text("body", body)
            .build()
    }

    #[test]
    fn test_advanced_writer_creation() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = AdvancedWriterConfig::default();

        let writer = AdvancedIndexWriter::new(storage, config).unwrap();

        assert_eq!(writer.pending_docs(), 0);
        assert_eq!(writer.stats().docs_added, 0);
    }

    #[test]
    fn test_add_document() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = AdvancedWriterConfig::default();

        let mut writer = AdvancedIndexWriter::new(storage, config).unwrap();
        let doc = create_test_document("Test Title", "This is test content");

        writer.add_document(doc).unwrap();

        assert_eq!(writer.pending_docs(), 1);
        assert_eq!(writer.stats().docs_added, 1);
        assert!(writer.stats().unique_terms > 0);
    }

    #[test]
    fn test_auto_flush() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = AdvancedWriterConfig {
            max_buffered_docs: 2,
            ..Default::default()
        };

        let mut writer = AdvancedIndexWriter::new(storage.clone(), config).unwrap();

        // Add first document
        writer
            .add_document(create_test_document("Doc 1", "Content 1"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 1);

        // Add second document - should trigger flush
        writer
            .add_document(create_test_document("Doc 2", "Content 2"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 0); // Flushed
        assert_eq!(writer.stats().segments_created, 1);

        // Check that files were created
        let files = storage.list_files().unwrap();
        assert!(files.iter().any(|f| f.contains("segment_000000")));
    }

    #[test]
    fn test_commit() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = AdvancedWriterConfig::default();

        let mut writer = AdvancedIndexWriter::new(storage.clone(), config).unwrap();

        writer
            .add_document(create_test_document("Test", "Content"))
            .unwrap();
        writer.commit().unwrap();

        assert_eq!(writer.pending_docs(), 0);

        // Check that files were created
        let files = storage.list_files().unwrap();
        assert!(files.contains(&"index.meta".to_string()));
        assert!(files.iter().any(|f| f.starts_with("segment_")));
    }

    #[test]
    fn test_rollback() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = AdvancedWriterConfig::default();

        let mut writer = AdvancedIndexWriter::new(storage, config).unwrap();

        writer
            .add_document(create_test_document("Test", "Content"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 1);

        writer.rollback().unwrap();
        assert_eq!(writer.pending_docs(), 0);
        assert_eq!(writer.stats().docs_added, 1); // Stats don't rollback
    }

    #[test]
    fn test_multiple_field_types() {
        // Schema-less mode: fields are inferred from document
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = AdvancedWriterConfig::default();

        let mut writer = AdvancedIndexWriter::new(storage, config).unwrap();

        let doc = Document::builder()
            .add_text("title", "Test Document")
            .add_text("id", "doc1")
            .add_numeric("count", 42.0)
            .build();

        writer.add_document(doc).unwrap();
        writer.commit().unwrap();

        assert_eq!(writer.stats().docs_added, 1);
        assert!(writer.stats().unique_terms >= 3); // At least title, id, count fields
    }
}
