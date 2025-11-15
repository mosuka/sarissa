//! Inverted index reader implementation.
//!
//! This module provides a production-ready inverted index reader that efficiently
//! handles multiple segments, caching, and optimized posting list access.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use ahash::AHashMap;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::analysis::token::Token;
use crate::document::document::Document;
use crate::document::field::FieldValue;
use crate::error::{Result, YatagarasuError};
use crate::lexical::core::dictionary::HybridTermDictionary;
use crate::lexical::core::dictionary::TermInfo;
use crate::lexical::core::doc_values::DocValuesReader;
use crate::lexical::index::inverted::core::posting::{Posting, PostingList};
use crate::lexical::index::inverted::core::terms::{
    InvertedIndexTerms, TermDictionaryAccess, Terms,
};
use crate::lexical::index::inverted::segment::SegmentInfo;
use crate::lexical::reader::FieldStats;
use crate::lexical::reader::PostingIterator;
use crate::storage::Storage;
use crate::storage::structured::StructReader;

/// Advanced index reader configuration.
#[derive(Clone)]
pub struct InvertedIndexReaderConfig {
    /// Maximum memory for caching (in bytes).
    pub max_cache_memory: usize,

    /// Enable term caching.
    pub enable_term_cache: bool,

    /// Enable posting cache.
    pub enable_posting_cache: bool,

    /// Preload segments on open.
    pub preload_segments: bool,

    /// Maximum number of cached terms per field.
    pub max_cached_terms_per_field: usize,

    /// Analyzer for query term analysis.
    pub analyzer: Arc<dyn Analyzer>,
}

impl std::fmt::Debug for InvertedIndexReaderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndexReaderConfig")
            .field("max_cache_memory", &self.max_cache_memory)
            .field("enable_term_cache", &self.enable_term_cache)
            .field("enable_posting_cache", &self.enable_posting_cache)
            .field("preload_segments", &self.preload_segments)
            .field(
                "max_cached_terms_per_field",
                &self.max_cached_terms_per_field,
            )
            .field("analyzer", &self.analyzer.name())
            .finish()
    }
}

impl Default for InvertedIndexReaderConfig {
    fn default() -> Self {
        InvertedIndexReaderConfig {
            max_cache_memory: 128 * 1024 * 1024, // 128MB
            enable_term_cache: true,
            enable_posting_cache: true,
            preload_segments: false,
            max_cached_terms_per_field: 10000,
            analyzer: Arc::new(
                StandardAnalyzer::new().expect("StandardAnalyzer should be creatable"),
            ),
        }
    }
}

/// Advanced posting iterator for efficiently reading postings from the index.
///
/// # Purpose
/// Used when executing queries against the actual index.
///
/// # Implemented Traits
/// - `reader::PostingIterator` trait
///
/// # Features
/// - `next()`: Move to the next document
/// - `skip_to(target)`: Efficiently skip to a specified document ID
/// - Block-based optimization for fast skip operations
/// - Position information retrieval
/// - Cost calculation for optimization
///
/// # Use Cases
/// - Returned as `Box<dyn reader::PostingIterator>` from `InvertedIndexReader.postings()`
/// - Used during query execution (BooleanQuery, FuzzyQuery, etc.)
/// - When efficient processing of multiple query conditions is needed
///
/// # Difference from `posting::PostingIterator`
/// - `posting::PostingIterator`: Simple in-memory iteration
/// - `InvertedIndexPostingIterator`: Advanced iterator for index queries
#[derive(Debug)]
pub struct InvertedIndexPostingIterator {
    /// The posting data.
    postings: Vec<crate::lexical::index::inverted::core::posting::Posting>,

    /// Current position in the posting list.
    position: usize,

    /// Block cache for efficient access.
    block_cache: Option<Vec<PostingBlock>>,

    /// Current block being processed.
    current_block: usize,

    /// Whether next() has been called at least once.
    started: bool,
}

/// A block of postings for efficient processing.
#[derive(Debug, Clone)]
pub struct PostingBlock {
    /// Minimum document ID in this block.
    pub min_doc_id: u64,

    /// Maximum document ID in this block.
    pub max_doc_id: u64,

    /// Starting position in the posting list.
    pub start_position: usize,

    /// Number of postings in this block.
    pub count: usize,
}

impl InvertedIndexPostingIterator {
    /// Create a new advanced posting iterator.
    pub fn new(postings: Vec<Posting>) -> Self {
        InvertedIndexPostingIterator {
            postings,
            position: 0,
            block_cache: None,
            current_block: 0,
            started: false,
        }
    }

    /// Create posting iterator with block optimization.
    pub fn with_blocks(postings: Vec<Posting>, block_size: usize) -> Self {
        let blocks = Self::create_blocks(&postings, block_size);
        InvertedIndexPostingIterator {
            postings,
            position: 0,
            block_cache: Some(blocks),
            current_block: 0,
            started: false,
        }
    }

    /// Create posting blocks for efficient skip-to operations.
    fn create_blocks(postings: &[Posting], block_size: usize) -> Vec<PostingBlock> {
        let mut blocks = Vec::new();
        let mut start = 0;

        while start < postings.len() {
            let end = (start + block_size).min(postings.len());
            let block_postings = &postings[start..end];

            if !block_postings.is_empty() {
                blocks.push(PostingBlock {
                    min_doc_id: block_postings[0].doc_id,
                    max_doc_id: block_postings[block_postings.len() - 1].doc_id,
                    start_position: start,
                    count: end - start,
                });
            }

            start = end;
        }

        blocks
    }

    /// Find the block containing the target document ID.
    fn find_block(&self, target: u64) -> Option<usize> {
        if let Some(blocks) = &self.block_cache {
            for (i, block) in blocks.iter().enumerate() {
                if target >= block.min_doc_id && target <= block.max_doc_id {
                    return Some(i);
                }
                if target < block.min_doc_id {
                    return if i > 0 { Some(i - 1) } else { None };
                }
            }

            // Target is beyond all blocks
            if !blocks.is_empty() {
                Some(blocks.len() - 1)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl crate::lexical::reader::PostingIterator for InvertedIndexPostingIterator {
    fn doc_id(&self) -> u64 {
        if self.position < self.postings.len() {
            self.postings[self.position].doc_id
        } else {
            u64::MAX // Convention for exhausted iterator
        }
    }

    fn term_freq(&self) -> u64 {
        if self.position < self.postings.len() {
            self.postings[self.position].frequency as u64
        } else {
            0
        }
    }

    fn positions(&self) -> Result<Vec<u64>> {
        if self.position < self.postings.len() {
            Ok(self.postings[self.position]
                .positions
                .as_ref()
                .unwrap_or(&Vec::new())
                .iter()
                .map(|&p| p as u64)
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.postings.is_empty() {
            return Ok(false);
        }

        if !self.started {
            // First call - position at first document
            self.started = true;
            Ok(true)
        } else {
            // Move to next document
            self.position += 1;
            Ok(self.position < self.postings.len())
        }
    }

    fn skip_to(&mut self, target_doc_id: u64) -> Result<bool> {
        // Mark as started
        self.started = true;

        // Use block optimization if available
        if let Some(block_idx) = self.find_block(target_doc_id)
            && let Some(blocks) = &self.block_cache
        {
            let block = &blocks[block_idx];
            self.position = block.start_position;
            self.current_block = block_idx;
        }

        // Linear search within the current range
        while self.position < self.postings.len() {
            if self.postings[self.position].doc_id >= target_doc_id {
                return Ok(true);
            }
            self.position += 1;
        }
        Ok(false)
    }

    fn cost(&self) -> u64 {
        self.postings.len() as u64
    }
}

/// Reader for a single segment (schema-less mode).
#[derive(Debug)]
pub struct SegmentReader {
    /// Segment information.
    info: SegmentInfo,

    /// Storage backend.
    storage: Arc<dyn Storage>,

    /// Term dictionary for efficient term lookup.
    term_dictionary: RwLock<Option<Arc<HybridTermDictionary>>>,

    /// Cached stored documents.
    stored_documents: RwLock<Option<BTreeMap<u64, Document>>>,

    /// Cached field lengths: doc_id -> (field_name -> length).
    field_lengths: RwLock<Option<BTreeMap<u64, AHashMap<String, u32>>>>,

    /// Cached field statistics: field_name -> FieldStats.
    field_stats: RwLock<Option<AHashMap<String, crate::lexical::reader::FieldStats>>>,

    /// DocValues reader for this segment.
    doc_values: RwLock<Option<Arc<DocValuesReader>>>,

    /// Whether the segment is loaded.
    loaded: AtomicBool,
}

impl SegmentReader {
    /// Open a segment reader (schema-less mode).
    pub fn open(info: SegmentInfo, storage: Arc<dyn Storage>) -> Result<Self> {
        let reader = SegmentReader {
            info,
            storage,
            term_dictionary: RwLock::new(None),
            stored_documents: RwLock::new(None),
            field_lengths: RwLock::new(None),
            field_stats: RwLock::new(None),
            doc_values: RwLock::new(None),
            loaded: AtomicBool::new(false),
        };

        Ok(reader)
    }

    /// Deprecated: Use `open()` instead. Schema is no longer required.
    #[deprecated(
        since = "0.2.0",
        note = "Use `open()` instead. Schema is no longer required."
    )]
    pub fn open_with_schema(
        info: SegmentInfo,
        _schema: Arc<()>,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        Self::open(info, storage)
    }

    /// Load the segment data.
    pub fn load(&mut self) -> Result<()> {
        if self.loaded.load(Ordering::Acquire) {
            return Ok(());
        }

        // Load term dictionary
        self.load_term_dictionary()?;

        // Load stored documents
        self.load_stored_documents()?;

        // Load DocValues
        self.load_doc_values()?;

        self.loaded.store(true, Ordering::Release);
        Ok(())
    }

    /// Load the term dictionary for this segment.
    fn load_term_dictionary(&self) -> Result<()> {
        let dict_file = format!("{}.dict", self.info.segment_id);

        if let Ok(input) = self.storage.open_input(&dict_file) {
            let mut reader = StructReader::new(input)?;
            let dictionary = HybridTermDictionary::read_from_storage(&mut reader).map_err(|e| {
                YatagarasuError::index(format!(
                    "Failed to read term dictionary from {dict_file}: {e}"
                ))
            })?;
            *self.term_dictionary.write().unwrap() = Some(Arc::new(dictionary));
        }

        Ok(())
    }

    /// Load stored documents for this segment.
    fn load_stored_documents(&self) -> Result<()> {
        // Try JSON format first (for compatibility)
        let json_file = format!("{}.json", self.info.segment_id);
        if self.storage.file_exists(&json_file) {
            let mut input = self.storage.open_input(&json_file)?;
            let mut json_data = String::new();
            std::io::Read::read_to_string(&mut input, &mut json_data)?;

            let docs: Vec<Document> = serde_json::from_str(&json_data).map_err(|e| {
                YatagarasuError::index(format!("Failed to parse JSON documents: {e}"))
            })?;

            let mut documents = BTreeMap::new();
            for (idx, doc) in docs.into_iter().enumerate() {
                let doc_id = self.info.doc_offset + idx as u64;
                documents.insert(doc_id, doc);
            }

            *self.stored_documents.write().unwrap() = Some(documents);
            return Ok(());
        }

        // Fallback to binary format with type information
        let docs_file = format!("{}.docs", self.info.segment_id);
        if let Ok(input) = self.storage.open_input(&docs_file) {
            let mut reader = StructReader::new(input)?;
            let doc_count = reader.read_varint()? as usize;
            let mut documents = BTreeMap::new();

            for _ in 0..doc_count {
                let doc_id = reader.read_u64()?;
                let field_count = reader.read_varint()? as usize;
                let mut doc = Document::new();

                for _ in 0..field_count {
                    let field_name = reader.read_string()?;

                    // Read type tag
                    let type_tag = reader.read_u8()?;

                    // Read value based on type tag
                    let field_value = match type_tag {
                        0 => {
                            // Text
                            let text = reader.read_string()?;
                            FieldValue::Text(text)
                        }
                        1 => {
                            // Integer
                            let num = reader.read_u64()? as i64; // Read as u64, convert to i64 preserving bit pattern
                            FieldValue::Integer(num)
                        }
                        2 => {
                            // Float
                            let num = reader.read_f64()?;
                            FieldValue::Float(num)
                        }
                        3 => {
                            // Boolean
                            let b = reader.read_u8()? != 0;
                            FieldValue::Boolean(b)
                        }
                        4 => {
                            // Binary
                            let bytes = reader.read_bytes()?;
                            FieldValue::Binary(bytes)
                        }
                        5 => {
                            // DateTime
                            let dt_str = reader.read_string()?;
                            let dt = chrono::DateTime::parse_from_rfc3339(&dt_str)
                                .map_err(|e| {
                                    YatagarasuError::index(format!("Failed to parse DateTime: {e}"))
                                })?
                                .with_timezone(&chrono::Utc);
                            FieldValue::DateTime(dt)
                        }
                        6 => {
                            // Geo
                            let lat = reader.read_f64()?;
                            let lon = reader.read_f64()?;
                            FieldValue::Geo(crate::lexical::index::inverted::query::geo::GeoPoint {
                                lat,
                                lon,
                            })
                        }
                        7 => {
                            // Null
                            FieldValue::Null
                        }
                        8 => {
                            // Vector
                            let text = reader.read_string()?;
                            FieldValue::Vector(text)
                        }
                        _ => {
                            return Err(YatagarasuError::index(format!(
                                "Unknown field type tag: {type_tag}"
                            )));
                        }
                    };

                    doc.add_field(
                        field_name,
                        crate::document::field::Field::with_default_option(field_value),
                    );
                }

                documents.insert(doc_id, doc);
            }

            *self.stored_documents.write().unwrap() = Some(documents);
        }

        Ok(())
    }

    /// Load DocValues for this segment.
    fn load_doc_values(&self) -> Result<()> {
        // Load DocValues file (required for field sorting)
        let reader = DocValuesReader::load(self.storage.clone(), &self.info.segment_id)?;

        let mut doc_values = self.doc_values.write().unwrap();
        *doc_values = Some(Arc::new(reader));

        Ok(())
    }

    /// Get a DocValues field value for a document.
    fn get_doc_value(&self, field: &str, doc_id: u64) -> Result<Option<FieldValue>> {
        // Ensure DocValues are loaded (lazy loading)
        if !self.loaded.load(Ordering::Acquire) {
            // Try to load if not loaded yet (this is safe for read-only operations after load)
            self.ensure_loaded()?;
        }

        let doc_values = self.doc_values.read().unwrap();
        if let Some(reader) = doc_values.as_ref() {
            Ok(reader.get_value(field, doc_id).cloned())
        } else {
            Ok(None)
        }
    }

    /// Ensure the segment is loaded (for lazy loading support)
    fn ensure_loaded(&self) -> Result<()> {
        if !self.loaded.load(Ordering::Acquire) {
            // Load DocValues only (other data loaded on-demand)
            self.load_doc_values()?;
            // Note: We don't set loaded=true here to avoid race conditions
            // Full load should be done via load() method
        }
        Ok(())
    }

    /// Check if DocValues are available for a field.
    fn has_doc_values(&self, field: &str) -> bool {
        let doc_values = self.doc_values.read().unwrap();
        if let Some(reader) = doc_values.as_ref() {
            reader.has_field(field)
        } else {
            false
        }
    }

    /// Load field lengths from the segment.
    fn load_field_lengths(&self) -> Result<()> {
        let lens_file = format!("{}.lens", self.info.segment_id);

        // Check if file exists (for backward compatibility with old indexes)
        if !self.storage.file_exists(&lens_file) {
            // Old index without field lengths - initialize empty
            *self.field_lengths.write().unwrap() = Some(BTreeMap::new());
            return Ok(());
        }

        let lens_input = self.storage.open_input(&lens_file)?;
        let mut lens_reader = StructReader::new(lens_input)?;

        let doc_count = lens_reader.read_varint()? as usize;
        let mut all_field_lengths = BTreeMap::new();

        for _ in 0..doc_count {
            let doc_id = lens_reader.read_u64()?;
            let field_count = lens_reader.read_varint()? as usize;

            let mut field_lens = AHashMap::new();
            for _ in 0..field_count {
                let field_name = lens_reader.read_string()?;
                let length = lens_reader.read_u32()?;
                field_lens.insert(field_name, length);
            }

            all_field_lengths.insert(doc_id, field_lens);
        }

        *self.field_lengths.write().unwrap() = Some(all_field_lengths);
        Ok(())
    }

    /// Load field statistics from the segment.
    fn load_field_stats(&self) -> Result<()> {
        let fstats_file = format!("{}.fstats", self.info.segment_id);

        // Check if file exists (for backward compatibility with old indexes)
        if !self.storage.file_exists(&fstats_file) {
            // Old index without field stats - initialize empty
            *self.field_stats.write().unwrap() = Some(AHashMap::new());
            return Ok(());
        }

        let fstats_input = self.storage.open_input(&fstats_file)?;
        let mut fstats_reader = StructReader::new(fstats_input)?;

        let field_count = fstats_reader.read_varint()? as usize;
        let mut all_field_stats = AHashMap::new();

        for _ in 0..field_count {
            let field_name = fstats_reader.read_string()?;
            let doc_count = fstats_reader.read_u64()?;
            let avg_length = fstats_reader.read_f64()?;
            let min_length = fstats_reader.read_u64()?;
            let max_length = fstats_reader.read_u64()?;

            all_field_stats.insert(
                field_name.clone(),
                crate::lexical::reader::FieldStats {
                    field: field_name,
                    unique_terms: 0, // Not stored, not needed for BM25
                    total_terms: 0,  // Not stored, not needed for BM25
                    doc_count,
                    avg_length,
                    min_length,
                    max_length,
                },
            );
        }

        *self.field_stats.write().unwrap() = Some(all_field_stats);
        Ok(())
    }

    /// Get field statistics for a specific field.
    pub fn field_stats(&self, field: &str) -> Result<Option<FieldStats>> {
        // Ensure field stats are loaded
        if self.field_stats.read().unwrap().is_none() {
            self.load_field_stats()?;
        }

        let field_stats = self.field_stats.read().unwrap();
        if let Some(ref stats_map) = *field_stats {
            return Ok(stats_map.get(field).cloned());
        }
        Ok(None)
    }

    /// Get field length for a specific document and field.
    pub fn field_length(&self, local_doc_id: u64, field: &str) -> Result<Option<u32>> {
        // Ensure field lengths are loaded
        if self.field_lengths.read().unwrap().is_none() {
            self.load_field_lengths()?;
        }

        let field_lengths = self.field_lengths.read().unwrap();
        if let Some(ref lengths_map) = *field_lengths
            && let Some(doc_lengths) = lengths_map.get(&local_doc_id)
        {
            return Ok(doc_lengths.get(field).copied());
        }
        Ok(None)
    }

    /// Get a document by ID from this segment.
    pub fn document(&self, local_doc_id: u64) -> Result<Option<Document>> {
        // Ensure documents are loaded
        if !self.loaded.load(Ordering::Acquire) {
            // Load documents on-demand
            self.load_stored_documents()?;
        }

        // Check if local_doc_id is within this segment's range
        if local_doc_id >= self.info.doc_count {
            return Ok(None);
        }

        // Convert local doc_id to global doc_id for storage lookup
        let global_doc_id = self.info.doc_offset + local_doc_id;

        let docs = self.stored_documents.read().unwrap();
        if let Some(ref documents) = *docs {
            Ok(documents.get(&global_doc_id).cloned())
        } else {
            Ok(None)
        }
    }

    /// Get term information for a field and term.
    pub fn term_info(&self, field: &str, term: &str) -> Result<Option<TermInfo>> {
        // Lazy load term dictionary if not loaded
        if self.term_dictionary.read().unwrap().is_none() && !self.loaded.load(Ordering::Acquire) {
            self.load_term_dictionary()?;
        }

        if let Some(ref dict) = *self.term_dictionary.read().unwrap() {
            let full_term = format!("{field}:{term}");
            Ok(dict.get(&full_term).cloned())
        } else {
            Ok(None)
        }
    }

    /// Get posting list for a field and term.
    pub fn postings(&self, field: &str, term: &str) -> Result<Option<Box<dyn PostingIterator>>> {
        // Load postings from storage
        let postings_file = format!("{}.post", self.info.segment_id);

        if !self.storage.file_exists(&postings_file) {
            // No inverted index, fall back to document scanning
            return self.scan_documents_for_term(field, term);
        }

        if let Some(term_info) = self.term_info(field, term)? {
            let input = self.storage.open_input(&postings_file)?;
            let mut reader = StructReader::new(input)?;

            // Skip to the posting position
            let mut current_pos = 0u64;
            while current_pos < term_info.posting_offset {
                reader.read_u8()?;
                current_pos += 1;
            }

            // Decode the posting list
            let posting_list = PostingList::decode(&mut reader)?;

            Ok(Some(Box::new(InvertedIndexPostingIterator::with_blocks(
                posting_list.postings,
                64,
            ))))
        } else {
            Ok(None)
        }
    }

    /// Scan documents for a term (fallback when no inverted index).
    fn scan_documents_for_term(
        &self,
        field: &str,
        term: &str,
    ) -> Result<Option<Box<dyn PostingIterator>>> {
        // Ensure documents are loaded
        if !self.loaded.load(Ordering::Acquire) {
            // Load documents on-demand
            self.load_stored_documents()?;
        }

        let docs = self.stored_documents.read().unwrap();
        if let Some(documents) = docs.as_ref() {
            let mut postings = Vec::new();
            let default_analyzer = StandardAnalyzer::new()?;

            for (doc_id, doc) in documents.iter() {
                if let Some(field_value) = doc.get_field(field)
                    && let Some(text) = field_value.value.as_text()
                {
                    // Use default analyzer (analyzers are configured at writer level)
                    let token_stream = default_analyzer.analyze(text)?;
                    let tokens: Vec<Token> = token_stream.collect();

                    let mut positions = Vec::new();
                    for token in tokens.iter() {
                        if token.text == term {
                            positions.push(token.position as u32);
                        }
                    }

                    if !positions.is_empty() {
                        postings.push(Posting {
                            doc_id: *doc_id,
                            frequency: positions.len() as u32,
                            positions: Some(positions),
                            weight: 1.0,
                        });
                    }
                }
            }

            if postings.is_empty() {
                Ok(None)
            } else {
                Ok(Some(Box::new(InvertedIndexPostingIterator::with_blocks(
                    postings, 64,
                ))))
            }
        } else {
            Ok(None)
        }
    }

    /// Get the number of documents in this segment.
    pub fn doc_count(&self) -> u64 {
        self.info.doc_count
    }
}

/// Cache manager for efficient data access.
#[derive(Debug)]
pub struct CacheManager {
    /// Term information cache.
    term_cache: RwLock<AHashMap<String, TermInfo>>,

    /// Posting list cache.
    #[allow(dead_code)]
    posting_cache:
        RwLock<AHashMap<String, Arc<Vec<crate::lexical::index::inverted::core::posting::Posting>>>>,

    /// Current memory usage.
    memory_usage: AtomicUsize,

    /// Maximum memory limit.
    memory_limit: usize,

    /// Cache statistics.
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
}

impl CacheManager {
    /// Create a new cache manager.
    pub fn new(memory_limit: usize) -> Self {
        CacheManager {
            term_cache: RwLock::new(AHashMap::new()),
            posting_cache: RwLock::new(AHashMap::new()),
            memory_usage: AtomicUsize::new(0),
            memory_limit,
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
        }
    }

    /// Get term information from cache.
    pub fn get_term_info(&self, key: &str) -> Option<TermInfo> {
        let cache = self.term_cache.read().unwrap();
        if let Some(info) = cache.get(key) {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            Some(info.clone())
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Cache term information.
    pub fn cache_term_info(&self, key: String, info: TermInfo) {
        if self.memory_usage.load(Ordering::Relaxed) < self.memory_limit {
            let mut cache = self.term_cache.write().unwrap();
            cache.insert(key, info);

            // Estimate memory usage (rough approximation)
            self.memory_usage.fetch_add(64, Ordering::Relaxed);
        }
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.cache_hits.load(Ordering::Relaxed),
            misses: self.cache_misses.load(Ordering::Relaxed),
            memory_usage: self.memory_usage.load(Ordering::Relaxed),
            memory_limit: self.memory_limit,
        }
    }
}

/// Cache performance statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: usize,

    /// Number of cache misses.
    pub misses: usize,

    /// Current memory usage.
    pub memory_usage: usize,

    /// Memory limit.
    pub memory_limit: usize,
}

impl CacheStats {
    /// Calculate hit ratio.
    pub fn hit_ratio(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }
}

/// Advanced index reader with multi-segment support (schema-less mode).
#[derive(Debug, Clone)]
pub struct InvertedIndexReader {
    /// Segment readers.
    segment_readers: Vec<Arc<RwLock<SegmentReader>>>,

    /// Cache manager.
    cache_manager: Arc<CacheManager>,

    /// Reader configuration.
    config: InvertedIndexReaderConfig,

    /// Whether the reader is closed.
    closed: Arc<AtomicBool>,

    /// Total document count across all segments.
    total_doc_count: u64,
}

impl InvertedIndexReader {
    /// Create a new advanced index reader (schema-less mode).
    pub fn new(
        segments: Vec<SegmentInfo>,
        storage: Arc<dyn Storage>,
        config: InvertedIndexReaderConfig,
    ) -> Result<Self> {
        let cache_manager = Arc::new(CacheManager::new(config.max_cache_memory));
        let mut segment_readers = Vec::new();
        let mut total_doc_count = 0;

        for segment_info in &segments {
            total_doc_count += segment_info.doc_count;
            let mut reader = SegmentReader::open(segment_info.clone(), storage.clone())?;

            if config.preload_segments {
                reader.load()?;
            }

            segment_readers.push(Arc::new(RwLock::new(reader)));
        }

        Ok(InvertedIndexReader {
            segment_readers,
            cache_manager,
            config,
            closed: Arc::new(AtomicBool::new(false)),
            total_doc_count,
        })
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache_manager.stats()
    }

    /// Get the analyzer from configuration.
    pub fn analyzer(&self) -> &Arc<dyn Analyzer> {
        &self.config.analyzer
    }

    /// Check if the reader is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            Err(YatagarasuError::index("Reader is closed"))
        } else {
            Ok(())
        }
    }

    /// Get the field length for a specific document and field.
    pub fn field_length(&self, doc_id: u64, field: &str) -> Result<Option<u32>> {
        self.check_closed()?;

        // Find the segment containing this document
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            let segment_info = &reader.info;

            // Check if this doc_id belongs to this segment
            let segment_start = segment_info.doc_offset;
            let segment_end = segment_start + segment_info.doc_count;

            if doc_id >= segment_start && doc_id < segment_end {
                // Convert to segment-local doc_id
                let local_doc_id = doc_id - segment_start;
                return reader.field_length(local_doc_id, field);
            }
        }

        Ok(None)
    }
}

impl crate::lexical::reader::LexicalIndexReader for InvertedIndexReader {
    fn doc_count(&self) -> u64 {
        self.total_doc_count
    }

    fn max_doc(&self) -> u64 {
        self.total_doc_count
    }

    fn is_deleted(&self, _doc_id: u64) -> bool {
        // TODO: Implement deletion support
        false
    }

    fn document(&self, doc_id: u64) -> Result<Option<Document>> {
        self.check_closed()?;

        // Find the segment containing this document
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            let segment_info = &reader.info;

            // Check if this doc_id belongs to this segment
            let segment_start = segment_info.doc_offset;
            let segment_end = segment_start + segment_info.doc_count;

            if doc_id >= segment_start && doc_id < segment_end {
                // Convert to segment-local doc_id
                let local_doc_id = doc_id - segment_start;
                return reader.document(local_doc_id);
            }
        }

        Ok(None)
    }

    fn term_info(
        &self,
        field: &str,
        term: &str,
    ) -> Result<Option<crate::lexical::reader::ReaderTermInfo>> {
        self.check_closed()?;

        let cache_key = format!("{field}:{term}");

        // Check cache first
        if let Some(cached_info) = self.cache_manager.get_term_info(&cache_key) {
            return Ok(Some(crate::lexical::reader::ReaderTermInfo {
                field: field.to_string(),
                term: term.to_string(),
                doc_freq: cached_info.doc_frequency,
                total_freq: cached_info.total_frequency,
                posting_offset: cached_info.posting_offset,
                posting_size: cached_info.posting_length,
            }));
        }

        // Search across all segments
        let mut total_doc_freq = 0;
        let mut total_term_freq = 0;
        let mut found = false;

        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            if let Some(term_info) = reader.term_info(field, term)? {
                total_doc_freq += term_info.doc_frequency;
                total_term_freq += term_info.total_frequency;
                found = true;
            }
        }

        if found {
            let reader_info = crate::lexical::reader::ReaderTermInfo {
                field: field.to_string(),
                term: term.to_string(),
                doc_freq: total_doc_freq,
                total_freq: total_term_freq,
                posting_offset: 0, // Aggregated value, not meaningful for multi-segment
                posting_size: 0,   // Aggregated value, not meaningful for multi-segment
            };

            // Cache the result
            let term_info = TermInfo {
                posting_offset: 0,
                posting_length: 0,
                doc_frequency: total_doc_freq,
                total_frequency: total_term_freq,
            };
            self.cache_manager.cache_term_info(cache_key, term_info);

            Ok(Some(reader_info))
        } else {
            Ok(None)
        }
    }

    fn postings(
        &self,
        field: &str,
        term: &str,
    ) -> Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>> {
        self.check_closed()?;

        let mut iterators = Vec::new();

        // Collect posting iterators from all segments
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            if let Some(iter) = reader.postings(field, term)? {
                iterators.push(iter);
            }
        }

        if iterators.is_empty() {
            Ok(None)
        } else if iterators.len() == 1 {
            // Single segment case
            Ok(Some(iterators.into_iter().next().unwrap()))
        } else {
            // Multi-segment case - need to merge iterators
            // For now, return the first iterator
            // TODO: Implement proper merged iterator
            Ok(Some(iterators.into_iter().next().unwrap()))
        }
    }

    fn field_stats(&self, field: &str) -> Result<Option<crate::lexical::reader::FieldStats>> {
        self.check_closed()?;

        let mut total_doc_count = 0u64;
        let mut total_length_sum = 0u64; // Sum of (avg_length * doc_count) for weighted average
        let mut min_length = u64::MAX;
        let mut max_length = 0u64;
        let mut found = false;

        // Aggregate statistics from all segments
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();

            // Get field stats from this segment
            if let Some(segment_stats) = reader.field_stats(field)? {
                total_doc_count += segment_stats.doc_count;
                total_length_sum +=
                    (segment_stats.avg_length * segment_stats.doc_count as f64) as u64;
                min_length = min_length.min(segment_stats.min_length);
                max_length = max_length.max(segment_stats.max_length);
                found = true;
            }
        }

        if found {
            Ok(Some(crate::lexical::reader::FieldStats {
                field: field.to_string(),
                unique_terms: 0, // Not aggregated
                total_terms: 0,  // Not aggregated
                doc_count: total_doc_count,
                avg_length: if total_doc_count > 0 {
                    total_length_sum as f64 / total_doc_count as f64
                } else {
                    0.0
                },
                min_length: if min_length == u64::MAX {
                    0
                } else {
                    min_length
                },
                max_length,
            }))
        } else {
            Ok(None)
        }
    }

    fn close(&mut self) -> Result<()> {
        self.closed.store(true, Ordering::Release);
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_doc_value(&self, field: &str, doc_id: u64) -> Result<Option<FieldValue>> {
        // Find which segment contains this doc_id and convert to segment-local doc_id
        let mut offset = 0u64;
        for segment_lock in &self.segment_readers {
            let segment = segment_lock.read().unwrap();
            let segment_doc_count = segment.info.doc_count;

            if doc_id < offset + segment_doc_count {
                // This segment contains the document
                let local_doc_id = doc_id - offset;
                return segment.get_doc_value(field, local_doc_id);
            }

            offset += segment_doc_count;
        }
        Ok(None)
    }

    fn has_doc_values(&self, field: &str) -> bool {
        // Check if any segment has DocValues for this field
        self.segment_readers.iter().any(|seg_lock| {
            let seg = seg_lock.read().unwrap();
            seg.has_doc_values(field)
        })
    }
}

// Implementation of TermDictionaryAccess for InvertedIndexReader
impl TermDictionaryAccess for InvertedIndexReader {
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms>>> {
        // Get the first segment's term dictionary
        // In a multi-segment index, we would need to merge terms from all segments.
        // For now, we'll use a simplified approach with the first segment.
        if let Some(seg_lock) = self.segment_readers.first() {
            let seg = seg_lock.read().unwrap();

            // Load the term dictionary if not already loaded
            if seg.term_dictionary.read().unwrap().is_none() {
                seg.load_term_dictionary()?;
            }

            if let Some(dict) = seg.term_dictionary.read().unwrap().clone() {
                let terms = InvertedIndexTerms::new(field, dict);
                return Ok(Some(Box::new(terms)));
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::reader::PostingIterator;

    #[test]
    fn test_advanced_posting_iterator() {
        let postings = vec![
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 1,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 3,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 5,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 7,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 9,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
        ];

        let mut iter = InvertedIndexPostingIterator::with_blocks(postings, 2);

        // Test skip_to functionality
        assert!(iter.skip_to(5).unwrap());
        assert_eq!(iter.doc_id(), 5);

        // Test next
        assert!(iter.next().unwrap());
        assert_eq!(iter.doc_id(), 7);

        // Test skip past end
        assert!(!iter.skip_to(15).unwrap());
        assert_eq!(iter.doc_id(), u64::MAX);
    }

    #[test]
    fn test_cache_manager() {
        let cache = CacheManager::new(1024);
        let key = "field:term".to_string();
        let term_info = TermInfo::new(100, 50, 5, 10);

        // Test cache miss
        assert!(cache.get_term_info(&key).is_none());

        // Test cache insertion and hit
        cache.cache_term_info(key.clone(), term_info.clone());
        let cached = cache.get_term_info(&key).unwrap();
        assert_eq!(cached.doc_frequency, term_info.doc_frequency);

        // Test cache statistics
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!(stats.hit_ratio() > 0.0);
    }

    #[test]
    fn test_segment_info() {
        let info = SegmentInfo {
            segment_id: "seg_000001".to_string(),
            doc_count: 1000,
            doc_offset: 0,
            generation: 1,
            has_deletions: false,
        };

        assert_eq!(info.segment_id, "seg_000001");
        assert_eq!(info.doc_count, 1000);
        assert!(!info.has_deletions);
    }
}
