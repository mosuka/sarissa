//! Advanced index reader with segment-based architecture.
//!
//! This module provides a production-ready index reader that efficiently
//! handles multiple segments, caching, and optimized posting list access.

use crate::error::{SarissaError, Result};
use crate::index::dictionary::HybridTermDictionary;
use crate::index::{SegmentInfo, TermInfo};
use crate::schema::{Document, FieldValue, Schema};
use crate::storage::{Storage, StructReader};
use ahash::AHashMap;
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, RwLock,
};

/// Advanced index reader configuration.
#[derive(Debug, Clone)]
pub struct AdvancedReaderConfig {
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
}

impl Default for AdvancedReaderConfig {
    fn default() -> Self {
        AdvancedReaderConfig {
            max_cache_memory: 128 * 1024 * 1024, // 128MB
            enable_term_cache: true,
            enable_posting_cache: true,
            preload_segments: false,
            max_cached_terms_per_field: 10000,
        }
    }
}

// SegmentInfo is now imported from crate::index

/// Advanced posting iterator with block-based optimization.
#[derive(Debug)]
pub struct AdvancedPostingIterator {
    /// The posting data.
    postings: Vec<crate::index::Posting>,

    /// Current position in the posting list.
    position: usize,

    /// Block cache for efficient access.
    block_cache: Option<Vec<PostingBlock>>,

    /// Current block being processed.
    current_block: usize,
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

impl AdvancedPostingIterator {
    /// Create a new advanced posting iterator.
    pub fn new(postings: Vec<crate::index::Posting>) -> Self {
        AdvancedPostingIterator {
            postings,
            position: 0,
            block_cache: None,
            current_block: 0,
        }
    }

    /// Create posting iterator with block optimization.
    pub fn with_blocks(postings: Vec<crate::index::Posting>, block_size: usize) -> Self {
        let blocks = Self::create_blocks(&postings, block_size);
        AdvancedPostingIterator {
            postings,
            position: 0,
            block_cache: Some(blocks),
            current_block: 0,
        }
    }

    /// Create posting blocks for efficient skip-to operations.
    fn create_blocks(postings: &[crate::index::Posting], block_size: usize) -> Vec<PostingBlock> {
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

impl crate::index::reader::PostingIterator for AdvancedPostingIterator {
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
        if self.position < self.postings.len() {
            self.position += 1;
            Ok(self.position < self.postings.len())
        } else {
            Ok(false)
        }
    }

    fn skip_to(&mut self, target_doc_id: u64) -> Result<bool> {
        // Use block optimization if available
        if let Some(block_idx) = self.find_block(target_doc_id) {
            if let Some(blocks) = &self.block_cache {
                let block = &blocks[block_idx];
                self.position = block.start_position;
                self.current_block = block_idx;
            }
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

/// Reader for a single segment.
#[derive(Debug)]
pub struct SegmentReader {
    /// Segment information.
    info: SegmentInfo,

    /// Schema for this segment.
    #[allow(dead_code)]
    schema: Arc<Schema>,

    /// Storage backend.
    storage: Arc<dyn Storage>,

    /// Term dictionary for efficient term lookup.
    term_dictionary: Option<Arc<HybridTermDictionary>>,

    /// Cached stored documents.
    stored_documents: RwLock<Option<BTreeMap<u64, Document>>>,

    /// Whether the segment is loaded.
    loaded: AtomicBool,
}

impl SegmentReader {
    /// Open a segment reader.
    pub fn open(info: SegmentInfo, schema: Arc<Schema>, storage: Arc<dyn Storage>) -> Result<Self> {
        let reader = SegmentReader {
            info,
            schema,
            storage,
            term_dictionary: None,
            stored_documents: RwLock::new(None),
            loaded: AtomicBool::new(false),
        };

        Ok(reader)
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

        self.loaded.store(true, Ordering::Release);
        Ok(())
    }

    /// Load the term dictionary for this segment.
    fn load_term_dictionary(&mut self) -> Result<()> {
        let dict_file = format!("{}.dict", self.info.segment_id);

        if let Ok(input) = self.storage.open_input(&dict_file) {
            let mut reader = StructReader::new(input)?;
            let dictionary = HybridTermDictionary::read_from_storage(&mut reader)?;
            self.term_dictionary = Some(Arc::new(dictionary));
        }

        Ok(())
    }

    /// Load stored documents for this segment.
    fn load_stored_documents(&self) -> Result<()> {
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
                    let field_value = reader.read_string()?;
                    doc.add_field(field_name, FieldValue::Text(field_value));
                }

                documents.insert(doc_id, doc);
            }

            *self.stored_documents.write().unwrap() = Some(documents);
        }

        Ok(())
    }

    /// Get a document by ID from this segment.
    pub fn document(&self, doc_id: u64) -> Result<Option<Document>> {
        // Adjust document ID for this segment
        if doc_id < self.info.doc_offset || doc_id >= self.info.doc_offset + self.info.doc_count {
            return Ok(None);
        }

        let local_doc_id = doc_id - self.info.doc_offset;

        let docs = self.stored_documents.read().unwrap();
        if let Some(ref documents) = *docs {
            Ok(documents.get(&local_doc_id).cloned())
        } else {
            Ok(None)
        }
    }

    /// Get term information for a field and term.
    pub fn term_info(&self, field: &str, term: &str) -> Result<Option<TermInfo>> {
        if let Some(ref dict) = self.term_dictionary {
            let full_term = format!("{field}:{term}");
            Ok(dict.get(&full_term).cloned())
        } else {
            Ok(None)
        }
    }

    /// Get posting list for a field and term.
    pub fn postings(
        &self,
        field: &str,
        term: &str,
    ) -> Result<Option<Box<dyn crate::index::reader::PostingIterator>>> {
        if let Some(_term_info) = self.term_info(field, term)? {
            // For now, create a dummy posting list
            // TODO: Implement actual posting list loading from storage
            let postings = vec![
                crate::index::Posting {
                    doc_id: self.info.doc_offset + 1,
                    frequency: 1,
                    positions: Some(vec![0]),
                    weight: 1.0,
                },
                crate::index::Posting {
                    doc_id: self.info.doc_offset + 2,
                    frequency: 1,
                    positions: Some(vec![0]),
                    weight: 1.0,
                },
            ];

            Ok(Some(Box::new(AdvancedPostingIterator::with_blocks(
                postings, 64,
            ))))
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
    posting_cache: RwLock<AHashMap<String, Arc<Vec<crate::index::Posting>>>>,

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

/// Advanced index reader with multi-segment support.
#[derive(Debug)]
pub struct AdvancedIndexReader {
    /// Schema for this reader.
    schema: Arc<Schema>,

    /// Segment readers.
    segment_readers: Vec<Arc<RwLock<SegmentReader>>>,

    /// Cache manager.
    cache_manager: Arc<CacheManager>,

    /// Reader configuration.
    #[allow(dead_code)]
    config: AdvancedReaderConfig,

    /// Whether the reader is closed.
    closed: AtomicBool,

    /// Total document count across all segments.
    total_doc_count: u64,
}

impl AdvancedIndexReader {
    /// Create a new advanced index reader.
    pub fn new(
        schema: Schema,
        segments: Vec<SegmentInfo>,
        storage: Arc<dyn Storage>,
        config: AdvancedReaderConfig,
    ) -> Result<Self> {
        let schema = Arc::new(schema);
        let cache_manager = Arc::new(CacheManager::new(config.max_cache_memory));
        let mut segment_readers = Vec::new();
        let mut total_doc_count = 0;

        for segment_info in segments {
            total_doc_count += segment_info.doc_count;
            let mut reader = SegmentReader::open(segment_info, schema.clone(), storage.clone())?;

            if config.preload_segments {
                reader.load()?;
            }

            segment_readers.push(Arc::new(RwLock::new(reader)));
        }

        Ok(AdvancedIndexReader {
            schema,
            segment_readers,
            cache_manager,
            config,
            closed: AtomicBool::new(false),
            total_doc_count,
        })
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache_manager.stats()
    }

    /// Check if the reader is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            Err(SarissaError::index("Reader is closed"))
        } else {
            Ok(())
        }
    }
}

impl crate::index::reader::IndexReader for AdvancedIndexReader {
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
            if let Some(doc) = reader.document(doc_id)? {
                return Ok(Some(doc));
            }
        }

        Ok(None)
    }

    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn term_info(
        &self,
        field: &str,
        term: &str,
    ) -> Result<Option<crate::index::reader::ReaderTermInfo>> {
        self.check_closed()?;

        let cache_key = format!("{field}:{term}");

        // Check cache first
        if let Some(cached_info) = self.cache_manager.get_term_info(&cache_key) {
            return Ok(Some(crate::index::reader::ReaderTermInfo {
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
            let reader_info = crate::index::reader::ReaderTermInfo {
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
    ) -> Result<Option<Box<dyn crate::index::reader::PostingIterator>>> {
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

    fn field_stats(&self, field: &str) -> Result<Option<crate::index::reader::FieldStats>> {
        self.check_closed()?;

        let total_unique_terms = 0;
        let total_terms = 0;
        let mut total_doc_count = 0;
        let min_length = u64::MAX;
        let max_length = 0;
        let mut found = false;

        // Aggregate statistics from all segments
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();

            // For now, we don't have field-specific statistics
            // This would require additional metadata storage
            total_doc_count += reader.doc_count();
            found = true;
        }

        if found {
            Ok(Some(crate::index::reader::FieldStats {
                field: field.to_string(),
                unique_terms: total_unique_terms,
                total_terms,
                doc_count: total_doc_count,
                avg_length: if total_doc_count > 0 {
                    total_terms as f64 / total_doc_count as f64
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::reader::PostingIterator;

    #[test]
    fn test_advanced_posting_iterator() {
        let postings = vec![
            crate::index::Posting {
                doc_id: 1,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::index::Posting {
                doc_id: 3,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::index::Posting {
                doc_id: 5,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::index::Posting {
                doc_id: 7,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::index::Posting {
                doc_id: 9,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
        ];

        let mut iter = AdvancedPostingIterator::with_blocks(postings, 2);

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
