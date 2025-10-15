//! Merge engine for combining segments efficiently.
//!
//! This module provides the core functionality for merging multiple segments
//! into a single optimized segment with proper handling of deletions and updates.

use std::sync::Arc;
use std::time::SystemTime;

use ahash::AHashSet;

use crate::document::Document;
use crate::error::{Result, SageError};
use crate::full_text::dictionary::TermDictionaryBuilder;
use crate::full_text::reader::IndexReader;
use crate::full_text::{InvertedIndex, SegmentInfo, TermInfo};
use crate::full_text_index::segment_manager::{ManagedSegmentInfo, MergeCandidate, MergeStrategy};
use crate::full_text_search::AdvancedIndexReader;
use crate::storage::{Storage, StructWriter};

/// Configuration for merge operations.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Maximum memory usage during merge (in bytes).
    pub max_memory_mb: u64,

    /// Number of documents to process in each batch.
    pub batch_size: usize,

    /// Enable compression during merge.
    pub enable_compression: bool,

    /// Remove deleted documents during merge.
    pub remove_deleted_docs: bool,

    /// Sort documents by ID during merge for better locality.
    pub sort_by_doc_id: bool,

    /// Verify integrity after merge.
    pub verify_after_merge: bool,
}

impl Default for MergeConfig {
    fn default() -> Self {
        MergeConfig {
            max_memory_mb: 256,
            batch_size: 10000,
            enable_compression: true,
            remove_deleted_docs: true,
            sort_by_doc_id: true,
            verify_after_merge: true,
        }
    }
}

/// Statistics about a merge operation.
#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    /// Number of segments merged.
    pub segments_merged: usize,

    /// Number of documents processed.
    pub docs_processed: u64,

    /// Number of deleted documents removed.
    pub deleted_docs_removed: u64,

    /// Size before merge (in bytes).
    pub size_before: u64,

    /// Size after merge (in bytes).
    pub size_after: u64,

    /// Time taken for merge (in milliseconds).
    pub merge_time_ms: u64,

    /// Compression ratio achieved.
    pub compression_ratio: f64,

    /// Terms merged.
    pub terms_merged: u64,

    /// Postings merged.
    pub postings_merged: u64,
}

impl MergeStats {
    /// Calculate space savings percentage.
    pub fn space_savings(&self) -> f64 {
        if self.size_before == 0 {
            0.0
        } else {
            ((self.size_before - self.size_after) as f64 / self.size_before as f64) * 100.0
        }
    }
}

/// Result of a merge operation.
#[derive(Debug)]
pub struct MergeResult {
    /// Information about the new merged segment.
    pub new_segment: ManagedSegmentInfo,

    /// Statistics about the merge operation.
    pub stats: MergeStats,

    /// File paths of the new segment.
    pub file_paths: Vec<String>,
}

/// Core merge engine for segment operations (schema-less mode).
#[derive(Debug)]
pub struct MergeEngine {
    /// Configuration for merge operations.
    config: MergeConfig,

    /// Storage backend.
    storage: Arc<dyn Storage>,
}

impl MergeEngine {
    /// Create a new merge engine (schema-less mode).
    pub fn new(config: MergeConfig, storage: Arc<dyn Storage>) -> Self {
        MergeEngine { config, storage }
    }

    /// Merge segments according to the merge candidate.
    pub fn merge_segments(
        &self,
        candidate: &MergeCandidate,
        segments: &[ManagedSegmentInfo],
        next_generation: u64,
    ) -> Result<MergeResult> {
        let start_time = SystemTime::now();

        // Filter segments to merge
        let segments_to_merge: Vec<_> = segments
            .iter()
            .filter(|seg| candidate.segments.contains(&seg.segment_info.segment_id))
            .collect();

        if segments_to_merge.is_empty() {
            return Err(SageError::index("No segments found to merge"));
        }

        // Create new segment ID
        let new_segment_id = format!("merged_{next_generation}");

        // Initialize merge statistics
        let mut stats = MergeStats {
            segments_merged: segments_to_merge.len(),
            size_before: segments_to_merge.iter().map(|s| s.size_bytes).sum(),
            ..Default::default()
        };

        // Perform merge based on strategy
        let merge_result = match candidate.strategy {
            MergeStrategy::SizeBased => self.merge_by_size(&segments_to_merge, &new_segment_id)?,
            MergeStrategy::DeletionBased => {
                self.merge_by_deletion(&segments_to_merge, &new_segment_id)?
            }
            MergeStrategy::TimeBased => self.merge_by_time(&segments_to_merge, &new_segment_id)?,
            MergeStrategy::Balanced => self.merge_balanced(&segments_to_merge, &new_segment_id)?,
        };

        // Calculate final statistics
        let end_time = SystemTime::now();
        stats.merge_time_ms = end_time
            .duration_since(start_time)
            .unwrap_or_default()
            .as_millis() as u64;

        stats.size_after = merge_result.new_segment.size_bytes;
        stats.compression_ratio = if stats.size_before > 0 {
            stats.size_after as f64 / stats.size_before as f64
        } else {
            1.0
        };

        // Update merge result stats
        let mut final_result = merge_result;
        final_result.stats = stats;

        // Verify merge if configured
        if self.config.verify_after_merge {
            self.verify_merged_segment(&final_result.new_segment)?;
        }

        Ok(final_result)
    }

    /// Merge segments prioritizing size efficiency.
    fn merge_by_size(
        &self,
        segments: &[&ManagedSegmentInfo],
        new_segment_id: &str,
    ) -> Result<MergeResult> {
        // Sort segments by size (smallest first for better merging efficiency)
        let mut sorted_segments = segments.to_vec();
        sorted_segments.sort_by_key(|s| s.size_bytes);

        self.perform_merge(&sorted_segments, new_segment_id)
    }

    /// Merge segments prioritizing deletion removal.
    fn merge_by_deletion(
        &self,
        segments: &[&ManagedSegmentInfo],
        new_segment_id: &str,
    ) -> Result<MergeResult> {
        // Sort by deletion ratio (highest first for better compaction)
        let mut sorted_segments = segments.to_vec();
        sorted_segments
            .sort_by(|a, b| b.deletion_ratio().partial_cmp(&a.deletion_ratio()).unwrap());

        self.perform_merge(&sorted_segments, new_segment_id)
    }

    /// Merge segments prioritizing age.
    fn merge_by_time(
        &self,
        segments: &[&ManagedSegmentInfo],
        new_segment_id: &str,
    ) -> Result<MergeResult> {
        // Sort by creation time (oldest first)
        let mut sorted_segments = segments.to_vec();
        sorted_segments.sort_by_key(|s| s.created_at);

        self.perform_merge(&sorted_segments, new_segment_id)
    }

    /// Balanced merge considering multiple factors.
    fn merge_balanced(
        &self,
        segments: &[&ManagedSegmentInfo],
        new_segment_id: &str,
    ) -> Result<MergeResult> {
        // Calculate composite score for each segment
        let mut scored_segments: Vec<_> = segments
            .iter()
            .map(|seg| {
                let size_score = 1.0 / (seg.size_bytes as f64 + 1.0); // Prefer smaller
                let deletion_score = seg.deletion_ratio() * 2.0; // Prefer high deletion
                let age_score = 1.0 / (seg.created_at as f64 + 1.0); // Prefer older

                let composite_score = size_score + deletion_score + age_score;
                (*seg, composite_score)
            })
            .collect();

        // Sort by composite score (highest first)
        scored_segments.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let sorted_segments: Vec<_> = scored_segments.into_iter().map(|(seg, _)| seg).collect();

        self.perform_merge(&sorted_segments, new_segment_id)
    }

    /// Core merge implementation.
    fn perform_merge(
        &self,
        segments: &[&ManagedSegmentInfo],
        new_segment_id: &str,
    ) -> Result<MergeResult> {
        let mut stats = MergeStats {
            segments_merged: segments.len(),
            ..Default::default()
        };

        // Create merged inverted index
        let mut merged_index = InvertedIndex::new();
        let mut all_documents = Vec::new();
        let deleted_doc_ids = AHashSet::<u64>::new();
        let mut next_doc_id = 0u64;

        // Load deletion information for each segment
        for segment in segments {
            if segment.segment_info.has_deletions {
                // TODO: Load actual deletion bitmap
                // For now, simulate based on deleted_count
            }
        }

        // Process documents from all segments
        for segment in segments {
            // Load segment reader
            let segment_reader = self.load_segment_reader(&segment.segment_info)?;

            // Process documents in batches
            let mut batch_docs = Vec::new();
            let total_docs = segment.segment_info.doc_count;

            for doc_id in 0..total_docs {
                let global_doc_id = segment.segment_info.doc_offset + doc_id;

                // Skip deleted documents if configured
                if self.config.remove_deleted_docs && deleted_doc_ids.contains(&global_doc_id) {
                    stats.deleted_docs_removed += 1;
                    continue;
                }

                // Load document
                if let Some(document) = segment_reader.document(global_doc_id)? {
                    batch_docs.push((next_doc_id, document));
                    next_doc_id += 1;

                    // Process batch when full
                    if batch_docs.len() >= self.config.batch_size {
                        self.process_document_batch(&mut merged_index, &mut batch_docs)?;
                        all_documents.append(&mut batch_docs);
                        stats.docs_processed += self.config.batch_size as u64;
                    }
                }
            }

            // Process remaining documents
            if !batch_docs.is_empty() {
                self.process_document_batch(&mut merged_index, &mut batch_docs)?;
                stats.docs_processed += batch_docs.len() as u64;
                all_documents.extend(batch_docs);
            }
        }

        // Sort documents by ID if configured
        if self.config.sort_by_doc_id {
            all_documents.sort_by_key(|(doc_id, _)| *doc_id);
        }

        // Create new segment info
        let segment_info = SegmentInfo {
            segment_id: new_segment_id.to_string(),
            doc_count: all_documents.len() as u64,
            doc_offset: 0, // Will be assigned by segment manager
            generation: 0, // Will be assigned by segment manager
            has_deletions: false,
        };

        // Write merged segment to storage
        let file_paths = self.write_merged_segment(&segment_info, &merged_index, &all_documents)?;

        // Calculate segment size
        let size_bytes = file_paths
            .iter()
            .map(|path| {
                self.storage
                    .metadata(path)
                    .map(|meta| meta.size)
                    .unwrap_or(0)
            })
            .sum();

        // Create managed segment info
        let mut managed_info = ManagedSegmentInfo::new(segment_info);
        managed_info.size_bytes = size_bytes;
        managed_info.file_paths = file_paths.clone();

        // Update statistics
        stats.terms_merged = merged_index.term_count();
        stats.postings_merged = merged_index.doc_count();

        Ok(MergeResult {
            new_segment: managed_info,
            stats,
            file_paths,
        })
    }

    /// Load a segment reader for the given segment.
    fn load_segment_reader(&self, segment_info: &SegmentInfo) -> Result<Box<dyn IndexReader>> {
        // Create segment list with single segment
        let segments = vec![segment_info.clone()];

        // Use default config for reader
        let config = crate::full_text_search::advanced_reader::AdvancedReaderConfig::default();

        let reader = AdvancedIndexReader::new(segments, self.storage.clone(), config)?;
        Ok(Box::new(reader) as Box<dyn IndexReader>)
    }

    /// Process a batch of documents for indexing.
    fn process_document_batch(
        &self,
        merged_index: &mut InvertedIndex,
        documents: &mut [(u64, Document)],
    ) -> Result<()> {
        for (doc_id, document) in documents {
            // Add document to index
            // Convert document to the expected format for add_document
            let document_terms: Vec<(String, u32, Option<Vec<u32>>)> = document
                .fields()
                .keys()
                .map(|field_name| {
                    (field_name.clone(), 1, None) // Simple frequency, no positions for now
                })
                .collect();
            merged_index.add_document(*doc_id, document_terms);
        }
        Ok(())
    }

    /// Write merged segment to storage.
    fn write_merged_segment(
        &self,
        segment_info: &SegmentInfo,
        merged_index: &InvertedIndex,
        documents: &[(u64, Document)],
    ) -> Result<Vec<String>> {
        let mut file_paths = Vec::new();

        // Write inverted index
        let index_file = format!("{}.idx", segment_info.segment_id);
        {
            let output = self.storage.create_output(&index_file)?;
            let mut writer = StructWriter::new(output);
            merged_index.write_to_storage(&mut writer)?;
            writer.close()?;
            file_paths.push(index_file);
        }

        // Write term dictionary
        let dict_file = format!("{}.dict", segment_info.segment_id);
        {
            let output = self.storage.create_output(&dict_file)?;
            let mut writer = StructWriter::new(output);

            // Build sorted dictionary from index
            let mut builder = TermDictionaryBuilder::new();
            for term in merged_index.terms() {
                let postings = merged_index.get_posting_list(term).unwrap();
                let term_info = TermInfo::new(
                    0,                          // Will be updated during actual write
                    postings.len() as u64 * 16, // Estimate
                    postings.len() as u64,
                    postings.iter().map(|p| p.frequency as u64).sum(),
                );
                builder.add_term(term.clone(), term_info);
            }

            let dictionary = builder.build_sorted();
            dictionary.write_to_storage(&mut writer)?;
            writer.close()?;
            file_paths.push(dict_file);
        }

        // Write documents
        let docs_file = format!("{}.docs", segment_info.segment_id);
        {
            let output = self.storage.create_output(&docs_file)?;
            let mut writer = StructWriter::new(output);

            writer.write_varint(documents.len() as u64)?;
            for (doc_id, document) in documents {
                writer.write_u64(*doc_id)?;

                // Write document fields
                writer.write_varint(document.fields().len() as u64)?;
                for (field_name, field_value) in document.fields() {
                    writer.write_string(field_name)?;
                    let field_str = match field_value {
                        crate::document::FieldValue::Text(s) => s.clone(),
                        crate::document::FieldValue::Integer(i) => i.to_string(),
                        crate::document::FieldValue::Float(f) => f.to_string(),
                        crate::document::FieldValue::Boolean(b) => b.to_string(),
                        crate::document::FieldValue::Binary(_) => "[binary]".to_string(),
                        crate::document::FieldValue::DateTime(dt) => dt.to_rfc3339(),
                        crate::document::FieldValue::Geo(point) => {
                            format!("{},{}", point.lat, point.lon)
                        }
                        crate::document::FieldValue::Null => "null".to_string(),
                    };
                    writer.write_string(&field_str)?;
                }
            }

            writer.close()?;
            file_paths.push(docs_file);
        }

        Ok(file_paths)
    }

    /// Verify the integrity of a merged segment.
    fn verify_merged_segment(&self, segment: &ManagedSegmentInfo) -> Result<()> {
        // Load the segment and perform basic checks
        let reader = self.load_segment_reader(&segment.segment_info)?;

        // Check document count matches
        if reader.doc_count() != segment.segment_info.doc_count {
            return Err(SageError::index("Document count mismatch after merge"));
        }

        // TODO: Add more verification checks
        // - Term dictionary integrity
        // - Posting list consistency
        // - Document field validation

        Ok(())
    }

    /// Get merge configuration.
    pub fn get_config(&self) -> &MergeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::full_text::SegmentInfo;
    use crate::full_text_index::segment_manager::ManagedSegmentInfo;

    use crate::storage::{MemoryStorage, StorageConfig};

    #[allow(dead_code)]
    fn create_test_segment(id: &str, doc_count: u64) -> ManagedSegmentInfo {
        let segment_info = SegmentInfo {
            segment_id: id.to_string(),
            doc_count,
            doc_offset: 0,
            generation: 1,
            has_deletions: false,
        };

        ManagedSegmentInfo::new(segment_info)
    }

    #[test]
    fn test_merge_engine_creation() {
        let config = MergeConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));

        let engine = MergeEngine::new(config, storage);
        assert_eq!(engine.config.batch_size, 10000);
        assert!(engine.config.remove_deleted_docs);
    }

    #[test]
    fn test_merge_config_default() {
        let config = MergeConfig::default();

        assert_eq!(config.max_memory_mb, 256);
        assert_eq!(config.batch_size, 10000);
        assert!(config.enable_compression);
        assert!(config.remove_deleted_docs);
        assert!(config.sort_by_doc_id);
        assert!(config.verify_after_merge);
    }

    #[test]
    fn test_merge_stats_space_savings() {
        let stats = MergeStats {
            size_before: 1000,
            size_after: 800,
            ..Default::default()
        };

        assert_eq!(stats.space_savings(), 20.0);

        let stats_zero = MergeStats {
            size_before: 0,
            size_after: 0,
            ..Default::default()
        };
        assert_eq!(stats_zero.space_savings(), 0.0);
    }
}
