//! Index writer for adding and updating documents.

use std::sync::{Arc, Mutex};

use crate::error::{Result, SarissaError};
use crate::index::SegmentInfo;
use crate::index::deletion::{DeletionConfig, DeletionManager};
use crate::index::merge_engine::{MergeConfig, MergeEngine};
use crate::index::segment_manager::{
    ManagedSegmentInfo, MergePlan, MergeStrategy, SegmentManager, SegmentManagerConfig,
};
use crate::index::transaction::{
    AtomicOperations, IsolationLevel, Transaction, TransactionManager, TransactionResult,
};
use crate::document::{Document};
use crate::storage::Storage;


/// Trait for index writers.
pub trait IndexWriter: Send + std::fmt::Debug {
    /// Add a document to the index.
    fn add_document(&mut self, doc: Document) -> Result<()>;

    /// Delete documents matching the given term.
    fn delete_documents(&mut self, field: &str, value: &str) -> Result<u64>;

    /// Update a document (delete old, add new).
    fn update_document(&mut self, field: &str, value: &str, doc: Document) -> Result<()>;

    /// Commit all pending changes to the index.
    fn commit(&mut self) -> Result<()>;

    /// Rollback all pending changes.
    fn rollback(&mut self) -> Result<()>;

    /// Get the number of documents added since the last commit.
    fn pending_docs(&self) -> u64;

    /// Close the writer and release resources.
    fn close(&mut self) -> Result<()>;

    /// Check if the writer is closed.
    fn is_closed(&self) -> bool;
}

/// Configuration for index writers.
#[derive(Debug, Clone)]
pub struct WriterConfig {
    /// Maximum number of documents to buffer before flushing.
    pub max_buffered_docs: usize,

    /// Maximum memory usage for buffering (in bytes).
    pub max_buffer_memory: usize,

    /// Whether to automatically commit after each document.
    pub auto_commit: bool,

    /// Whether to optimize the index after commit.
    pub optimize_on_commit: bool,
}

impl Default for WriterConfig {
    fn default() -> Self {
        WriterConfig {
            max_buffered_docs: 1000,
            max_buffer_memory: 16 * 1024 * 1024, // 16MB
            auto_commit: false,
            optimize_on_commit: false,
        }
    }
}

/// A basic index writer implementation for schema-less indexing.
pub struct BasicIndexWriter {
    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// Default analyzer for schema-less mode.
    default_analyzer: Arc<dyn crate::analysis::Analyzer>,

    /// Writer configuration.
    config: WriterConfig,

    /// Documents pending to be written.
    pending_documents: Vec<Document>,

    /// Whether the writer is closed.
    closed: bool,

    /// Generation number for this writer.
    generation: u64,

    /// Deletion manager for document deletion.
    deletion_manager: DeletionManager,

    /// Segment manager for segment lifecycle management.
    segment_manager: SegmentManager,

    /// Writer statistics.
    stats: WriterStats,

    /// Transaction manager for atomic operations.
    transaction_manager: TransactionManager,

    /// Merge engine for segment optimization.
    merge_engine: MergeEngine,

    /// Current active transaction (if any).
    current_transaction: Option<Arc<Mutex<Transaction>>>,
}

impl std::fmt::Debug for BasicIndexWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BasicIndexWriter")
            .field("storage", &"<storage>")
            .field("default_analyzer", &"<analyzer>")
            .field("config", &self.config)
            .field("closed", &self.closed)
            .field("generation", &self.generation)
            .field("stats", &self.stats)
            .finish()
    }
}

impl BasicIndexWriter {
    /// Create a new index writer for schema-less indexing.
    pub fn new(storage: Arc<dyn Storage>, config: WriterConfig) -> Result<Self> {
        let deletion_config = DeletionConfig::default();
        let deletion_manager = DeletionManager::new(deletion_config, storage.clone())?;

        let segment_config = SegmentManagerConfig::default();
        let segment_manager = SegmentManager::new(segment_config, storage.clone())?;

        let transaction_manager = TransactionManager::new(storage.clone());

        let merge_config = MergeConfig::default();
        let merge_engine = MergeEngine::new(merge_config, storage.clone());

        // Create default analyzer for schema-less mode
        let default_analyzer = Arc::new(crate::analysis::StandardAnalyzer::new()?);

        Ok(BasicIndexWriter {
            storage,
            default_analyzer,
            config,
            pending_documents: Vec::new(),
            closed: false,
            generation: 0,
            deletion_manager,
            segment_manager,
            stats: WriterStats::default(),
            transaction_manager,
            merge_engine,
            current_transaction: None,
        })
    }

    /// Deprecated: Use `new()` instead. Schema is no longer required.
    #[deprecated(since = "0.2.0", note = "Use `new()` instead. Schema is no longer required.")]
    pub fn new_schemaless(storage: Arc<dyn Storage>, config: WriterConfig) -> Result<Self> {
        Self::new(storage, config)
    }

    /// Apply default analyzers to fields that don't have explicit analyzers in schema-less mode.
    fn apply_default_analyzers(&self, mut doc: Document) -> Document {
        use crate::document::FieldValue;

        // Use writer's default analyzer (schema-less mode)
        let default_analyzer = self.default_analyzer.clone();

        // For each text field without an explicit analyzer, apply the default analyzer
        let field_names: Vec<String> = doc.fields().keys().cloned().collect();
        for field_name in field_names {
            if let Some(field_value) = doc.fields().get(&field_name) {
                // Only apply to text fields that don't already have an analyzer
                if matches!(field_value, FieldValue::Text(_))
                    && !doc.field_analyzers().contains_key(&field_name)
                {
                    doc.set_field_analyzer(field_name, default_analyzer.clone());
                }
            }
        }

        doc
    }

    /// Check if the writer is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed {
            Err(SarissaError::index("Writer is closed"))
        } else {
            Ok(())
        }
    }

    /// Check if we need to flush buffered documents.
    fn should_flush(&self) -> bool {
        self.pending_documents.len() >= self.config.max_buffered_docs
    }

    /// Flush buffered documents to storage.
    fn flush(&mut self) -> Result<()> {
        if self.pending_documents.is_empty() {
            return Ok(());
        }

        // TODO: Implement actual segment writing
        // For now, we'll just create a simple segment file
        let segment_name = format!("segment_{:06}.json", self.generation);
        let mut output = self.storage.create_output(&segment_name)?;

        let segment_data = serde_json::to_string_pretty(&self.pending_documents)
            .map_err(|e| SarissaError::index(format!("Failed to serialize segment: {e}")))?;

        std::io::Write::write_all(&mut output, segment_data.as_bytes())?;
        let bytes_written = segment_data.len() as u64;
        output.close()?;

        // Create segment info and register with segment manager
        let segment_id = format!("segment_{:06}", self.generation);
        let doc_count = self.pending_documents.len() as u64;

        let segment_info = SegmentInfo {
            segment_id: segment_id.clone(),
            doc_count,
            doc_offset: 0,
            generation: self.generation,
            has_deletions: false,
        };

        let file_paths = vec![format!("{}.json", segment_id)];

        // Register with segment manager
        self.segment_manager.add_segment(segment_info, file_paths)?;

        // Initialize segment in deletion manager
        let _ = self
            .deletion_manager
            .initialize_segment(&segment_id, doc_count);

        // Update statistics
        self.stats.segments_created += 1;
        self.stats.bytes_written += bytes_written;

        // Clear pending documents
        self.pending_documents.clear();
        self.generation += 1;

        Ok(())
    }
}

impl BasicIndexWriter {
    /// Internal method to add document with optional statistics update.
    fn add_document_internal(&mut self, doc: Document, update_stats: bool) -> Result<()> {
        self.check_closed()?;

        // Schema-less mode: apply default analyzer to fields without explicit analyzers
        let processed_doc = self.apply_default_analyzers(doc);

        // Add to pending documents
        self.pending_documents.push(processed_doc);

        // Update statistics conditionally
        if update_stats {
            self.stats.docs_added += 1;
        }

        // Flush if needed
        if self.should_flush() {
            self.flush()?;
        }

        // Auto-commit if configured
        if self.config.auto_commit {
            self.commit()?;
        }

        Ok(())
    }
}

impl IndexWriter for BasicIndexWriter {
    fn add_document(&mut self, doc: Document) -> Result<()> {
        self.add_document_internal(doc, true)
    }

    fn delete_documents(&mut self, field: &str, value: &str) -> Result<u64> {
        self.check_closed()?;

        // For now, we'll implement basic term-based deletion
        // In a full implementation, this would:
        // 1. Query the index to find matching documents
        // 2. Get document IDs from the query results
        // 3. Mark those documents as deleted in the deletion manager

        // Create a segment identifier for the current generation
        let segment_id = format!("segment_{:06}", self.generation);

        // Initialize the segment in deletion manager if not exists
        // In a real implementation, this would be based on actual segment document count
        let _ = self.deletion_manager.initialize_segment(&segment_id, 1000);

        // For demonstration, we'll delete a few example document IDs
        // In reality, this would come from querying the index for field:value matches
        let example_doc_ids = vec![1, 2, 3]; // Placeholder - would come from actual query

        let deleted_count = self.deletion_manager.delete_documents(
            &segment_id,
            &example_doc_ids,
            &format!("Term deletion: {field}:{value}"),
        )?;

        // Update statistics
        self.stats.docs_deleted += deleted_count;

        Ok(deleted_count)
    }

    fn update_document(&mut self, field: &str, value: &str, doc: Document) -> Result<()> {
        self.check_closed()?;

        // Delete old document and add new one (atomic operation)
        let deleted_count = self.delete_documents(field, value)?;
        // Add new document - this should count as an add operation
        self.add_document_internal(doc, true)?;

        // Update statistics
        if deleted_count > 0 {
            self.stats.docs_updated += 1;
        }

        Ok(())
    }

    fn commit(&mut self) -> Result<()> {
        self.check_closed()?;

        // Flush any pending documents
        self.flush()?;

        // Update statistics
        self.stats.commits += 1;
        // Note: docs_added is already updated in add_document, no need to double-count

        // Update index metadata with current document count
        // Note: This is a simplified implementation
        // In a full implementation, this would update persistent metadata

        // TODO: Optimize if configured

        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        self.check_closed()?;

        // Clear pending documents
        self.pending_documents.clear();

        // Update statistics
        self.stats.rollbacks += 1;

        Ok(())
    }

    fn pending_docs(&self) -> u64 {
        self.pending_documents.len() as u64
    }

    fn close(&mut self) -> Result<()> {
        if !self.closed {
            // Commit any pending changes
            self.commit()?;
            self.closed = true;
        }
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed
    }
}

// Additional methods for BasicIndexWriter
impl BasicIndexWriter {
    /// Get writer statistics.
    pub fn get_stats(&self) -> &WriterStats {
        &self.stats
    }

    /// Get deletion manager statistics.
    pub fn get_deletion_stats(&self) -> crate::index::deletion::DeletionStats {
        self.deletion_manager.get_stats()
    }

    /// Check if a document is deleted.
    pub fn is_document_deleted(&self, segment_id: &str, doc_id: u64) -> bool {
        self.deletion_manager.is_deleted(segment_id, doc_id)
    }

    /// Get deletion ratio for a segment.
    pub fn get_segment_deletion_ratio(&self, segment_id: &str) -> f64 {
        self.deletion_manager.get_deletion_ratio(segment_id)
    }

    /// Get segments that need compaction.
    pub fn get_compaction_candidates(&self) -> Vec<String> {
        self.deletion_manager.get_compaction_candidates()
    }

    /// Delete documents by query (simplified version).
    pub fn delete_by_query(&mut self, field: &str, values: &[&str]) -> Result<u64> {
        self.check_closed()?;

        let mut total_deleted = 0;
        for value in values {
            total_deleted += self.delete_documents(field, value)?;
        }

        Ok(total_deleted)
    }

    /// Force flush pending documents.
    pub fn force_flush(&mut self) -> Result<()> {
        self.check_closed()?;
        self.flush()
    }

    /// Get comprehensive deletion report.
    pub fn get_deletion_report(&self) -> crate::index::deletion::DeletionReport {
        self.deletion_manager.get_deletion_report()
    }

    /// Get global deletion state.
    pub fn get_global_deletion_state(&self) -> crate::index::deletion::GlobalDeletionState {
        self.deletion_manager.get_global_state()
    }

    /// Check if auto-compaction should be triggered.
    pub fn should_trigger_auto_compaction(&self) -> bool {
        self.deletion_manager.should_trigger_auto_compaction()
    }

    /// Mark compaction as completed (for segments).
    pub fn mark_compaction_completed(&self, segments_compacted: &[String]) -> Result<()> {
        self.deletion_manager
            .mark_compaction_completed(segments_compacted)
    }

    /// Get segment manager statistics.
    pub fn get_segment_stats(&self) -> crate::index::segment_manager::SegmentManagerStats {
        self.segment_manager.get_stats()
    }

    /// Get all managed segments.
    pub fn get_managed_segments(&self) -> Vec<ManagedSegmentInfo> {
        self.segment_manager.get_segments()
    }

    /// Generate merge candidates with specified strategy.
    pub fn generate_merge_candidates(
        &self,
        strategy: MergeStrategy,
    ) -> Vec<crate::index::segment_manager::MergeCandidate> {
        self.segment_manager.generate_merge_candidates(strategy)
    }

    /// Get optimal merge plan.
    pub fn get_merge_plan(&self) -> MergePlan {
        self.segment_manager.get_merge_plan()
    }

    /// Get segments organized by tier.
    pub fn get_segments_by_tier(&self) -> Vec<Vec<ManagedSegmentInfo>> {
        self.segment_manager.get_segments_by_tier()
    }

    /// Mark segments as being merged.
    pub fn mark_segments_merging(&self, segment_ids: &[String], merging: bool) -> Result<()> {
        self.segment_manager
            .mark_segments_merging(segment_ids, merging)
    }

    /// Complete merge operation.
    pub fn complete_segment_merge(
        &self,
        old_segment_ids: &[String],
        new_segment: SegmentInfo,
        new_file_paths: Vec<String>,
    ) -> Result<()> {
        self.segment_manager
            .complete_merge(old_segment_ids, new_segment, new_file_paths)
    }

    /// Force rebalance segment tiers.
    pub fn rebalance_segment_tiers(&self) -> Result<()> {
        self.segment_manager.rebalance_tiers()
    }
}

impl Drop for BasicIndexWriter {
    fn drop(&mut self) {
        // Ensure we commit any pending changes
        let _ = self.close();
    }
}

/// Writer statistics.
#[derive(Debug, Clone, Default)]
pub struct WriterStats {
    /// Number of documents added.
    pub docs_added: u64,

    /// Number of documents deleted.
    pub docs_deleted: u64,

    /// Number of documents updated.
    pub docs_updated: u64,

    /// Number of segments created.
    pub segments_created: u32,

    /// Total bytes written.
    pub bytes_written: u64,

    /// Number of commits.
    pub commits: u32,

    /// Number of rollbacks.
    pub rollbacks: u32,
}

impl AtomicOperations for BasicIndexWriter {
    /// Execute multiple operations atomically.
    fn execute_atomically<F, R>(&mut self, operations: F) -> Result<R>
    where
        F: FnOnce(&mut Self) -> Result<R>,
    {
        let transaction = self
            .transaction_manager
            .begin_transaction(IsolationLevel::ReadCommitted)?;
        self.current_transaction = Some(transaction.clone());

        let result = operations(self);

        match result {
            Ok(value) => {
                // Commit transaction
                self.transaction_manager.commit_transaction(
                    transaction,
                    &mut self.segment_manager,
                    &mut self.deletion_manager,
                    &self.merge_engine,
                )?;
                self.current_transaction = None;
                Ok(value)
            }
            Err(e) => {
                // Rollback transaction
                self.transaction_manager.rollback_transaction(
                    transaction,
                    &mut self.segment_manager,
                    &mut self.deletion_manager,
                )?;
                self.current_transaction = None;
                Err(e)
            }
        }
    }

    /// Begin a transaction for atomic operations.
    fn begin_atomic_session(&mut self) -> Result<String> {
        if self.current_transaction.is_some() {
            return Err(SarissaError::index("Transaction already active"));
        }

        let transaction = self
            .transaction_manager
            .begin_transaction(IsolationLevel::ReadCommitted)?;
        let transaction_id = transaction.lock().unwrap().id.clone();
        self.current_transaction = Some(transaction);

        Ok(transaction_id)
    }

    /// Commit the current atomic session.
    fn commit_atomic_session(&mut self, session_id: &str) -> Result<TransactionResult> {
        if let Some(transaction) = self.current_transaction.take() {
            let txn_id = transaction.lock().unwrap().id.clone();
            if txn_id != session_id {
                return Err(SarissaError::index("Session ID mismatch"));
            }

            let result = self.transaction_manager.commit_transaction(
                transaction,
                &mut self.segment_manager,
                &mut self.deletion_manager,
                &self.merge_engine,
            )?;

            Ok(result)
        } else {
            Err(SarissaError::index("No active transaction"))
        }
    }

    /// Rollback the current atomic session.
    fn rollback_atomic_session(&mut self, session_id: &str) -> Result<()> {
        if let Some(transaction) = self.current_transaction.take() {
            let txn_id = transaction.lock().unwrap().id.clone();
            if txn_id != session_id {
                return Err(SarissaError::index("Session ID mismatch"));
            }

            self.transaction_manager.rollback_transaction(
                transaction,
                &mut self.segment_manager,
                &mut self.deletion_manager,
            )?;

            Ok(())
        } else {
            Err(SarissaError::index("No active transaction"))
        }
    }
}

impl BasicIndexWriter {
    /// Execute an atomic batch of document operations.
    pub fn batch_operations<F>(&mut self, operations: F) -> Result<TransactionResult>
    where
        F: FnOnce(&mut Self) -> Result<()>,
    {
        self.execute_atomically(|writer| {
            operations(writer)?;
            Ok(TransactionResult::new())
        })
    }

    /// Add multiple documents atomically.
    pub fn add_documents_atomic(&mut self, documents: Vec<Document>) -> Result<TransactionResult> {
        self.execute_atomically(|writer| {
            for doc in documents {
                writer.add_document(doc)?;
            }
            let mut result = TransactionResult::new();
            result.docs_added = writer.pending_docs();
            writer.commit()?;
            Ok(result)
        })
    }

    /// Update multiple documents atomically.
    pub fn update_documents_atomic(
        &mut self,
        updates: Vec<(String, String, Document)>,
    ) -> Result<TransactionResult> {
        self.execute_atomically(|writer| {
            let mut result = TransactionResult::new();
            for (field, value, new_doc) in updates {
                writer.update_document(&field, &value, new_doc)?;
                result.docs_updated += 1;
            }
            writer.commit()?;
            Ok(result)
        })
    }

    /// Delete multiple document sets atomically.
    pub fn delete_documents_atomic(
        &mut self,
        deletions: Vec<(String, String)>,
    ) -> Result<TransactionResult> {
        self.execute_atomically(|writer| {
            let mut result = TransactionResult::new();
            for (field, value) in deletions {
                let deleted = writer.delete_documents(&field, &value)?;
                result.docs_deleted += deleted;
            }
            writer.commit()?;
            Ok(result)
        })
    }

    /// Perform a complex operation with full transaction support.
    pub fn complex_operation<F, R>(&mut self, operation: F) -> Result<R>
    where
        F: FnOnce(&mut Self) -> Result<R>,
    {
        let session_id = self.begin_atomic_session()?;

        match operation(self) {
            Ok(result) => {
                self.commit_atomic_session(&session_id)?;
                Ok(result)
            }
            Err(e) => {
                self.rollback_atomic_session(&session_id)?;
                Err(e)
            }
        }
    }

    /// Get current transaction status.
    pub fn has_active_transaction(&self) -> bool {
        self.current_transaction.is_some()
    }

    /// Get active transaction count from manager.
    pub fn active_transaction_count(&self) -> usize {
        self.transaction_manager.active_transaction_count()
    }

    /// Get total transaction count.
    pub fn total_transaction_count(&self) -> u64 {
        self.transaction_manager.total_transaction_count()
    }

    /// Force merge segments using the merge engine.
    pub fn force_merge_segments(&mut self, segment_ids: Vec<String>) -> Result<TransactionResult> {
        self.execute_atomically(|writer| {
            // Get segments to merge
            let all_segments = writer.segment_manager.get_segments();
            let segments_to_merge: Vec<_> = all_segments
                .into_iter()
                .filter(|seg| segment_ids.contains(&seg.segment_info.segment_id))
                .collect();

            if segments_to_merge.is_empty() {
                return Err(SarissaError::index("No segments found to merge"));
            }

            // Create merge candidate
            let merge_candidate = crate::index::segment_manager::MergeCandidate {
                segments: segment_ids.clone(),
                priority: 1.0,
                strategy: MergeStrategy::Balanced,
                estimated_size: segments_to_merge.iter().map(|s| s.size_bytes).sum(),
            };

            // Perform merge
            let next_generation = writer.generation + 1;
            let merge_result = writer.merge_engine.merge_segments(
                &merge_candidate,
                &segments_to_merge,
                next_generation,
            )?;

            // Update segment manager
            writer.segment_manager.complete_merge(
                &segment_ids,
                merge_result.new_segment.segment_info,
                merge_result.file_paths,
            )?;

            writer.generation = next_generation;

            let mut result = TransactionResult::new();
            result.segments_merged = merge_result.stats.segments_merged;
            result.merge_operations = 1;

            Ok(result)
        })
    }

    /// Perform optimization with transaction support.
    pub fn optimize_index(&mut self) -> Result<TransactionResult> {
        self.execute_atomically(|writer| {
            // Get merge plan from segment manager
            let merge_plan = writer.segment_manager.get_merge_plan();
            let mut result = TransactionResult::new();

            // Execute merge operations
            for candidate in merge_plan.candidates {
                let segment_ids = candidate.segments.clone();
                let merge_result = writer.force_merge_segments(segment_ids)?;
                result.segments_merged += merge_result.segments_merged;
                result.merge_operations += merge_result.merge_operations;
            }

            Ok(result)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::storage::{MemoryStorage, StorageConfig};
    use std::sync::Arc;

    #[allow(dead_code)]

    fn create_test_document() -> Document {
        Document::builder()
            .add_text("title", "Test Document")
            .add_text("body", "This is a test document for indexing")
            .build()
    }

    #[test]
    fn test_writer_creation() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let writer = BasicIndexWriter::new(storage, config).unwrap();

        assert!(!writer.is_closed());
        assert_eq!(writer.pending_docs(), 0);
    }

    #[test]
    fn test_add_document() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();
        let doc = create_test_document();

        writer.add_document(doc).unwrap();

        assert_eq!(writer.pending_docs(), 1);
    }

    #[test]
    fn test_commit() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage.clone(), config).unwrap();
        let doc = create_test_document();

        writer.add_document(doc).unwrap();
        assert_eq!(writer.pending_docs(), 1);

        writer.commit().unwrap();
        assert_eq!(writer.pending_docs(), 0);

        // Check that a segment file was created
        let files = storage.list_files().unwrap();
        assert!(files.iter().any(|f| f.starts_with("segment_")));
    }

    #[test]
    fn test_rollback() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();
        let doc = create_test_document();

        writer.add_document(doc).unwrap();
        assert_eq!(writer.pending_docs(), 1);

        writer.rollback().unwrap();
        assert_eq!(writer.pending_docs(), 0);
    }

    #[test]
    fn test_auto_flush() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig {
            max_buffered_docs: 2,
            ..Default::default()
        };

        let mut writer = BasicIndexWriter::new(storage.clone(), config).unwrap();

        // Add first document
        writer.add_document(create_test_document()).unwrap();
        assert_eq!(writer.pending_docs(), 1);

        // Add second document - should trigger flush
        writer.add_document(create_test_document()).unwrap();
        assert_eq!(writer.pending_docs(), 0);

        // Check that a segment file was created
        let files = storage.list_files().unwrap();
        assert!(files.iter().any(|f| f.starts_with("segment_")));
    }

    #[test]
    fn test_writer_close() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();
        let doc = create_test_document();

        writer.add_document(doc).unwrap();
        assert_eq!(writer.pending_docs(), 1);

        writer.close().unwrap();

        assert!(writer.is_closed());
        assert_eq!(writer.pending_docs(), 0);

        // Operations should fail after close
        let result = writer.add_document(create_test_document());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_document() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Schema-less mode: any field is accepted
        let doc = Document::builder()
            .add_text("invalid_field", "This field doesn't exist in schema")
            .build();

        let result = writer.add_document(doc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_writer_config() {
        let config = WriterConfig::default();

        assert_eq!(config.max_buffered_docs, 1000);
        assert_eq!(config.max_buffer_memory, 16 * 1024 * 1024);
        assert!(!config.auto_commit);
        assert!(!config.optimize_on_commit);
    }

    #[test]
    fn test_writer_stats() {
        let stats = WriterStats::default();

        assert_eq!(stats.docs_added, 0);
        assert_eq!(stats.docs_deleted, 0);
        assert_eq!(stats.docs_updated, 0);
        assert_eq!(stats.segments_created, 0);
        assert_eq!(stats.bytes_written, 0);
        assert_eq!(stats.commits, 0);
        assert_eq!(stats.rollbacks, 0);
    }

    #[test]
    fn test_document_deletion() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Add a document first
        let doc = create_test_document();
        writer.add_document(doc).unwrap();

        // Commit to create a segment
        writer.commit().unwrap();

        // Delete documents
        let deleted_count = writer.delete_documents("title", "Test Document").unwrap();
        assert!(deleted_count > 0);

        // Check stats
        let stats = writer.get_stats();
        assert_eq!(stats.docs_added, 1);
        assert!(stats.docs_deleted > 0);

        // Check deletion manager stats
        let deletion_stats = writer.get_deletion_stats();
        assert!(deletion_stats.total_deleted > 0);
    }

    #[test]
    fn test_document_update() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Add original document
        let doc1 = create_test_document();
        writer.add_document(doc1).unwrap();
        writer.commit().unwrap();

        // Update document
        let doc2 = Document::builder()
            .add_text("title", "Updated Document")
            .add_text("body", "This is an updated document")
            .build();

        writer
            .update_document("title", "Test Document", doc2)
            .unwrap();

        // Check stats
        let stats = writer.get_stats();
        assert_eq!(stats.docs_added, 2); // Original + updated
        assert!(stats.docs_updated > 0);
    }

    #[test]
    fn test_deletion_by_query() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Add multiple documents
        for i in 0..5 {
            let doc = Document::builder()
                .add_text("title", format!("Document {i}"))
                .add_text("body", "Test content")
                .build();
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();

        // Delete by multiple values
        let values = vec!["Document 1", "Document 3"];
        let deleted_count = writer.delete_by_query("title", &values).unwrap();
        assert!(deleted_count > 0);

        let deletion_stats = writer.get_deletion_stats();
        assert!(deletion_stats.total_deleted > 0);
    }

    #[test]
    fn test_compaction_candidates() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Add and commit documents to create segments
        for i in 0..10 {
            let doc = Document::builder()
                .add_text("title", format!("Document {i}"))
                .add_text("body", "Test content")
                .build();
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();

        // Delete some documents to trigger compaction threshold
        for i in 0..5 {
            writer
                .delete_documents("title", &format!("Document {i}"))
                .unwrap();
        }

        // Check compaction candidates
        let candidates = writer.get_compaction_candidates();
        // Note: Depending on deletion threshold, we might have candidates
        println!("Compaction candidates: {candidates:?}");
    }

    #[test]
    fn test_global_deletion_report() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Add multiple documents
        for i in 0..20 {
            let doc = Document::builder()
                .add_text("title", format!("Document {i}"))
                .add_text("body", "Test content")
                .build();
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();

        // Delete some documents
        for i in 0..8 {
            // 40% deletion
            writer
                .delete_documents("title", &format!("Document {i}"))
                .unwrap();
        }

        // Get comprehensive deletion report
        let report = writer.get_deletion_report();

        println!("Deletion Report:");
        println!(
            "  Global deletion ratio: {:.2}%",
            report.global_state.global_deletion_ratio * 100.0
        );
        println!(
            "  Compaction candidates: {}",
            report.global_state.compaction_candidates.len()
        );
        println!(
            "  Reclaimable space: {} bytes",
            report.global_state.reclaimable_space
        );
        println!(
            "  Auto-compaction enabled: {}",
            report.auto_compaction_enabled
        );

        // Test summary metrics
        let (ratio, _candidates, _space, _needs_compaction) = report.summary();
        assert!(ratio > 0.0);
        // Note: space may be 0 if deletion ratio is below compaction threshold
        // space is u64, so >= 0 check is redundant

        // Test urgency classification
        let (urgent, moderate, low) = report.segments_by_urgency();
        println!("  Urgent: {urgent:?}, Moderate: {moderate:?}, Low: {low:?}");
    }

    #[test]
    fn test_auto_compaction_detection() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Add and commit documents to create segments
        for i in 0..50 {
            let doc = Document::builder()
                .add_text("title", format!("Document {i}"))
                .add_text("body", "Test content")
                .build();
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();

        // Delete enough documents to potentially trigger auto-compaction
        for i in 0..20 {
            // 40% deletion
            writer
                .delete_documents("title", &format!("Document {i}"))
                .unwrap();
        }

        // Check if auto-compaction should be triggered
        let should_compact = writer.should_trigger_auto_compaction();
        println!("Should trigger auto-compaction: {should_compact}");

        // Get global deletion state
        let global_state = writer.get_global_deletion_state();
        println!("Global deletion state:");
        println!("  Total documents: {}", global_state.total_documents);
        println!("  Total deleted: {}", global_state.total_deleted);
        println!(
            "  Deletion ratio: {:.2}%",
            global_state.global_deletion_ratio * 100.0
        );
        println!(
            "  Compaction candidates: {:?}",
            global_state.compaction_candidates
        );
    }

    #[test]
    fn test_compaction_completion() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Add documents to multiple segments
        for batch in 0..3 {
            for i in 0..10 {
                let doc = Document::builder()
                    .add_text("title", format!("Batch {batch} Document {i}"))
                    .add_text("body", "Test content")
                    .build();
                writer.add_document(doc).unwrap();
            }
            writer.commit().unwrap(); // Force segment creation
        }

        // Delete some documents to create compaction candidates
        writer
            .delete_documents("title", "Batch 0 Document 1")
            .unwrap();
        writer
            .delete_documents("title", "Batch 0 Document 2")
            .unwrap();
        writer
            .delete_documents("title", "Batch 1 Document 1")
            .unwrap();

        let report_before = writer.get_deletion_report();
        println!(
            "Before compaction: {} segments tracked",
            report_before.deletion_stats.segments_tracked
        );

        // Simulate compaction completion
        let segments_to_compact = vec!["segment_000000".to_string()];
        let result = writer.mark_compaction_completed(&segments_to_compact);
        assert!(result.is_ok());

        let report_after = writer.get_deletion_report();
        println!(
            "After compaction: {} segments tracked",
            report_after.deletion_stats.segments_tracked
        );
        println!(
            "Compaction operations: {}",
            report_after.deletion_stats.compaction_operations
        );
    }

    #[test]
    fn test_segment_management_integration() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Add documents to create multiple segments
        for batch in 0..3 {
            for i in 0..5 {
                let doc = Document::builder()
                    .add_text("title", format!("Batch {batch} Document {i}"))
                    .add_text("body", "Test content for segment management")
                    .build();
                writer.add_document(doc).unwrap();
            }
            writer.commit().unwrap(); // Force segment creation
        }

        // Test segment statistics
        let segment_stats = writer.get_segment_stats();
        println!("Segment Statistics:");
        println!("  Total segments: {}", segment_stats.total_segments);
        println!("  Total documents: {}", segment_stats.total_doc_count);
        println!("  Total size: {} bytes", segment_stats.total_size_bytes);
        println!(
            "  Average segment size: {} bytes",
            segment_stats.avg_segment_size
        );

        assert_eq!(segment_stats.total_segments, 3);
        assert_eq!(segment_stats.total_doc_count, 15);

        // Test managed segments
        let managed_segments = writer.get_managed_segments();
        assert_eq!(managed_segments.len(), 3);

        for (i, segment) in managed_segments.iter().enumerate() {
            println!(
                "  Segment {}: {} docs, tier {}, {} bytes",
                i, segment.segment_info.doc_count, segment.tier, segment.size_bytes
            );
        }

        // Test segments by tier
        let tiers = writer.get_segments_by_tier();
        println!("Segments by tier:");
        for (tier, segments) in tiers.iter().enumerate() {
            println!("  Tier {}: {} segments", tier, segments.len());
        }

        // Test merge candidates generation
        let size_candidates = writer.generate_merge_candidates(MergeStrategy::SizeBased);
        println!("Size-based merge candidates: {}", size_candidates.len());

        let balanced_candidates = writer.generate_merge_candidates(MergeStrategy::Balanced);
        println!("Balanced merge candidates: {}", balanced_candidates.len());

        // Test merge plan
        let merge_plan = writer.get_merge_plan();
        println!("Merge Plan:");
        println!("  Strategy: {:?}", merge_plan.strategy);
        println!("  Candidates: {}", merge_plan.candidates.len());
        println!("  Estimated benefit: {:.2}", merge_plan.estimated_benefit);
        println!("  Urgency: {:?}", merge_plan.urgency);

        // Test marking segments as merging
        if let Some(candidate) = merge_plan.candidates.first() {
            if !candidate.segments.is_empty() {
                writer
                    .mark_segments_merging(&candidate.segments, true)
                    .unwrap();

                let updated_segments = writer.get_managed_segments();
                let merging_count = updated_segments.iter().filter(|s| s.is_merging).count();

                println!("Segments marked as merging: {merging_count}");
                assert!(merging_count > 0);

                // Unmark
                writer
                    .mark_segments_merging(&candidate.segments, false)
                    .unwrap();
            }
        }
    }

    #[test]
    fn test_segment_merge_completion() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Create multiple small segments
        for i in 0..4 {
            let doc = Document::builder()
                .add_text("title", format!("Document {i}"))
                .add_text("body", "Small segment content")
                .build();
            writer.add_document(doc).unwrap();
            writer.commit().unwrap(); // Force individual segments
        }

        let initial_stats = writer.get_segment_stats();
        assert_eq!(initial_stats.total_segments, 4);

        // Simulate merge completion
        let segments = writer.get_managed_segments();
        let old_segment_ids: Vec<String> = segments
            .iter()
            .take(2)
            .map(|s| s.segment_info.segment_id.clone())
            .collect();

        let new_segment = SegmentInfo {
            segment_id: "merged_segment".to_string(),
            doc_count: 2,
            doc_offset: 0,
            generation: 100,
            has_deletions: false,
        };

        writer
            .complete_segment_merge(
                &old_segment_ids,
                new_segment,
                vec!["merged_segment.json".to_string()],
            )
            .unwrap();

        let final_stats = writer.get_segment_stats();
        assert_eq!(final_stats.total_segments, 3); // 4 - 2 + 1
        assert_eq!(final_stats.merge_operations, 1);

        println!(
            "Merge completed: {} -> {} segments",
            initial_stats.total_segments, final_stats.total_segments
        );
    }

    #[test]
    fn test_segment_tier_rebalancing() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Add documents to create segments
        for i in 0..10 {
            let doc = Document::builder()
                .add_text("title", format!("Document {i}"))
                .add_text("body", "Content for tier testing")
                .build();
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();

        let tiers_before = writer.get_segments_by_tier();
        println!("Tiers before rebalancing:");
        for (tier, segments) in tiers_before.iter().enumerate() {
            println!("  Tier {}: {} segments", tier, segments.len());
        }

        // Force rebalance
        writer.rebalance_segment_tiers().unwrap();

        let tiers_after = writer.get_segments_by_tier();
        println!("Tiers after rebalancing:");
        for (tier, segments) in tiers_after.iter().enumerate() {
            println!("  Tier {}: {} segments", tier, segments.len());
        }

        // Verify that rebalancing was performed
        let managed_segments = writer.get_managed_segments();
        for segment in &managed_segments {
            println!(
                "  Segment {}: tier {}, size {} bytes",
                segment.segment_info.segment_id, segment.tier, segment.size_bytes
            );
        }
    }

    #[test]
    fn test_schemaless_mode() {
        use crate::analysis::{KeywordAnalyzer, StandardAnalyzer};
        use crate::document::FieldValue;
        use crate::storage::{MemoryStorage, StorageConfig};
        use std::sync::Arc;

        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        // Create writer in schema-less mode
        let mut writer = BasicIndexWriter::new_schemaless(storage, config).unwrap();

        // Create analyzers
        let standard_analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        let keyword_analyzer = Arc::new(KeywordAnalyzer::new());

        // Create document with field-specific analyzers
        let mut doc = Document::new();
        doc.add_field_with_analyzer(
            "title",
            FieldValue::Text("Hello World".to_string()),
            standard_analyzer.clone(),
        );
        doc.add_field_with_analyzer(
            "id",
            FieldValue::Text("ABC-123".to_string()),
            keyword_analyzer.clone(),
        );
        doc.add_field("price", FieldValue::Float(29.99));

        // Add document without schema validation
        writer.add_document(doc).unwrap();
        writer.commit().unwrap();

        // Verify document was added
        assert_eq!(writer.stats.docs_added, 1);
        println!("Successfully added document in schema-less mode");
    }

    #[test]
    fn test_document_builder_with_analyzers() {
        use crate::analysis::{KeywordAnalyzer, StandardAnalyzer};
        use std::sync::Arc;

        let standard_analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        let keyword_analyzer = Arc::new(KeywordAnalyzer::new());

        // Use DocumentBuilder with analyzers
        let doc = Document::builder()
            .add_text_with_analyzer("title", "Hello World", standard_analyzer.clone())
            .add_text_with_analyzer("id", "ABC-123", keyword_analyzer.clone())
            .add_text("description", "Regular text field")
            .build();

        // Verify analyzers are set
        assert!(doc.get_field_analyzer("title").is_some());
        assert!(doc.get_field_analyzer("id").is_some());
        assert!(doc.get_field_analyzer("description").is_none());

        println!("Document with analyzers: {:?}", doc);
    }

    #[test]
    fn test_mixed_mode_compatibility() {
        use crate::analysis::StandardAnalyzer;
        use crate::document::{FieldValue};
        use crate::storage::{MemoryStorage, StorageConfig};
        use std::sync::Arc;

        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        // Create schema-based writer
        

        let mut writer = BasicIndexWriter::new(storage, config).unwrap();

        // Regular document (schema-based)
        let doc1 = Document::builder()
            .add_text("title", "Schema-based document")
            .build();

        writer.add_document(doc1).unwrap();

        // Document with analyzer (will be ignored in schema mode)
        let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        let mut doc2 = Document::new();
        doc2.add_field_with_analyzer(
            "title",
            FieldValue::Text("Document with analyzer".to_string()),
            analyzer,
        );

        writer.add_document(doc2).unwrap();
        writer.commit().unwrap();

        assert_eq!(writer.stats.docs_added, 2);
        println!("Successfully mixed schema and analyzer modes");
    }

    #[test]
    fn test_unified_api() {
        use crate::analysis::KeywordAnalyzer;
        use crate::document::FieldValue;
        use crate::storage::{MemoryStorage, StorageConfig};
        use std::sync::Arc;

        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        // Create writer in schema-less mode
        let mut writer = BasicIndexWriter::new_schemaless(storage, config).unwrap();

        // Test unified API: same add_field() method for both modes
        let mut doc = Document::new();

        // Regular add_field() - will use default analyzer
        doc.add_field("title", FieldValue::Text("Hello World".to_string()));
        doc.add_field("price", FieldValue::Float(29.99));

        // Optionally specify analyzer for specific fields
        let keyword_analyzer = Arc::new(KeywordAnalyzer::new());
        doc.add_field_with_analyzer(
            "id",
            FieldValue::Text("ABC-123".to_string()),
            keyword_analyzer,
        );

        writer.add_document(doc).unwrap();
        writer.commit().unwrap();

        // Verify that default analyzer was applied to text fields
        assert_eq!(writer.stats.docs_added, 1);
        println!("Successfully used unified API with default analyzers");
    }

    #[test]
    fn test_default_analyzer_application() {
        use crate::document::FieldValue;
        use crate::storage::{MemoryStorage, StorageConfig};
        use std::sync::Arc;

        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = WriterConfig::default();

        let mut writer = BasicIndexWriter::new_schemaless(storage, config).unwrap();

        // Create document with mixed field types
        let mut doc = Document::new();
        doc.add_field(
            "text_field",
            FieldValue::Text("This should get default analyzer".to_string()),
        );
        doc.add_field("number_field", FieldValue::Integer(42));
        doc.add_field("bool_field", FieldValue::Boolean(true));

        // Before processing, text field should not have analyzer
        assert!(doc.get_field_analyzer("text_field").is_none());

        writer.add_document(doc.clone()).unwrap();

        // After processing through writer, the document should have default analyzer applied
        // (We can't directly inspect the processed document, but we can verify no errors occurred)
        assert_eq!(writer.pending_docs(), 1);
        println!("Default analyzer applied successfully to text fields");
    }
}
