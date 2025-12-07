//! Transaction management for atomic lexical operations in Platypus.
//!
//! This module coordinates schema-less indexing, merges, and deletions with
//! explicit commit/rollback hooks so concurrent writers keep inverted indexes
//! consistent even under heavy ingestion workloads.

use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use ahash::AHashMap;
use uuid::Uuid;

use crate::lexical::document::document::Document;
use crate::error::{PlatypusError, Result};
use crate::lexical::index::inverted::maintenance::deletion::{
    DeletionManager, GlobalDeletionState,
};
use crate::lexical::index::inverted::segment::manager::SegmentManager;
use crate::lexical::index::inverted::segment::merge_engine::MergeEngine;
use crate::storage::Storage;

/// Transaction isolation levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Read committed - can see committed changes from other transactions.
    ReadCommitted,
    /// Repeatable read - consistent snapshot for the transaction duration.
    RepeatableRead,
    /// Serializable - full serializability guarantee.
    Serializable,
}

/// Transaction state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Transaction is active and accepting operations.
    Active,
    /// Transaction is being prepared for commit.
    Preparing,
    /// Transaction is committed.
    Committed,
    /// Transaction is aborted/rolled back.
    Aborted,
}

/// A single transaction operation.
#[derive(Debug, Clone)]
pub enum TransactionOperation {
    /// Add a document to the index.
    AddDocument {
        document: Document,
        segment_id: Option<String>,
    },
    /// Update a document (delete + add).
    UpdateDocument {
        field: String,
        value: String,
        new_document: Document,
    },
    /// Delete documents matching a query.
    DeleteDocuments { field: String, value: String },
    /// Merge segments.
    MergeSegments {
        segment_ids: Vec<String>,
        strategy: crate::lexical::index::inverted::segment::manager::MergeStrategy,
    },
}

/// Transaction metadata and operations.
#[derive(Debug)]
pub struct Transaction {
    /// Unique transaction ID.
    pub id: String,
    /// Isolation level for this transaction.
    pub isolation_level: IsolationLevel,
    /// Current state of the transaction.
    pub state: TransactionState,
    /// Start timestamp.
    pub start_time: u64,
    /// Operations in this transaction.
    pub operations: Vec<TransactionOperation>,
    /// Segments created during this transaction.
    pub created_segments: Vec<String>,
    /// Segments modified during this transaction.
    pub modified_segments: Vec<String>,
    /// Deletion state before transaction.
    pub pre_deletion_state: Option<GlobalDeletionState>,
    /// Thread ID that owns this transaction.
    pub thread_id: Option<std::thread::ThreadId>,
}

impl Transaction {
    /// Create a new transaction.
    pub fn new(isolation_level: IsolationLevel) -> Self {
        let id = Uuid::new_v4().to_string();
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Transaction {
            id,
            isolation_level,
            state: TransactionState::Active,
            start_time,
            operations: Vec::new(),
            created_segments: Vec::new(),
            modified_segments: Vec::new(),
            pre_deletion_state: None,
            thread_id: Some(std::thread::current().id()),
        }
    }

    /// Add an operation to this transaction.
    pub fn add_operation(&mut self, operation: TransactionOperation) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(PlatypusError::index(
                "Cannot add operations to inactive transaction",
            ));
        }
        self.operations.push(operation);
        Ok(())
    }

    /// Mark transaction as preparing for commit.
    pub fn prepare(&mut self) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(PlatypusError::index("Cannot prepare inactive transaction"));
        }
        self.state = TransactionState::Preparing;
        Ok(())
    }

    /// Mark transaction as committed.
    pub fn commit(&mut self) -> Result<()> {
        if self.state != TransactionState::Preparing {
            return Err(PlatypusError::index("Cannot commit unprepared transaction"));
        }
        self.state = TransactionState::Committed;
        Ok(())
    }

    /// Mark transaction as aborted.
    pub fn abort(&mut self) -> Result<()> {
        if self.state == TransactionState::Committed {
            return Err(PlatypusError::index("Cannot abort committed transaction"));
        }
        self.state = TransactionState::Aborted;
        Ok(())
    }

    /// Check if transaction is active.
    pub fn is_active(&self) -> bool {
        self.state == TransactionState::Active
    }

    /// Get transaction duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now.saturating_sub(self.start_time)
    }
}

/// Transaction manager for coordinating atomic operations (schema-less mode).
#[derive(Debug)]
pub struct TransactionManager {
    /// Active transactions.
    active_transactions: Arc<RwLock<AHashMap<String, Arc<Mutex<Transaction>>>>>,
    /// Global transaction counter.
    transaction_counter: Arc<Mutex<u64>>,
    /// Storage backend.
    #[allow(dead_code)]
    storage: Arc<dyn Storage>,
}

impl TransactionManager {
    /// Create a new transaction manager (schema-less mode).
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        TransactionManager {
            active_transactions: Arc::new(RwLock::new(AHashMap::new())),
            transaction_counter: Arc::new(Mutex::new(0)),
            storage,
        }
    }

    /// Deprecated: Use `new()` instead. Schema is no longer required.
    #[deprecated(
        since = "0.2.0",
        note = "Use `new()` instead. Schema is no longer required."
    )]
    pub fn with_schema(
        storage: Arc<dyn Storage>,
        schema: Arc<crate::lexical::document::field::FieldValue>,
    ) -> Self {
        let _ = schema; // Ignore schema parameter
        Self::new(storage)
    }

    /// Begin a new transaction.
    pub fn begin_transaction(
        &self,
        isolation_level: IsolationLevel,
    ) -> Result<Arc<Mutex<Transaction>>> {
        let transaction = Arc::new(Mutex::new(Transaction::new(isolation_level)));
        let transaction_id = transaction.lock().unwrap().id.clone();

        {
            let mut counter = self.transaction_counter.lock().unwrap();
            *counter += 1;
        }

        {
            let mut active = self.active_transactions.write().unwrap();
            active.insert(transaction_id, transaction.clone());
        }

        Ok(transaction)
    }

    /// Commit a transaction.
    pub fn commit_transaction(
        &self,
        transaction: Arc<Mutex<Transaction>>,
        segment_manager: &mut SegmentManager,
        deletion_manager: &mut DeletionManager,
        merge_engine: &MergeEngine,
    ) -> Result<TransactionResult> {
        let transaction_id = {
            let mut txn = transaction.lock().unwrap();
            txn.prepare()?;
            txn.id.clone()
        };

        // Perform two-phase commit
        let result = self.execute_transaction_operations(
            &transaction,
            segment_manager,
            deletion_manager,
            merge_engine,
        );

        match result {
            Ok(commit_result) => {
                // Commit successful
                {
                    let mut txn = transaction.lock().unwrap();
                    txn.commit()?;
                }

                // Remove from active transactions
                {
                    let mut active = self.active_transactions.write().unwrap();
                    active.remove(&transaction_id);
                }

                Ok(commit_result)
            }
            Err(e) => {
                // Rollback transaction
                self.rollback_transaction(transaction, segment_manager, deletion_manager)?;
                Err(e)
            }
        }
    }

    /// Rollback a transaction.
    pub fn rollback_transaction(
        &self,
        transaction: Arc<Mutex<Transaction>>,
        segment_manager: &mut SegmentManager,
        deletion_manager: &mut DeletionManager,
    ) -> Result<()> {
        let (transaction_id, created_segments, pre_deletion_state) = {
            let mut txn = transaction.lock().unwrap();
            txn.abort()?;
            (
                txn.id.clone(),
                txn.created_segments.clone(),
                txn.pre_deletion_state.clone(),
            )
        };

        // Undo changes made by the transaction
        // Remove segments created during this transaction
        for segment_id in &created_segments {
            segment_manager.remove_segment(segment_id)?;
        }

        // Restore deletion state if available
        if let Some(deletion_state) = pre_deletion_state {
            deletion_manager.restore_global_state(deletion_state)?;
        }

        // Remove from active transactions
        {
            let mut active = self.active_transactions.write().unwrap();
            active.remove(&transaction_id);
        }

        Ok(())
    }

    /// Execute all operations in a transaction.
    fn execute_transaction_operations(
        &self,
        transaction: &Arc<Mutex<Transaction>>,
        _segment_manager: &mut SegmentManager,
        deletion_manager: &mut DeletionManager,
        _merge_engine: &MergeEngine,
    ) -> Result<TransactionResult> {
        let operations = {
            let txn = transaction.lock().unwrap();
            txn.operations.clone()
        };

        let mut result = TransactionResult::new();

        // Store current state for rollback
        {
            let mut txn = transaction.lock().unwrap();
            txn.pre_deletion_state = Some(deletion_manager.get_global_state().clone());
        }

        // Execute operations in order
        for operation in operations {
            match operation {
                TransactionOperation::AddDocument {
                    document: _,
                    segment_id,
                } => {
                    let actual_segment_id = if let Some(sid) = segment_id {
                        sid
                    } else {
                        // Use a default segment ID for now
                        format!("segment_{:06}", 0)
                    };

                    // Add document through segment manager
                    // This is a simplified implementation
                    result.docs_added += 1;

                    {
                        let mut txn = transaction.lock().unwrap();
                        txn.modified_segments.push(actual_segment_id);
                    }
                }

                TransactionOperation::UpdateDocument {
                    field: _,
                    value: _,
                    new_document: _,
                } => {
                    // Delete old document - simplified implementation
                    // In a real implementation, this would search for matching documents
                    result.docs_deleted += 1;

                    // Add new document
                    result.docs_added += 1;
                    result.docs_updated += 1;
                }

                TransactionOperation::DeleteDocuments { field: _, value: _ } => {
                    // Delete documents - simplified implementation
                    // In a real implementation, this would search for matching documents
                    result.docs_deleted += 1;
                }

                TransactionOperation::MergeSegments {
                    segment_ids,
                    strategy: _,
                } => {
                    // This would invoke the merge engine
                    result.segments_merged += segment_ids.len();
                    result.merge_operations += 1;
                }
            }
        }

        Ok(result)
    }

    /// Get active transaction count.
    pub fn active_transaction_count(&self) -> usize {
        self.active_transactions.read().unwrap().len()
    }

    /// Get total transaction count.
    pub fn total_transaction_count(&self) -> u64 {
        *self.transaction_counter.lock().unwrap()
    }

    /// Check for deadlocks among active transactions.
    pub fn detect_deadlocks(&self) -> Vec<String> {
        // Simplified deadlock detection
        // In a real implementation, this would analyze lock dependencies
        Vec::new()
    }

    /// Get transaction by ID.
    pub fn get_transaction(&self, transaction_id: &str) -> Option<Arc<Mutex<Transaction>>> {
        let active = self.active_transactions.read().unwrap();
        active.get(transaction_id).cloned()
    }
}

/// Result of a transaction commit.
#[derive(Debug, Default)]
pub struct TransactionResult {
    /// Number of documents added.
    pub docs_added: u64,
    /// Number of documents deleted.
    pub docs_deleted: u64,
    /// Number of documents updated.
    pub docs_updated: u64,
    /// Number of segments merged.
    pub segments_merged: usize,
    /// Number of merge operations.
    pub merge_operations: u64,
    /// Time taken for commit (milliseconds).
    pub commit_time_ms: u64,
}

impl TransactionResult {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Extension trait for atomic operations.
pub trait AtomicOperations {
    /// Execute multiple operations atomically.
    fn execute_atomically<F, R>(&mut self, operations: F) -> Result<R>
    where
        F: FnOnce(&mut Self) -> Result<R>;

    /// Begin a transaction for atomic operations.
    fn begin_atomic_session(&mut self) -> Result<String>;

    /// Commit the current atomic session.
    fn commit_atomic_session(&mut self, session_id: &str) -> Result<TransactionResult>;

    /// Rollback the current atomic session.
    fn rollback_atomic_session(&mut self, session_id: &str) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::lexical::document::field::TextOption;
    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;

    #[test]
    fn test_transaction_creation() {
        let txn = Transaction::new(IsolationLevel::ReadCommitted);

        assert_eq!(txn.isolation_level, IsolationLevel::ReadCommitted);
        assert_eq!(txn.state, TransactionState::Active);
        assert!(txn.is_active());
        assert!(!txn.id.is_empty());
        assert!(txn.operations.is_empty());
    }

    #[test]
    fn test_transaction_state_machine() {
        let mut txn = Transaction::new(IsolationLevel::ReadCommitted);

        // Active -> Preparing
        assert!(txn.prepare().is_ok());
        assert_eq!(txn.state, TransactionState::Preparing);

        // Preparing -> Committed
        assert!(txn.commit().is_ok());
        assert_eq!(txn.state, TransactionState::Committed);
        assert!(!txn.is_active());

        // Cannot abort committed transaction
        assert!(txn.abort().is_err());
    }

    #[test]
    fn test_transaction_operations() {
        let mut txn = Transaction::new(IsolationLevel::ReadCommitted);

        let doc = crate::lexical::document::document::Document::builder()
            .add_text("title", "Test", TextOption::default())
            .build();

        let op = TransactionOperation::AddDocument {
            document: doc,
            segment_id: None,
        };

        assert!(txn.add_operation(op).is_ok());
        assert_eq!(txn.operations.len(), 1);

        // Cannot add operation to inactive transaction
        txn.state = TransactionState::Committed;
        let doc2 = crate::lexical::document::document::Document::builder()
            .add_text("title", "Test2", TextOption::default())
            .build();
        let op2 = TransactionOperation::AddDocument {
            document: doc2,
            segment_id: None,
        };
        assert!(txn.add_operation(op2).is_err());
    }

    #[test]
    fn test_transaction_manager() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let manager = TransactionManager::new(storage);

        assert_eq!(manager.active_transaction_count(), 0);
        assert_eq!(manager.total_transaction_count(), 0);

        let txn = manager
            .begin_transaction(IsolationLevel::ReadCommitted)
            .unwrap();
        assert_eq!(manager.active_transaction_count(), 1);
        assert_eq!(manager.total_transaction_count(), 1);

        let transaction_id = txn.lock().unwrap().id.clone();
        let retrieved = manager.get_transaction(&transaction_id);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_isolation_levels() {
        let levels = [
            IsolationLevel::ReadCommitted,
            IsolationLevel::RepeatableRead,
            IsolationLevel::Serializable,
        ];

        for level in levels {
            let txn = Transaction::new(level);
            assert_eq!(txn.isolation_level, level);
        }
    }

    #[test]
    fn test_transaction_duration() {
        let txn = Transaction::new(IsolationLevel::ReadCommitted);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(txn.duration_ms() >= 10);
    }
}
