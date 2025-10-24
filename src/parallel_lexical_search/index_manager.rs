//! Index management for parallel search operations.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::error::{Result, SageError};
use crate::lexical::reader::IndexReader;

/// Metadata associated with an index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Human-readable name for the index.
    pub name: String,

    /// Description of the index contents.
    pub description: Option<String>,

    /// Creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last update timestamp.
    pub updated_at: chrono::DateTime<chrono::Utc>,

    /// Number of documents in the index.
    pub doc_count: Option<u64>,

    /// Size of the index in bytes.
    pub size_bytes: Option<u64>,

    /// Custom metadata fields.
    pub custom_fields: HashMap<String, String>,
}

impl Default for IndexMetadata {
    fn default() -> Self {
        let now = chrono::Utc::now();
        Self {
            name: String::new(),
            description: None,
            created_at: now,
            updated_at: now,
            doc_count: None,
            size_bytes: None,
            custom_fields: HashMap::new(),
        }
    }
}

/// Handle to an individual index with its reader and metadata.
#[derive(Debug, Clone)]
pub struct IndexHandle {
    /// Unique identifier for this index.
    pub id: String,

    /// The index reader instance.
    pub reader: Arc<dyn IndexReader>,

    /// Weight factor for this index (used in weighted merge).
    pub weight: f32,

    /// Metadata about this index.
    pub metadata: IndexMetadata,

    /// Whether this index is currently active.
    pub is_active: bool,
}

impl IndexHandle {
    /// Create a new index handle.
    pub fn new(id: String, reader: Arc<dyn IndexReader>) -> Self {
        Self {
            id: id.clone(),
            reader,
            weight: 1.0,
            metadata: IndexMetadata {
                name: id,
                ..Default::default()
            },
            is_active: true,
        }
    }

    /// Create a new index handle with metadata.
    pub fn with_metadata(
        id: String,
        reader: Arc<dyn IndexReader>,
        metadata: IndexMetadata,
    ) -> Self {
        Self {
            id,
            reader,
            weight: 1.0,
            metadata,
            is_active: true,
        }
    }

    /// Set the weight for this index.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Update the document count in metadata.
    pub fn update_doc_count(&mut self, count: u64) {
        self.metadata.doc_count = Some(count);
        self.metadata.updated_at = chrono::Utc::now();
    }
}

/// Manager for multiple indices used in parallel search.
pub struct IndexManager {
    /// Map of index ID to index handle.
    indices: Arc<RwLock<HashMap<String, IndexHandle>>>,

    /// Total weight of all active indices (cached for efficiency).
    total_weight: Arc<RwLock<f32>>,
}

impl IndexManager {
    /// Create a new empty index manager.
    pub fn new() -> Self {
        Self {
            indices: Arc::new(RwLock::new(HashMap::new())),
            total_weight: Arc::new(RwLock::new(0.0)),
        }
    }

    /// Add a new index to the manager.
    pub fn add_index(&self, handle: IndexHandle) -> Result<()> {
        let id = handle.id.clone();
        let weight = handle.weight;

        let mut indices = self
            .indices
            .write()
            .map_err(|_| SageError::internal("Failed to acquire write lock on indices"))?;

        if indices.contains_key(&id) {
            return Err(SageError::invalid_argument(format!(
                "Index with ID '{id}' already exists"
            )));
        }

        indices.insert(id, handle);

        // Update total weight
        let mut total = self
            .total_weight
            .write()
            .map_err(|_| SageError::internal("Failed to acquire write lock on total weight"))?;
        *total += weight;

        Ok(())
    }

    /// Remove an index from the manager.
    pub fn remove_index(&self, id: &str) -> Result<IndexHandle> {
        let mut indices = self
            .indices
            .write()
            .map_err(|_| SageError::internal("Failed to acquire write lock on indices"))?;

        let handle = indices
            .remove(id)
            .ok_or_else(|| SageError::not_found(format!("Index with ID '{id}' not found")))?;

        // Update total weight
        let mut total = self
            .total_weight
            .write()
            .map_err(|_| SageError::internal("Failed to acquire write lock on total weight"))?;
        *total -= handle.weight;

        Ok(handle)
    }

    /// Get an index by ID.
    pub fn get_index(&self, id: &str) -> Result<IndexHandle> {
        let indices = self
            .indices
            .read()
            .map_err(|_| SageError::internal("Failed to acquire read lock on indices"))?;

        indices
            .get(id)
            .cloned()
            .ok_or_else(|| SageError::not_found(format!("Index with ID '{id}' not found")))
    }

    /// Get all active indices.
    pub fn get_active_indices(&self) -> Result<Vec<IndexHandle>> {
        let indices = self
            .indices
            .read()
            .map_err(|_| SageError::internal("Failed to acquire read lock on indices"))?;

        Ok(indices
            .values()
            .filter(|handle| handle.is_active)
            .cloned()
            .collect())
    }

    /// Get all indices (including inactive).
    pub fn get_all_indices(&self) -> Result<Vec<IndexHandle>> {
        let indices = self
            .indices
            .read()
            .map_err(|_| SageError::internal("Failed to acquire read lock on indices"))?;

        Ok(indices.values().cloned().collect())
    }

    /// Update the weight of an index.
    pub fn update_weight(&self, id: &str, weight: f32) -> Result<()> {
        let mut indices = self
            .indices
            .write()
            .map_err(|_| SageError::internal("Failed to acquire write lock on indices"))?;

        let handle = indices
            .get_mut(id)
            .ok_or_else(|| SageError::not_found(format!("Index with ID '{id}' not found")))?;

        let old_weight = handle.weight;
        handle.weight = weight;

        // Update total weight
        let mut total = self
            .total_weight
            .write()
            .map_err(|_| SageError::internal("Failed to acquire write lock on total weight"))?;
        *total = *total - old_weight + weight;

        Ok(())
    }

    /// Set the active status of an index.
    pub fn set_active(&self, id: &str, active: bool) -> Result<()> {
        let mut indices = self
            .indices
            .write()
            .map_err(|_| SageError::internal("Failed to acquire write lock on indices"))?;

        let handle = indices
            .get_mut(id)
            .ok_or_else(|| SageError::not_found(format!("Index with ID '{id}' not found")))?;

        handle.is_active = active;
        Ok(())
    }

    /// Get the total weight of all active indices.
    pub fn get_total_weight(&self) -> Result<f32> {
        let total = self
            .total_weight
            .read()
            .map_err(|_| SageError::internal("Failed to acquire read lock on total weight"))?;

        Ok(*total)
    }

    /// Get the number of indices.
    pub fn len(&self) -> Result<usize> {
        let indices = self
            .indices
            .read()
            .map_err(|_| SageError::internal("Failed to acquire read lock on indices"))?;

        Ok(indices.len())
    }

    /// Check if the manager is empty.
    pub fn is_empty(&self) -> Result<bool> {
        self.len().map(|len| len == 0)
    }
}

impl Default for IndexManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::reader::inverted_index::{
        InvertedIndexReader, InvertedIndexReaderConfig,
    };
    use crate::storage::memory::MemoryStorage;
    use crate::storage::traits::StorageConfig;

    fn create_test_reader() -> Arc<dyn IndexReader> {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        Arc::new(
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap(),
        )
    }

    #[test]
    fn test_index_handle_creation() {
        let reader = create_test_reader();
        let handle = IndexHandle::new("test_index".to_string(), reader).with_weight(2.0);

        assert_eq!(handle.id, "test_index");
        assert_eq!(handle.weight, 2.0);
        assert!(handle.is_active);
    }

    #[test]
    fn test_index_manager_operations() {
        let manager = IndexManager::new();

        // Add indices
        let reader1 = create_test_reader();
        let handle1 = IndexHandle::new("index1".to_string(), reader1).with_weight(1.5);
        manager.add_index(handle1).unwrap();

        let reader2 = create_test_reader();
        let handle2 = IndexHandle::new("index2".to_string(), reader2).with_weight(2.5);
        manager.add_index(handle2).unwrap();

        // Test retrieval
        assert_eq!(manager.len().unwrap(), 2);
        assert_eq!(manager.get_total_weight().unwrap(), 4.0);

        let retrieved = manager.get_index("index1").unwrap();
        assert_eq!(retrieved.id, "index1");
        assert_eq!(retrieved.weight, 1.5);

        // Test weight update
        manager.update_weight("index1", 3.0).unwrap();
        assert_eq!(manager.get_total_weight().unwrap(), 5.5);

        // Test active status
        manager.set_active("index2", false).unwrap();
        let active_indices = manager.get_active_indices().unwrap();
        assert_eq!(active_indices.len(), 1);
        assert_eq!(active_indices[0].id, "index1");

        // Test removal
        let removed = manager.remove_index("index1").unwrap();
        assert_eq!(removed.id, "index1");
        assert_eq!(manager.len().unwrap(), 1);
        assert_eq!(manager.get_total_weight().unwrap(), 2.5);
    }

    #[test]
    fn test_duplicate_index_error() {
        let manager = IndexManager::new();
        let reader = create_test_reader();

        let handle1 = IndexHandle::new("duplicate".to_string(), reader.clone());
        manager.add_index(handle1).unwrap();

        let handle2 = IndexHandle::new("duplicate".to_string(), reader);
        let result = manager.add_index(handle2);
        assert!(result.is_err());
    }
}
