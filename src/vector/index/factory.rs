//! Factory for creating vector index instances.

use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;
use crate::vector::index::VectorIndex;
use crate::vector::index::config::VectorIndexConfig;
use crate::vector::index::flat::FlatIndex;
use crate::vector::index::hnsw::HnswIndex;
use crate::vector::index::ivf::IvfIndex;

/// Factory for creating vector index instances.
///
/// This factory follows the Factory design pattern to create appropriate
/// index implementations based on the provided configuration.
///
/// # Design Benefits
///
/// - **Decoupling**: Client code doesn't need to know about concrete index types
/// - **Extensibility**: New index types can be added by extending the enum
/// - **Type safety**: Pattern matching ensures all cases are handled
///
/// # Example with StorageFactory
///
/// ```
/// use yatagarasu::vector::index::factory::VectorIndexFactory;
/// use yatagarasu::vector::index::config::VectorIndexConfig;
/// use yatagarasu::storage::{StorageFactory, StorageConfig};
/// use yatagarasu::storage::memory::MemoryStorageConfig;
///
/// # fn main() -> yatagarasu::error::Result<()> {
/// // Create storage using factory
/// let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
///
/// // Create index using factory
/// let config = VectorIndexConfig::default();
/// let index = VectorIndexFactory::create(storage, config)?;
/// # Ok(())
/// # }
/// ```
pub struct VectorIndexFactory;

impl VectorIndexFactory {
    /// Create a new vector index with the given storage and configuration.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend (created using `StorageFactory`)
    /// * `config` - Index configuration enum containing type-specific settings
    ///
    /// # Returns
    ///
    /// A boxed trait object implementing `VectorIndex` trait.
    /// The concrete type is determined by the config variant.
    ///
    /// # Example
    ///
    /// ```
    /// use yatagarasu::vector::index::factory::VectorIndexFactory;
    /// use yatagarasu::vector::index::config::{VectorIndexConfig, FlatIndexConfig};
    /// use yatagarasu::storage::{StorageFactory, StorageConfig};
    /// use yatagarasu::storage::file::FileStorageConfig;
    ///
    /// # fn main() -> yatagarasu::error::Result<()> {
    /// // Create file storage
    /// let storage_config = StorageConfig::File(FileStorageConfig::new("/tmp/index"));
    /// let storage = StorageFactory::create(storage_config)?;
    ///
    /// // Create flat index
    /// let index_config = VectorIndexConfig::Flat(FlatIndexConfig::default());
    /// let index = VectorIndexFactory::create(storage, index_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create(
        storage: Arc<dyn Storage>,
        config: VectorIndexConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        match config {
            VectorIndexConfig::Flat(flat_config) => {
                let index = FlatIndex::create(storage, flat_config)?;
                Ok(Box::new(index))
            }
            VectorIndexConfig::HNSW(hnsw_config) => {
                let index = HnswIndex::create(storage, hnsw_config)?;
                Ok(Box::new(index))
            }
            VectorIndexConfig::IVF(ivf_config) => {
                let index = IvfIndex::create(storage, ivf_config)?;
                Ok(Box::new(index))
            }
        }
    }

    /// Open an existing vector index with the given storage and configuration.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend containing the existing index
    /// * `config` - Index configuration (must match the stored index type)
    ///
    /// # Returns
    ///
    /// A boxed index implementation based on the configured index type.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use yatagarasu::vector::index::factory::VectorIndexFactory;
    /// use yatagarasu::vector::index::config::{VectorIndexConfig, FlatIndexConfig};
    /// use yatagarasu::storage::file::{FileStorage, FileStorageConfig};
    /// use std::sync::Arc;
    ///
    /// # fn main() -> yatagarasu::error::Result<()> {
    /// let storage = Arc::new(FileStorage::new("./index", FileStorageConfig::new("./index"))?);
    /// let config = VectorIndexConfig::Flat(FlatIndexConfig::default());
    /// let index = VectorIndexFactory::open(storage, config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(
        storage: Arc<dyn Storage>,
        config: VectorIndexConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        match config {
            VectorIndexConfig::Flat(flat_config) => {
                let index = FlatIndex::open(storage, flat_config)?;
                Ok(Box::new(index))
            }
            VectorIndexConfig::HNSW(hnsw_config) => {
                let index = HnswIndex::open(storage, hnsw_config)?;
                Ok(Box::new(index))
            }
            VectorIndexConfig::IVF(ivf_config) => {
                let index = IvfIndex::open(storage, ivf_config)?;
                Ok(Box::new(index))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;
    use crate::vector::index::config::{
        FlatIndexConfig, HnswIndexConfig, IvfIndexConfig, VectorIndexConfig,
    };

    #[test]
    fn test_vector_index_creation() {
        let config = VectorIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = VectorIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_vector_index_open() {
        let config = VectorIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Create index
        let mut index = VectorIndexFactory::create(storage.clone(), config.clone()).unwrap();
        index.close().unwrap();

        // Open index
        let index = VectorIndexFactory::open(storage, config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_vector_index_stats() {
        let config = VectorIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = VectorIndexFactory::create(storage, config).unwrap();
        let stats = index.stats().unwrap();

        assert_eq!(stats.vector_count, 0);
        assert_eq!(stats.dimension, 128); // Default dimension
        assert_eq!(stats.deleted_count, 0);
        assert!(stats.last_modified > 0);
    }

    #[test]
    fn test_vector_index_close() {
        let config = VectorIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut index = VectorIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());

        index.close().unwrap();

        assert!(index.is_closed());

        // Operations should fail after close
        let result = index.stats();
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_index_config() {
        let config = VectorIndexConfig::default();

        // Test that default is Flat and check its configuration
        match config {
            VectorIndexConfig::Flat(flat) => {
                assert_eq!(flat.dimension, 128);
            }
            _ => panic!("Expected Flat config as default"),
        }
    }

    #[test]
    fn test_factory_create_flat() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = VectorIndexConfig::Flat(FlatIndexConfig::default());

        let index = VectorIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_factory_create_hnsw() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = VectorIndexConfig::HNSW(HnswIndexConfig::default());

        let index = VectorIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_factory_create_ivf() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = VectorIndexConfig::IVF(IvfIndexConfig::default());

        let index = VectorIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());
    }
}
