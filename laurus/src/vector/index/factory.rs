//! Factory for creating vector index instances.

use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;
use crate::vector::index::VectorIndex;
use crate::vector::index::config::VectorIndexTypeConfig;
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
/// use laurus::vector::index::factory::VectorIndexFactory;
/// use laurus::vector::index::config::VectorIndexTypeConfig;
/// use laurus::storage::{StorageFactory, StorageConfig};
/// use laurus::storage::memory::MemoryStorageConfig;
///
/// # fn main() -> laurus::Result<()> {
/// // Create storage using factory
/// let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
///
/// // Create index using factory
/// let config = VectorIndexTypeConfig::default();
/// let index = VectorIndexFactory::create(storage, "test_index", config)?;
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
    /// use laurus::vector::index::factory::VectorIndexFactory;
    /// use laurus::vector::index::config::{VectorIndexTypeConfig, FlatIndexConfig};
    /// use laurus::storage::{StorageFactory, StorageConfig};
    /// use laurus::storage::file::FileStorageConfig;
    ///
    /// # fn main() -> laurus::Result<()> {
    /// // Create file storage
    /// let storage_config = StorageConfig::File(FileStorageConfig::new("/tmp/index"));
    /// let storage = StorageFactory::create(storage_config)?;
    ///
    /// // Create flat index
    /// let index_config = VectorIndexTypeConfig::Flat(FlatIndexConfig::default());
    /// let index = VectorIndexFactory::create(storage, "test_index", index_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create(
        storage: Arc<dyn Storage>,
        name: &str,
        config: VectorIndexTypeConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        match config {
            VectorIndexTypeConfig::Flat(flat_config) => {
                let index = FlatIndex::create(storage, name, flat_config)?;
                Ok(Box::new(index))
            }
            VectorIndexTypeConfig::HNSW(hnsw_config) => {
                Self::dispatch_hnsw(storage, name, hnsw_config, false)
            }
            VectorIndexTypeConfig::IVF(ivf_config) => {
                let index = IvfIndex::create(storage, name, ivf_config)?;
                Ok(Box::new(index))
            }
        }
    }

    /// Dispatch an HNSW config to the segmented or monolithic implementation
    /// (#634 / #882).
    ///
    /// The `segmented` flag (the default since #882) routes to
    /// [`SegmentedHnswIndex::open_or_create`], which handles create, open,
    /// AND the zero-copy migration of a legacy monolithic index — so BOTH
    /// factory arms must come through here: a legacy index always has a
    /// `metadata.json` and therefore always takes the *open* arm, which is
    /// exactly where the migration must fire (#882 review).
    ///
    /// With the flag off, opening a directory that contains a segment
    /// manifest is rejected: the monolithic index would silently serve only
    /// the migrated segment-0 data and hide every later segment.
    fn dispatch_hnsw(
        storage: Arc<dyn Storage>,
        name: &str,
        hnsw_config: crate::vector::index::config::HnswIndexConfig,
        open_only: bool,
    ) -> Result<Box<dyn VectorIndex>> {
        use crate::vector::index::hnsw::segmented::SegmentedHnswIndex;

        if hnsw_config.segmented {
            let index = SegmentedHnswIndex::open_or_create(storage, name, hnsw_config)?;
            return Ok(Box::new(index));
        }
        // Reverse guard (#882 review): a segmented directory must not be
        // opened monolithically — the single-file view would silently hide
        // every segment sealed after the migration.
        if storage.file_exists("segments.json") {
            return Err(crate::error::LaurusError::invalid_config(format!(
                "index '{name}' uses the segmented layout (segments.json exists) but \
                 `segmented` is disabled; enable it to open this index"
            )));
        }
        let index = if open_only {
            HnswIndex::open(storage, name, hnsw_config)?
        } else {
            HnswIndex::create(storage, name, hnsw_config)?
        };
        Ok(Box::new(index))
    }

    /// Open an existing index or create a new one if it doesn't exist.
    ///
    /// This is the recommended method for general use, as it handles both
    /// creation and opening transparently.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend
    /// * `name` - Index name
    /// * `config` - Index configuration
    ///
    /// # Returns
    ///
    /// A boxed index implementation.
    pub fn open_or_create(
        storage: Arc<dyn Storage>,
        name: &str,
        config: VectorIndexTypeConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        if storage.file_exists("metadata.json") {
            Self::open(storage, name, config)
        } else {
            Self::create(storage, name, config)
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
    /// use laurus::vector::index::factory::VectorIndexFactory;
    /// use laurus::vector::index::config::{VectorIndexTypeConfig, FlatIndexConfig};
    /// use laurus::storage::file::{FileStorage, FileStorageConfig};
    /// use std::sync::Arc;
    ///
    /// # fn main() -> laurus::Result<()> {
    /// let storage = Arc::new(FileStorage::new("./index", FileStorageConfig::new("./index"))?);
    /// let config = VectorIndexTypeConfig::Flat(FlatIndexConfig::default());
    /// let index = VectorIndexFactory::open(storage, "test_index", config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(
        storage: Arc<dyn Storage>,
        name: &str,
        config: VectorIndexTypeConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        match config {
            VectorIndexTypeConfig::Flat(flat_config) => {
                let index = FlatIndex::open(storage, name, flat_config)?;
                Ok(Box::new(index))
            }
            VectorIndexTypeConfig::HNSW(hnsw_config) => {
                // Same dispatch as `create` (#882): the segmented path's
                // `open_or_create` opens existing manifests and migrates
                // legacy monolithic indexes; the flag-off path is
                // reverse-guarded against segmented directories.
                Self::dispatch_hnsw(storage, name, hnsw_config, true)
            }
            VectorIndexTypeConfig::IVF(ivf_config) => {
                let index = IvfIndex::open(storage, name, ivf_config)?;
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
        FlatIndexConfig, HnswIndexConfig, IvfIndexConfig, VectorIndexTypeConfig,
    };

    #[test]
    fn test_vector_index_creation() {
        let config = VectorIndexTypeConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = VectorIndexFactory::create(storage, "test_index", config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_vector_index_open() {
        let config = VectorIndexTypeConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Create index
        let index =
            VectorIndexFactory::create(storage.clone(), "test_index", config.clone()).unwrap();
        index.close().unwrap();

        // Open index
        let index = VectorIndexFactory::open(storage, "test_index", config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_vector_index_stats() {
        let config = VectorIndexTypeConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = VectorIndexFactory::create(storage, "test_index", config).unwrap();
        let stats = index.stats().unwrap();

        assert_eq!(stats.vector_count, 0);
        assert_eq!(stats.dimension, 128); // Default dimension
        assert_eq!(stats.deleted_count, 0);
        assert!(stats.last_modified > 0);
    }

    #[test]
    fn test_vector_index_close() {
        let config = VectorIndexTypeConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = VectorIndexFactory::create(storage, "test_index", config).unwrap();

        assert!(!index.is_closed());

        index.close().unwrap();

        assert!(index.is_closed());

        // Operations should fail after close
        let result = index.stats();
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_index_config() {
        let config = VectorIndexTypeConfig::default();

        // Test that default is Flat and check its configuration
        match config {
            VectorIndexTypeConfig::Flat(flat) => {
                assert_eq!(flat.dimension, 128);
            }
            _ => panic!("Expected Flat config as default"),
        }
    }

    #[test]
    fn test_factory_create_flat() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = VectorIndexTypeConfig::Flat(FlatIndexConfig::default());

        let index = VectorIndexFactory::create(storage, "flat_index", config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_factory_create_hnsw() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = VectorIndexTypeConfig::HNSW(HnswIndexConfig::default());

        let index = VectorIndexFactory::create(storage, "hnsw_index", config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_factory_create_ivf() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = VectorIndexTypeConfig::IVF(IvfIndexConfig::default());

        let index = VectorIndexFactory::create(storage, "ivf_index", config).unwrap();

        assert!(!index.is_closed());
    }
}
