//! Factory for creating vector collections.
//!
//! This module provides a factory pattern for creating `VectorCollection`
//! implementations, similar to `LexicalIndexFactory` for lexical search.

use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;
use crate::vector::collection::VectorCollection;
use crate::vector::collection::multifield::MultiFieldVectorCollection;
use crate::vector::engine::config::VectorEngineConfig;

/// Factory for creating vector collections.
///
/// This factory provides a centralized way to create different types of
/// vector collections based on configuration. Currently, only
/// `MultiFieldVectorCollection` is supported.
///
/// # Example
///
/// ```ignore
/// use platypus::vector::collection::factory::VectorCollectionFactory;
/// use platypus::vector::engine::config::VectorEngineConfig;
///
/// let config = VectorEngineConfig::default();
/// let collection = VectorCollectionFactory::create(config, storage, None)?;
/// ```
pub struct VectorCollectionFactory;

impl VectorCollectionFactory {
    /// Create a new vector collection from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The vector engine configuration
    /// * `storage` - The storage backend for persistence
    /// * `initial_doc_id` - Optional initial document ID (defaults to 1)
    ///
    /// # Returns
    ///
    /// A boxed `VectorCollection` trait object.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection cannot be created.
    pub fn create(
        config: VectorEngineConfig,
        storage: Arc<dyn Storage>,
        initial_doc_id: Option<u64>,
    ) -> Result<Box<dyn VectorCollection>> {
        let collection = MultiFieldVectorCollection::new(config, storage, initial_doc_id)?;
        Ok(Box::new(collection))
    }
}
