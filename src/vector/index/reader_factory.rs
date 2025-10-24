//! Factory for creating vector index readers.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::index::reader::flat::FlatVectorIndexReader;
use crate::vector::index::reader::hnsw::HnswIndexReader;
use crate::vector::index::reader::ivf::IvfIndexReader;
use crate::vector::reader::VectorIndexReader;

/// Factory for creating vector index readers.
pub struct VectorIndexReaderFactory;

impl VectorIndexReaderFactory {
    /// Create a reader for a specific index type.
    pub fn create_reader(
        index_type: &str,
        index_data: &[u8],
    ) -> Result<Arc<dyn VectorIndexReader>> {
        match index_type.to_lowercase().as_str() {
            "flat" => {
                let reader = FlatVectorIndexReader::from_bytes(index_data)?;
                Ok(Arc::new(reader))
            }
            "hnsw" => {
                let reader = HnswIndexReader::from_bytes(index_data)?;
                Ok(Arc::new(reader))
            }
            "ivf" => {
                let reader = IvfIndexReader::from_bytes(index_data)?;
                Ok(Arc::new(reader))
            }
            _ => Err(crate::error::SageError::InvalidOperation(format!(
                "Unknown index type: {index_type}"
            ))),
        }
    }
}
