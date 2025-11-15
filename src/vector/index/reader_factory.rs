//! Factory for creating vector index readers.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::index::flat::reader::FlatVectorIndexReader;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::ivf::reader::IvfIndexReader;
use crate::vector::index::reader::VectorIndexReader;

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
            _ => Err(crate::error::YatagarasuError::InvalidOperation(format!(
                "Unknown index type: {index_type}"
            ))),
        }
    }
}
