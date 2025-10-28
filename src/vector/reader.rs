//! Vector index reader traits and implementations.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::vector::types::{ValidationReport, VectorIndexMetadata, VectorStats};
use crate::vector::{DistanceMetric, Vector};

/// Trait for reading vector indexes (similar to IndexReader for inverted indexes).
pub trait VectorIndexReader: Send + Sync {
    /// Get a vector by document ID.
    fn get_vector(&self, doc_id: u64) -> Result<Option<Vector>>;

    /// Get multiple vectors by document IDs.
    fn get_vectors(&self, doc_ids: &[u64]) -> Result<Vec<Option<Vector>>>;

    /// Get all vector IDs in the index.
    fn vector_ids(&self) -> Result<Vec<u64>>;

    /// Get the total number of vectors.
    fn vector_count(&self) -> usize;

    /// Get the vector dimension.
    fn dimension(&self) -> usize;

    /// Get the distance metric used.
    fn distance_metric(&self) -> DistanceMetric;

    /// Get index statistics.
    fn stats(&self) -> VectorStats;

    /// Check if a vector exists.
    fn contains_vector(&self, doc_id: u64) -> bool;

    /// Get vectors in a specific range.
    fn get_vector_range(&self, start_doc_id: u64, end_doc_id: u64) -> Result<Vec<(u64, Vector)>>;

    /// Get an iterator over all vectors.
    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>>;

    /// Get index metadata.
    fn metadata(&self) -> Result<VectorIndexMetadata>;

    /// Validate index integrity.
    fn validate(&self) -> Result<ValidationReport>;
}

/// Iterator over vectors in an index.
pub trait VectorIterator: Send {
    /// Get the next vector.
    fn next(&mut self) -> Result<Option<(u64, Vector)>>;

    /// Skip to a specific document ID.
    fn skip_to(&mut self, doc_id: u64) -> Result<bool>;

    /// Get the current position.
    fn position(&self) -> u64;

    /// Reset to the beginning.
    fn reset(&mut self) -> Result<()>;
}

/// Simple in-memory vector reader for basic use cases.
/// This is a lightweight reader that holds vectors in memory.
pub struct SimpleVectorReader {
    vectors: HashMap<u64, Vector>,
    vector_ids: Vec<u64>,
    dimension: usize,
    distance_metric: DistanceMetric,
}

impl SimpleVectorReader {
    /// Create a new simple vector reader.
    pub fn new(
        vectors: Vec<(u64, Vector)>,
        dimension: usize,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        let vector_ids: Vec<u64> = vectors.iter().map(|(id, _)| *id).collect();
        let vectors: HashMap<u64, Vector> = vectors.into_iter().collect();

        Ok(Self {
            vectors,
            vector_ids,
            dimension,
            distance_metric,
        })
    }
}

impl VectorIndexReader for SimpleVectorReader {
    fn get_vector(&self, doc_id: u64) -> Result<Option<Vector>> {
        Ok(self.vectors.get(&doc_id).cloned())
    }

    fn get_vectors(&self, doc_ids: &[u64]) -> Result<Vec<Option<Vector>>> {
        Ok(doc_ids
            .iter()
            .map(|id| self.vectors.get(id).cloned())
            .collect())
    }

    fn vector_ids(&self) -> Result<Vec<u64>> {
        Ok(self.vector_ids.clone())
    }

    fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    fn stats(&self) -> VectorStats {
        let memory_usage = self.vectors.len() * (8 + self.dimension * 4);
        VectorStats {
            vector_count: self.vectors.len(),
            dimension: self.dimension,
            memory_usage,
            build_time_ms: 0,
        }
    }

    fn contains_vector(&self, doc_id: u64) -> bool {
        self.vectors.contains_key(&doc_id)
    }

    fn get_vector_range(&self, start_doc_id: u64, end_doc_id: u64) -> Result<Vec<(u64, Vector)>> {
        Ok(self
            .vectors
            .iter()
            .filter(|(id, _)| **id >= start_doc_id && **id <= end_doc_id)
            .map(|(id, v)| (*id, v.clone()))
            .collect())
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        Ok(Box::new(SimpleVectorIterator::new(
            self.vectors
                .iter()
                .map(|(id, v)| (*id, v.clone()))
                .collect(),
        )))
    }

    fn metadata(&self) -> Result<VectorIndexMetadata> {
        Ok(VectorIndexMetadata {
            index_type: "Simple".to_string(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            version: "1.0".to_string(),
            build_config: serde_json::json!({}),
            custom_metadata: HashMap::new(),
        })
    }

    fn validate(&self) -> Result<ValidationReport> {
        let all_valid = self.vectors.values().all(|v| v.is_valid());
        Ok(ValidationReport {
            is_valid: all_valid,
            errors: vec![],
            warnings: vec![],
            repair_suggestions: vec![],
        })
    }
}

/// Simple vector iterator.
pub struct SimpleVectorIterator {
    vectors: Vec<(u64, Vector)>,
    position: usize,
}

impl SimpleVectorIterator {
    /// Create a new simple vector iterator.
    pub fn new(vectors: Vec<(u64, Vector)>) -> Self {
        Self {
            vectors,
            position: 0,
        }
    }
}

impl VectorIterator for SimpleVectorIterator {
    fn next(&mut self) -> Result<Option<(u64, Vector)>> {
        if self.position < self.vectors.len() {
            let result = self.vectors[self.position].clone();
            self.position += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    fn skip_to(&mut self, doc_id: u64) -> Result<bool> {
        while self.position < self.vectors.len() {
            if self.vectors[self.position].0 >= doc_id {
                return Ok(true);
            }
            self.position += 1;
        }
        Ok(false)
    }

    fn position(&self) -> u64 {
        self.position as u64
    }

    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        Ok(())
    }
}

/// Factory for creating vector index readers.
///
/// This factory provides a centralized way to create readers for different
/// vector index types (Flat, HNSW, IVF) from serialized index data.
pub struct VectorIndexReaderFactory;

impl VectorIndexReaderFactory {
    /// Create a reader for a specific index type from serialized data.
    ///
    /// # Arguments
    ///
    /// * `index_type` - The type of index ("flat", "hnsw", or "ivf")
    /// * `index_data` - Serialized index data
    ///
    /// # Returns
    ///
    /// An `Arc<dyn VectorIndexReader>` that can be used to query the index.
    pub fn create_reader(
        index_type: &str,
        index_data: &[u8],
    ) -> Result<Arc<dyn VectorIndexReader>> {
        use crate::vector::index::reader::flat::FlatVectorIndexReader;
        use crate::vector::index::reader::hnsw::HnswIndexReader;
        use crate::vector::index::reader::ivf::IvfIndexReader;

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
