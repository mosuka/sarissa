//! Vector index reader traits and implementations.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;

/// Statistics about a vector index.
#[derive(Debug, Clone)]
pub struct VectorStats {
    /// Total number of vectors in the index.
    pub vector_count: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Index memory usage in bytes.
    pub memory_usage: usize,
    /// Build time in milliseconds.
    pub build_time_ms: u64,
}

/// Metadata about a vector index.
#[derive(Debug, Clone)]
pub struct VectorIndexMetadata {
    /// Index type (HNSW, Flat, IVF, etc.).
    pub index_type: String,
    /// Creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp.
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Index version.
    pub version: String,
    /// Build configuration.
    pub build_config: serde_json::Value,
    /// Custom metadata.
    pub custom_metadata: std::collections::HashMap<String, String>,
}

/// Index validation report.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Whether the index is valid.
    pub is_valid: bool,
    /// Validation errors found.
    pub errors: Vec<String>,
    /// Validation warnings.
    pub warnings: Vec<String>,
    /// Repair suggestions.
    pub repair_suggestions: Vec<String>,
}

/// Trait for reading vector indexes (similar to IndexReader for inverted indexes).
pub trait VectorIndexReader: Send + Sync + std::fmt::Debug {
    /// Cast to Any for downcasting to concrete types.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get a vector by document ID and field name.
    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>>;

    /// Get all vectors for a document ID (all fields).
    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>>;

    /// Get multiple vectors by document IDs and field names.
    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>>;

    /// Get all vector IDs with their field names in the index.
    fn vector_ids(&self) -> Result<Vec<(u64, String)>>;

    /// Get the total number of vectors.
    fn vector_count(&self) -> usize;

    /// Get the vector dimension.
    fn dimension(&self) -> usize;

    /// Get the distance metric used.
    fn distance_metric(&self) -> DistanceMetric;

    /// Get index statistics.
    fn stats(&self) -> VectorStats;

    /// Check if a vector exists for a specific field.
    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool;

    /// Get vectors in a specific range with field names.
    fn get_vector_range(
        &self,
        start_doc_id: u64,
        end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>>;

    /// Get all vectors with a specific field name.
    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>>;

    /// Get all unique field names in the index.
    fn field_names(&self) -> Result<Vec<String>>;

    /// Get an iterator over all vectors.
    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>>;

    /// Get index metadata.
    fn metadata(&self) -> Result<VectorIndexMetadata>;

    /// Validate index integrity.
    fn validate(&self) -> Result<ValidationReport>;
}

/// Iterator over vectors in an index.
pub trait VectorIterator: Send {
    /// Get the next vector with field name.
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>>;

    /// Skip to a specific document ID and field.
    fn skip_to(&mut self, doc_id: u64, field_name: &str) -> Result<bool>;

    /// Get the current position.
    fn position(&self) -> (u64, String);

    /// Reset to the beginning.
    fn reset(&mut self) -> Result<()>;
}

/// Simple in-memory vector reader for basic use cases.
/// This is a lightweight reader that holds vectors in memory.
#[derive(Debug)]
pub struct SimpleVectorReader {
    vectors: HashMap<(u64, String), Vector>,
    vector_ids: Vec<(u64, String)>,
    dimension: usize,
    distance_metric: DistanceMetric,
}

impl SimpleVectorReader {
    /// Create a new simple vector reader.
    pub fn new(
        vectors: Vec<(u64, String, Vector)>,
        dimension: usize,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        let vector_ids: Vec<(u64, String)> = vectors
            .iter()
            .map(|(id, field, _)| (*id, field.clone()))
            .collect();
        let vectors: HashMap<(u64, String), Vector> = vectors
            .into_iter()
            .map(|(id, field, vec)| ((id, field), vec))
            .collect();

        Ok(Self {
            vectors,
            vector_ids,
            dimension,
            distance_metric,
        })
    }
}

impl VectorIndexReader for SimpleVectorReader {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        Ok(self.vectors.get(&(doc_id, field_name.to_string())).cloned())
    }

    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>> {
        Ok(self
            .vectors
            .iter()
            .filter(|((id, _), _)| *id == doc_id)
            .map(|((_, field), vec)| (field.clone(), vec.clone()))
            .collect())
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        Ok(doc_ids
            .iter()
            .map(|(id, field)| self.vectors.get(&(*id, field.clone())).cloned())
            .collect())
    }

    fn vector_ids(&self) -> Result<Vec<(u64, String)>> {
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

    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool {
        self.vectors.contains_key(&(doc_id, field_name.to_string()))
    }

    fn get_vector_range(
        &self,
        start_doc_id: u64,
        end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>> {
        Ok(self
            .vectors
            .iter()
            .filter(|((id, _), _)| *id >= start_doc_id && *id <= end_doc_id)
            .map(|((id, field), v)| (*id, field.clone(), v.clone()))
            .collect())
    }

    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>> {
        Ok(self
            .vectors
            .iter()
            .filter(|((_, field), _)| field == field_name)
            .map(|((id, _), vec)| (*id, vec.clone()))
            .collect())
    }

    fn field_names(&self) -> Result<Vec<String>> {
        use std::collections::HashSet;
        let fields: HashSet<String> = self
            .vectors
            .keys()
            .map(|(_, field)| field.clone())
            .collect();
        Ok(fields.into_iter().collect())
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        Ok(Box::new(SimpleVectorIterator::new(
            self.vectors
                .iter()
                .map(|((id, field), v)| (*id, field.clone(), v.clone()))
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
    vectors: Vec<(u64, String, Vector)>,
    position: usize,
}

impl SimpleVectorIterator {
    /// Create a new simple vector iterator.
    pub fn new(vectors: Vec<(u64, String, Vector)>) -> Self {
        Self {
            vectors,
            position: 0,
        }
    }
}

impl VectorIterator for SimpleVectorIterator {
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>> {
        if self.position < self.vectors.len() {
            let result = self.vectors[self.position].clone();
            self.position += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    fn skip_to(&mut self, doc_id: u64, field_name: &str) -> Result<bool> {
        while self.position < self.vectors.len() {
            let (id, field, _) = &self.vectors[self.position];
            if *id > doc_id || (*id == doc_id && field.as_str() >= field_name) {
                return Ok(true);
            }
            self.position += 1;
        }
        Ok(false)
    }

    fn position(&self) -> (u64, String) {
        if self.position < self.vectors.len() {
            let (id, field, _) = &self.vectors[self.position];
            (*id, field.clone())
        } else {
            (u64::MAX, String::new())
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        Ok(())
    }
}

/// Factory for creating vector index readers.
///
/// This provides a single entry point for constructing concrete reader
/// implementations (Flat, HNSW, IVF) from serialized index data.
pub struct VectorIndexReaderFactory;

impl VectorIndexReaderFactory {
    /// Create a reader for a specific index type from serialized bytes.
    pub fn create_reader(
        index_type: &str,
        index_data: &[u8],
    ) -> Result<Arc<dyn VectorIndexReader>> {
        use crate::vector::index::flat::reader::FlatVectorIndexReader;
        use crate::vector::index::hnsw::reader::HnswIndexReader;
        use crate::vector::index::ivf::reader::IvfIndexReader;

        match index_type.to_lowercase().as_str() {
            "flat" => Ok(Arc::new(FlatVectorIndexReader::from_bytes(index_data)?)),
            "hnsw" => Ok(Arc::new(HnswIndexReader::from_bytes(index_data)?)),
            "ivf" => Ok(Arc::new(IvfIndexReader::from_bytes(index_data)?)),
            _ => Err(crate::error::SarissaError::InvalidOperation(format!(
                "Unknown index type: {index_type}"
            ))),
        }
    }
}
