//! Flat vector field reader for exact search.
//!
//! This module provides a `VectorFieldReader` implementation that performs
//! exact (brute force) nearest neighbor search on flat vector indices.

use std::cmp::Ordering;
use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;

use crate::error::{Result, SarissaError};
use crate::vector::core::document::METADATA_WEIGHT;
use crate::vector::core::vector::Vector;
use crate::vector::field::{
    FieldHit, FieldSearchInput, FieldSearchResults, VectorFieldReader, VectorFieldStats,
};
use crate::vector::reader::VectorIndexReader;

/// Flat vector field reader that performs exact (brute force) search.
///
/// This reader directly implements `VectorFieldReader` without going through
/// the legacy `VectorSearcher` adapter layer.
#[derive(Debug)]
pub struct FlatFieldReader {
    field_name: String,
    index_reader: Arc<dyn VectorIndexReader>,
}

impl FlatFieldReader {
    /// Create a new flat field reader.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The name of the vector field this reader serves
    /// * `index_reader` - The underlying index reader for vector access
    pub fn new(field_name: impl Into<String>, index_reader: Arc<dyn VectorIndexReader>) -> Self {
        Self {
            field_name: field_name.into(),
            index_reader,
        }
    }

    /// Execute search for a single query vector.
    fn search_single_vector(
        &self,
        limit: usize,
        weight: f32,
        query: &Vector,
    ) -> Result<Vec<FieldHit>> {
        // Get all vector IDs for this field
        let vector_ids = self.index_reader.vector_ids()?;
        let filtered_ids: Vec<(u64, String)> = vector_ids
            .into_iter()
            .filter(|(_, f)| f == &self.field_name)
            .collect();

        // Calculate similarities for all vectors
        let mut candidates: Vec<(u64, f32, f32, HashMap<String, String>)> =
            Vec::with_capacity(filtered_ids.len());

        for (doc_id, field_name) in filtered_ids {
            if let Ok(Some(vector)) = self.index_reader.get_vector(doc_id, &field_name) {
                let similarity = self
                    .index_reader
                    .distance_metric()
                    .similarity(&query.data, &vector.data)?;
                let distance = self
                    .index_reader
                    .distance_metric()
                    .distance(&query.data, &vector.data)?;
                candidates.push((doc_id, similarity, distance, vector.metadata.clone()));
            }
        }

        // Sort by similarity (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Take top results and convert to FieldHit
        let top_k = limit.min(candidates.len());
        let hits: Vec<FieldHit> = candidates
            .into_iter()
            .take(top_k)
            .map(|(doc_id, similarity, distance, metadata)| {
                let vector_weight = Self::metadata_weight(&metadata);
                FieldHit {
                    doc_id,
                    field: self.field_name.clone(),
                    score: similarity * weight * vector_weight,
                    distance,
                    metadata,
                }
            })
            .collect();

        Ok(hits)
    }

    /// Extract weight from vector metadata.
    fn metadata_weight(metadata: &HashMap<String, String>) -> f32 {
        metadata
            .get(METADATA_WEIGHT)
            .and_then(|raw| raw.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(1.0)
    }
}

impl VectorFieldReader for FlatFieldReader {
    fn search(&self, request: FieldSearchInput) -> Result<FieldSearchResults> {
        // Validate field name
        if request.field != self.field_name {
            return Err(SarissaError::invalid_argument(format!(
                "field mismatch: expected '{}', got '{}'",
                self.field_name, request.field
            )));
        }

        // Handle empty query
        if request.query_vectors.is_empty() {
            return Ok(FieldSearchResults::default());
        }

        // Merge results from all query vectors
        let mut merged: HashMap<u64, FieldHit> = HashMap::new();
        for query in &request.query_vectors {
            let effective_weight = query.weight * query.vector.weight;
            let hits = self.search_single_vector(
                request.limit,
                effective_weight,
                &query.vector.to_vector(),
            )?;

            for hit in hits {
                match merged.entry(hit.doc_id) {
                    Entry::Vacant(slot) => {
                        slot.insert(hit);
                    }
                    Entry::Occupied(mut slot) => {
                        let entry = slot.get_mut();
                        entry.score += hit.score;
                        entry.distance = entry.distance.min(hit.distance);
                        if entry.metadata.is_empty() {
                            entry.metadata = hit.metadata.clone();
                        }
                    }
                }
            }
        }

        // Sort by score and truncate to limit
        let mut hits: Vec<FieldHit> = merged.into_values().collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        if hits.len() > request.limit {
            hits.truncate(request.limit);
        }

        Ok(FieldSearchResults { hits })
    }

    fn stats(&self) -> Result<VectorFieldStats> {
        let stats = self.index_reader.stats();
        Ok(VectorFieldStats {
            vector_count: stats.vector_count,
            dimension: stats.dimension,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::distance::DistanceMetric;
    use crate::vector::core::document::{StoredVector, VectorType};
    use crate::vector::core::vector::Vector;
    use crate::vector::engine::QueryVector;
    use crate::vector::reader::SimpleVectorReader;

    fn create_test_reader() -> Arc<dyn VectorIndexReader> {
        let vectors = vec![
            (1, "body".to_string(), Vector::new(vec![1.0, 0.0, 0.0])),
            (2, "body".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
            (3, "body".to_string(), Vector::new(vec![0.0, 0.0, 1.0])),
        ];
        Arc::new(SimpleVectorReader::new(vectors, 3, DistanceMetric::Cosine).unwrap())
    }

    fn create_query_vector(data: Vec<f32>) -> QueryVector {
        let stored = StoredVector::new(data.into(), VectorType::Generic);
        QueryVector {
            vector: stored,
            weight: 1.0,
        }
    }

    #[test]
    fn test_flat_field_reader_search() {
        let index_reader = create_test_reader();
        let reader = FlatFieldReader::new("body", index_reader);

        let query = create_query_vector(vec![1.0, 0.0, 0.0]);
        let input = FieldSearchInput {
            field: "body".to_string(),
            query_vectors: vec![query],
            limit: 10,
        };

        let results = reader.search(input).unwrap();
        assert!(!results.hits.is_empty());
        assert_eq!(results.hits[0].doc_id, 1);
    }

    #[test]
    fn test_flat_field_reader_field_mismatch() {
        let index_reader = create_test_reader();
        let reader = FlatFieldReader::new("body", index_reader);

        let query = create_query_vector(vec![1.0, 0.0, 0.0]);
        let input = FieldSearchInput {
            field: "wrong_field".to_string(),
            query_vectors: vec![query],
            limit: 10,
        };

        let result = reader.search(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_flat_field_reader_empty_query() {
        let index_reader = create_test_reader();
        let reader = FlatFieldReader::new("body", index_reader);

        let input = FieldSearchInput {
            field: "body".to_string(),
            query_vectors: vec![],
            limit: 10,
        };

        let results = reader.search(input).unwrap();
        assert!(results.hits.is_empty());
    }

    #[test]
    fn test_flat_field_reader_stats() {
        let index_reader = create_test_reader();
        let reader = FlatFieldReader::new("body", index_reader);

        let stats = reader.stats().unwrap();
        assert_eq!(stats.dimension, 3);
        assert_eq!(stats.vector_count, 3);
    }
}
