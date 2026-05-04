//! HNSW vector field reader for approximate nearest neighbor search.
//!
//! This module provides a `VectorFieldReader` implementation that performs
//! approximate nearest neighbor search using HNSW (Hierarchical Navigable Small World) algorithm.

use std::cmp::Ordering;
use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::vector::core::vector::Vector;
use crate::vector::index::field::{
    FieldHit, FieldSearchInput, FieldSearchResults, VectorFieldReader, VectorFieldStats,
};
use crate::vector::reader::VectorIndexReader;

/// HNSW vector field reader that performs approximate nearest neighbor search.
///
/// This reader directly implements `VectorFieldReader` without going through
/// the legacy `VectorSearcher` adapter layer.
#[derive(Debug)]
pub struct HnswFieldReader {
    field_name: String,
    index_reader: Arc<dyn VectorIndexReader>,
    ef_search: usize,
}

impl HnswFieldReader {
    /// Default ef_search parameter value.
    pub const DEFAULT_EF_SEARCH: usize = 500;

    /// Create a new HNSW field reader.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The name of the vector field this reader serves
    /// * `index_reader` - The underlying index reader for vector access
    pub fn new(field_name: impl Into<String>, index_reader: Arc<dyn VectorIndexReader>) -> Self {
        Self {
            field_name: field_name.into(),
            index_reader,
            ef_search: Self::DEFAULT_EF_SEARCH,
        }
    }

    /// Create a new HNSW field reader with custom ef_search parameter.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The name of the vector field this reader serves
    /// * `index_reader` - The underlying index reader for vector access
    /// * `ef_search` - The search-time ef parameter (higher = more accurate but slower)
    pub fn with_ef_search(
        field_name: impl Into<String>,
        index_reader: Arc<dyn VectorIndexReader>,
        ef_search: usize,
    ) -> Self {
        Self {
            field_name: field_name.into(),
            index_reader,
            ef_search,
        }
    }

    /// Set the search parameter ef.
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.ef_search = ef_search;
    }

    /// Get the current ef_search value.
    pub fn ef_search(&self) -> usize {
        self.ef_search
    }

    /// Execute search for a single query vector.
    fn search_single_vector(
        &self,
        limit: usize,
        weight: f32,
        query: &Vector,
        allowed_ids: Option<&std::collections::HashSet<u64>>,
    ) -> Result<Vec<FieldHit>> {
        // Fetch the per-field doc-id slice from the reader's pre-built
        // index (#405). The `allowed_ids` filter — when supplied by the
        // caller — is applied against this already-narrowed slice rather
        // than the full corpus.
        let field_ids = self.index_reader.doc_ids_for_field(&self.field_name);
        let filtered: Vec<u64> = match allowed_ids {
            Some(allowed) => field_ids
                .iter()
                .copied()
                .filter(|id| allowed.contains(id))
                .collect(),
            None => field_ids.iter().copied().collect(),
        };

        // Linear scan: examine all filtered vectors (ef_search only applies to graph search)
        let mut candidates: Vec<(u64, f32, f32)> = Vec::with_capacity(filtered.len());

        for &doc_id in &filtered {
            if let Ok(Some(vector)) = self.index_reader.get_vector(doc_id, &self.field_name) {
                // Compute distance once and derive similarity (mirrors
                // the #404 pattern: `similarity()` would otherwise rerun
                // the SIMD distance kernel internally).
                let metric = self.index_reader.distance_metric();
                let distance = metric.distance(&query.data, &vector.data)?;
                let similarity = metric.distance_to_similarity(distance);
                candidates.push((doc_id, similarity, distance));
            }
        }

        // Sort by similarity (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Take top results and convert to FieldHit
        let top_k = limit.min(candidates.len());
        let hits: Vec<FieldHit> = candidates
            .into_iter()
            .take(top_k)
            .map(|(doc_id, similarity, distance)| FieldHit {
                doc_id,
                field: self.field_name.clone(),
                score: similarity * weight,
                distance,
            })
            .collect();

        Ok(hits)
    }
}

impl VectorFieldReader for HnswFieldReader {
    fn search(&self, request: FieldSearchInput) -> Result<FieldSearchResults> {
        // Validate field name
        if request.field != self.field_name {
            return Err(LaurusError::invalid_argument(format!(
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
            let effective_weight = query.weight;
            let query_vec = query.vector.clone();
            let hits = self.search_single_vector(
                request.limit,
                effective_weight,
                &query_vec,
                request.allowed_ids.as_ref(),
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

    use crate::vector::core::vector::Vector;
    use crate::vector::reader::SimpleVectorReader;
    use crate::vector::store::request::QueryVector;

    fn create_test_reader() -> Arc<dyn VectorIndexReader> {
        let vectors = vec![
            (1, "body".to_string(), Vector::new(vec![1.0, 0.0, 0.0])),
            (2, "body".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
            (3, "body".to_string(), Vector::new(vec![0.0, 0.0, 1.0])),
        ];
        Arc::new(SimpleVectorReader::new(vectors, 3, DistanceMetric::Cosine).unwrap())
    }

    fn create_query_vector(data: Vec<f32>) -> QueryVector {
        QueryVector {
            vector: Vector::new(data),
            weight: 1.0,
            fields: None,
        }
    }

    #[test]
    fn test_hnsw_field_reader_search() {
        let index_reader = create_test_reader();
        let reader = HnswFieldReader::new("body", index_reader);

        let query = create_query_vector(vec![1.0, 0.0, 0.0]);
        let input = FieldSearchInput {
            field: "body".to_string(),
            query_vectors: vec![query],
            limit: 10,
            allowed_ids: None,
        };

        let results = reader.search(input).unwrap();
        assert!(!results.hits.is_empty());
        assert_eq!(results.hits[0].doc_id, 1);
    }

    #[test]
    fn test_hnsw_field_reader_with_ef_search() {
        let index_reader = create_test_reader();
        let reader = HnswFieldReader::with_ef_search("body", index_reader, 100);

        assert_eq!(reader.ef_search(), 100);
    }

    #[test]
    fn test_hnsw_field_reader_field_mismatch() {
        let index_reader = create_test_reader();
        let reader = HnswFieldReader::new("body", index_reader);

        let query = create_query_vector(vec![1.0, 0.0, 0.0]);
        let input = FieldSearchInput {
            field: "wrong_field".to_string(),
            query_vectors: vec![query],
            limit: 10,
            allowed_ids: None,
        };

        let result = reader.search(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_hnsw_field_reader_stats() {
        let index_reader = create_test_reader();
        let reader = HnswFieldReader::new("body", index_reader);

        let stats = reader.stats().unwrap();
        assert_eq!(stats.dimension, 3);
        assert_eq!(stats.vector_count, 3);
    }
}
