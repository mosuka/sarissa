//! Compatibility adapters between doc-centric field traits and legacy vector indexes.

use std::cmp::Ordering;
use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{PlatypusError, Result};
use crate::vector::core::document::{FieldVectors, METADATA_WEIGHT};
use crate::vector::core::vector::Vector;
use crate::vector::engine::VectorFieldConfig;
use crate::vector::field::{
    FieldHit, FieldSearchInput, FieldSearchResults, VectorField, VectorFieldReader,
    VectorFieldStats, VectorFieldWriter,
};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::{VectorSearchRequest, VectorSearcher};
use crate::vector::writer::VectorIndexWriter;

/// Bridges the new doc-centric `VectorFieldWriter` trait to existing index writers.
#[derive(Debug)]
pub struct LegacyVectorFieldWriter<W: VectorIndexWriter> {
    field_name: String,
    writer: Mutex<W>,
}

impl<W: VectorIndexWriter> LegacyVectorFieldWriter<W> {
    /// Create a new adapter for the provided field name and index writer.
    pub fn new(field_name: impl Into<String>, writer: W) -> Self {
        Self {
            field_name: field_name.into(),
            writer: Mutex::new(writer),
        }
    }

    /// Returns the owning field name.
    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    fn to_legacy_vectors(
        &self,
        doc_id: u64,
        field: &FieldVectors,
        field_weight: f32,
    ) -> Vec<(u64, String, Vector)> {
        let mut vectors = Vec::with_capacity(field.vectors.len());
        for stored in &field.vectors {
            let mut vector = stored.to_vector();
            Self::apply_weight_metadata(&mut vector, field_weight);
            vectors.push((doc_id, self.field_name.clone(), vector));
        }
        vectors
    }

    fn apply_weight_metadata(vector: &mut Vector, field_weight: f32) {
        let stored_weight = vector
            .metadata
            .get(METADATA_WEIGHT)
            .and_then(|raw| raw.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(1.0);
        let combined = stored_weight * field_weight;
        vector
            .metadata
            .insert(METADATA_WEIGHT.to_string(), combined.to_string());
    }

    fn effective_field_weight(weight: f32) -> f32 {
        if weight.is_finite() && weight > 0.0 {
            weight
        } else {
            1.0
        }
    }

    #[cfg(test)]
    pub(crate) fn pending_vectors(&self) -> Vec<(u64, String, Vector)> {
        let guard = self.writer.lock();
        guard.vectors().to_vec()
    }
}

impl<W> VectorFieldWriter for LegacyVectorFieldWriter<W>
where
    W: VectorIndexWriter,
{
    fn add_field_vectors(&self, doc_id: u64, field: &FieldVectors, _version: u64) -> Result<()> {
        if field.vectors.is_empty() {
            return Ok(());
        }

        let mut guard = self.writer.lock();
        let legacy =
            self.to_legacy_vectors(doc_id, field, Self::effective_field_weight(field.weight));
        guard.add_vectors(legacy)
    }

    fn delete_document(&self, _doc_id: u64, _version: u64) -> Result<()> {
        // Full delete semantics will be wired once the registry tracks per-field segments.
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        // Flush is a no-op for the in-memory prototype.
        Ok(())
    }
}

/// Bridges `VectorFieldReader` calls to the legacy `VectorSearcher` API.
#[derive(Debug)]
pub struct LegacyVectorFieldReader {
    field_name: String,
    searcher: Arc<dyn VectorSearcher>,
    index_reader: Arc<dyn VectorIndexReader>,
}

impl LegacyVectorFieldReader {
    pub fn new(
        field_name: impl Into<String>,
        searcher: Arc<dyn VectorSearcher>,
        index_reader: Arc<dyn VectorIndexReader>,
    ) -> Self {
        Self {
            field_name: field_name.into(),
            searcher,
            index_reader,
        }
    }

    fn search_single_vector(
        &self,
        input: &FieldSearchInput,
        weight: f32,
        query: Vector,
    ) -> Result<Vec<FieldHit>> {
        let mut request = VectorSearchRequest::new(query);
        request.params.top_k = input.limit;
        request.field_name = Some(self.field_name.clone());
        let results = self.searcher.search(&request)?;
        Ok(results
            .results
            .into_iter()
            .map(|result| {
                let metadata = result.metadata;
                let vector_weight = Self::metadata_weight(&metadata);
                FieldHit {
                    doc_id: result.doc_id,
                    field: result.field_name,
                    score: result.similarity * weight * vector_weight,
                    distance: result.distance,
                    metadata,
                }
            })
            .collect())
    }

    fn metadata_weight(metadata: &HashMap<String, String>) -> f32 {
        metadata
            .get(METADATA_WEIGHT)
            .and_then(|raw| raw.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(1.0)
    }
}

impl VectorFieldReader for LegacyVectorFieldReader {
    fn search(&self, request: FieldSearchInput) -> Result<FieldSearchResults> {
        if request.field != self.field_name {
            return Err(PlatypusError::invalid_argument(format!(
                "field mismatch: expected '{}', got '{}'",
                self.field_name, request.field
            )));
        }

        if request.query_vectors.is_empty() {
            return Ok(FieldSearchResults::default());
        }

        let mut merged: HashMap<u64, FieldHit> = HashMap::new();
        for query in &request.query_vectors {
            let effective_weight = query.weight * query.vector.weight;
            let hits =
                self.search_single_vector(&request, effective_weight, query.vector.to_vector())?;
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

/// Concrete [`VectorField`] implementation backed by legacy adapters.
#[derive(Debug)]
pub struct AdapterBackedVectorField {
    name: String,
    config: VectorFieldConfig,
    writer: Arc<dyn VectorFieldWriter>,
    reader: Arc<dyn VectorFieldReader>,
}

impl AdapterBackedVectorField {
    /// Create a new adapter-backed vector field definition.
    pub fn new(
        name: impl Into<String>,
        config: VectorFieldConfig,
        writer: Arc<dyn VectorFieldWriter>,
        reader: Arc<dyn VectorFieldReader>,
    ) -> Self {
        Self {
            name: name.into(),
            config,
            writer,
            reader,
        }
    }

    /// Returns the shared writer handle.
    pub fn writer_handle(&self) -> &Arc<dyn VectorFieldWriter> {
        &self.writer
    }

    /// Returns the shared reader handle.
    pub fn reader_handle(&self) -> &Arc<dyn VectorFieldReader> {
        &self.reader
    }
}

impl VectorField for AdapterBackedVectorField {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &VectorFieldConfig {
        &self.config
    }

    fn writer(&self) -> &dyn VectorFieldWriter {
        self.writer.as_ref()
    }

    fn reader(&self) -> &dyn VectorFieldReader {
        self.reader.as_ref()
    }

    fn writer_handle(&self) -> Arc<dyn VectorFieldWriter> {
        self.writer.clone()
    }

    fn reader_handle(&self) -> Arc<dyn VectorFieldReader> {
        self.reader.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::distance::DistanceMetric;
    use crate::vector::core::document::{FieldVectors, StoredVector, VectorType};
    use crate::vector::core::vector::Vector;
    use crate::vector::engine::QueryVector;
    use crate::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
    use crate::vector::index::flat::writer::FlatIndexWriter;
    use crate::vector::index::hnsw::writer::HnswIndexWriter;
    use crate::vector::index::ivf::writer::IvfIndexWriter;
    use crate::vector::reader::SimpleVectorReader;
    use crate::vector::search::searcher::{
        VectorSearchRequest, VectorSearchResult, VectorSearchResults, VectorSearcher,
    };
    use crate::vector::writer::VectorIndexWriterConfig;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn sample_field_vectors() -> FieldVectors {
        let mut field = FieldVectors::default();
        field.vectors.push(StoredVector::new(
            Arc::<[f32]>::from([1.0_f32, 0.0_f32]),
            "embedder-a".into(),
            VectorType::Text,
        ));
        field
    }

    fn flat_writer() -> FlatIndexWriter {
        let mut config = FlatIndexConfig::default();
        config.dimension = 2;
        config.normalize_vectors = false;
        FlatIndexWriter::new(config, VectorIndexWriterConfig::default()).unwrap()
    }

    fn hnsw_writer() -> HnswIndexWriter {
        let mut config = HnswIndexConfig::default();
        config.dimension = 2;
        config.normalize_vectors = false;
        HnswIndexWriter::new(config, VectorIndexWriterConfig::default()).unwrap()
    }

    fn ivf_writer() -> IvfIndexWriter {
        let mut config = IvfIndexConfig::default();
        config.dimension = 2;
        config.normalize_vectors = false;
        IvfIndexWriter::new(config, VectorIndexWriterConfig::default()).unwrap()
    }

    #[test]
    fn adapter_buffers_vectors_in_inner_writer() {
        let adapter = LegacyVectorFieldWriter::new("body", flat_writer());
        let field = sample_field_vectors();

        adapter.add_field_vectors(7, &field, 1).unwrap();

        let pending = adapter.pending_vectors();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, 7);
        assert_eq!(pending[0].1, "body");
    }

    #[test]
    fn adapter_supports_hnsw_writer() {
        let adapter = LegacyVectorFieldWriter::new("body", hnsw_writer());
        let field = sample_field_vectors();

        adapter.add_field_vectors(3, &field, 1).unwrap();

        let pending = adapter.pending_vectors();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, 3);
    }

    #[test]
    fn adapter_supports_ivf_writer() {
        let adapter = LegacyVectorFieldWriter::new("body", ivf_writer());
        let field = sample_field_vectors();

        adapter.add_field_vectors(11, &field, 1).unwrap();

        let pending = adapter.pending_vectors();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, 11);
    }

    #[test]
    fn adapter_propagates_field_weight_into_metadata() {
        let adapter = LegacyVectorFieldWriter::new("body", flat_writer());
        let mut field = sample_field_vectors();
        field.weight = 2.5;
        field.vectors[0].weight = 0.4;

        adapter.add_field_vectors(5, &field, 1).unwrap();

        let pending = adapter.pending_vectors();
        assert_eq!(pending.len(), 1);
        let stored_weight = pending[0]
            .2
            .metadata
            .get(METADATA_WEIGHT)
            .and_then(|raw| raw.parse::<f32>().ok())
            .unwrap();
        assert!((stored_weight - 1.0).abs() < f32::EPSILON);
    }

    #[derive(Debug)]
    struct StubSearcher {
        responses: Vec<VectorSearchResult>,
    }

    impl VectorSearcher for StubSearcher {
        fn search(&self, _request: &VectorSearchRequest) -> Result<VectorSearchResults> {
            let mut results = VectorSearchResults::new();
            results.results = self.responses.clone();
            results.candidates_examined = self.responses.len();
            Ok(results)
        }

        fn count(&self, _request: VectorSearchRequest) -> Result<u64> {
            Ok(self.responses.len() as u64)
        }
    }

    #[test]
    fn legacy_reader_merges_weighted_scores() {
        let vectors = vec![
            (1, "body".to_string(), Vector::new(vec![1.0, 0.0])),
            (2, "body".to_string(), Vector::new(vec![0.0, 1.0])),
        ];
        let reader =
            Arc::new(SimpleVectorReader::new(vectors, 2, DistanceMetric::Cosine).expect("reader"))
                as Arc<dyn VectorIndexReader>;

        let responses = vec![
            VectorSearchResult {
                doc_id: 1,
                field_name: "body".to_string(),
                similarity: 0.9,
                distance: 0.1,
                vector: None,
                metadata: HashMap::new(),
            },
            VectorSearchResult {
                doc_id: 2,
                field_name: "body".to_string(),
                similarity: 0.4,
                distance: 0.6,
                vector: None,
                metadata: HashMap::new(),
            },
        ];

        let searcher = Arc::new(StubSearcher { responses }) as Arc<dyn VectorSearcher>;
        let reader_adapter = LegacyVectorFieldReader::new("body", searcher, reader);

        let mut stored = StoredVector::new(
            Arc::<[f32]>::from([1.0_f32, 0.0_f32]),
            String::new(),
            VectorType::Text,
        );
        stored.weight = 0.5; // additional weighting factor
        let query = QueryVector {
            vector: stored,
            weight: 2.0,
        };

        let request = FieldSearchInput {
            field: "body".to_string(),
            query_vectors: vec![query],
            limit: 5,
        };

        let results = reader_adapter.search(request).expect("search");
        assert_eq!(results.hits.len(), 2);
        // Effective weight = 2.0 * 0.5 = 1.0, so score should equal original similarity
        assert!((results.hits[0].score - 0.9).abs() < f32::EPSILON);
        assert_eq!(results.hits[0].doc_id, 1);
    }
}
