use parking_lot::RwLock;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use crate::error::{Result, SarissaError};
use crate::storage::Storage;
use crate::vector::core::document::{METADATA_WEIGHT, StoredVector};
use crate::vector::core::vector::Vector;
use crate::vector::engine::config::VectorFieldConfig;
use crate::vector::field::{
    FieldHit, FieldSearchInput, FieldSearchResults, VectorField, VectorFieldReader,
    VectorFieldStats, VectorFieldWriter,
};
use crate::vector::index::VectorIndexWriter;
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::searcher::HnswSearcher;
use crate::vector::index::hnsw::segment::manager::{ManagedSegmentInfo, SegmentManager};
use crate::vector::index::hnsw::segment::merge_engine::{MergeConfig, MergeEngine};
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::search::searcher::{
    VectorIndexSearchParams, VectorIndexSearchRequest, VectorIndexSearcher,
};
use crate::vector::writer::VectorIndexWriterConfig;
use std::cmp::Ordering;

/// A vector field implementation that partitions data into segments.
///
/// This implementation allows for efficient ingestion and background merging
/// of HNSW vector segments.
#[derive(Debug, Clone)]
pub struct SegmentedVectorField {
    /// Field name.
    pub name: String,

    /// Field configuration.
    pub config: VectorFieldConfig,

    /// Manager for segments.
    pub segment_manager: Arc<SegmentManager>,

    /// Storage backend.
    pub storage: Arc<dyn Storage>,

    /// Active segment for current writes.
    pub active_segment: Arc<RwLock<Option<(String, HnswIndexWriter)>>>,
}

impl SegmentedVectorField {
    pub fn create(
        name: impl Into<String>,
        config: VectorFieldConfig,
        segment_manager: Arc<SegmentManager>,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        let name_str = name.into();

        let field = Self {
            name: name_str,
            config,
            segment_manager,
            storage,
            active_segment: Arc::new(RwLock::new(None)),
        };

        Ok(field)
    }

    fn ensure_active_segment(&self) -> Result<()> {
        // Optimistic check
        if self.active_segment.read().is_some() {
            return Ok(());
        }

        let mut active_lock = self.active_segment.write();
        if active_lock.is_some() {
            return Ok(());
        }

        // Create new active segment
        let segment_id = self.segment_manager.generate_segment_id();

        // Get HNSW parameters from metadata if available
        let hnsw_config_meta = HnswMetadataConfig::from_metadata(&self.config.metadata);

        let hnsw_config = HnswIndexConfig {
            dimension: self.config.dimension,
            distance_metric: self.config.distance,
            m: hnsw_config_meta.m,
            ef_construction: hnsw_config_meta.ef_construction,
            normalize_vectors: self.config.distance
                == crate::vector::core::distance::DistanceMetric::Cosine,
            ..Default::default()
        };

        let writer_config = VectorIndexWriterConfig {
            ..Default::default()
        };

        let writer = HnswIndexWriter::with_storage(
            hnsw_config,
            writer_config,
            &segment_id,
            self.storage.clone(),
        )?;
        *active_lock = Some((segment_id, writer));

        Ok(())
    }

    /// Trigger a background merge of segments if needed.
    pub fn perform_merge(&self) -> Result<()> {
        let policy = crate::vector::index::hnsw::segment::merge_policy::SimpleMergePolicy::new();
        if let Some(candidate) = self.segment_manager.check_merge(&policy) {
            // Get HNSW parameters from metadata if available
            let hnsw_config_meta = HnswMetadataConfig::from_metadata(&self.config.metadata);

            let engine = MergeEngine::new(
                MergeConfig::default(),
                self.storage.clone(),
                HnswIndexConfig {
                    dimension: self.config.dimension,
                    m: hnsw_config_meta.m,
                    ef_construction: hnsw_config_meta.ef_construction,
                    ..Default::default()
                },
                VectorIndexWriterConfig {
                    ..Default::default()
                },
            );

            let new_segment_id = self.segment_manager.generate_segment_id();
            let result =
                engine.merge_segments(candidate.segments.clone(), new_segment_id.clone())?;

            let info = ManagedSegmentInfo::new(new_segment_id, result.stats.vectors_merged, 0, 0);

            self.segment_manager.apply_merge(candidate, info)?;
        }
        Ok(())
    }
}

impl VectorField for SegmentedVectorField {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &VectorFieldConfig {
        &self.config
    }

    fn writer(&self) -> &dyn VectorFieldWriter {
        self
    }

    fn reader(&self) -> &dyn VectorFieldReader {
        self
    }

    fn writer_handle(&self) -> Arc<dyn VectorFieldWriter> {
        Arc::new(self.clone())
    }

    fn reader_handle(&self) -> Arc<dyn VectorFieldReader> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl VectorFieldWriter for SegmentedVectorField {
    fn add_stored_vector(&self, doc_id: u64, vector: &StoredVector, _version: u64) -> Result<()> {
        let vec = vector.to_vector();

        // 2. Update memory
        self.ensure_active_segment()?;
        let mut active_opt = self.active_segment.write();
        if let Some((_, writer)) = active_opt.as_mut() {
            writer.add_vectors(vec![(doc_id, self.name.clone(), vec)])?;
        } else {
            return Err(SarissaError::internal(
                "No active segment available".to_string(),
            ));
        }
        Ok(())
    }

    fn has_storage(&self) -> bool {
        self.active_segment
            .read()
            .as_ref()
            .map(|(_, w)| w.has_storage())
            .unwrap_or(false)
    }

    fn vectors(&self) -> Vec<(u64, String, Vector)> {
        if let Some((_, writer)) = self.active_segment.read().as_ref() {
            writer.vectors().to_vec()
        } else {
            Vec::new()
        }
    }

    fn rebuild(&self, _vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        Ok(())
    }

    fn delete_document(&self, doc_id: u64, _version: u64) -> Result<()> {
        // 2. Update memory
        if let Some((_, writer)) = self.active_segment.write().as_mut() {
            let _ = writer.delete_document(doc_id);
        }
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        let mut active_lock = self.active_segment.write();
        if let Some((segment_id, mut writer)) = active_lock.take() {
            writer.finalize()?;
            writer.write()?;

            let vector_count = writer.vectors().len() as u64;

            let info = ManagedSegmentInfo::new(
                segment_id,
                vector_count,
                0, // offset
                0, // generation
            );

            self.segment_manager.add_segment(info)?;
        }
        Ok(())
    }

    fn optimize(&self) -> Result<()> {
        self.perform_merge()
    }
}

struct HnswMetadataConfig {
    m: usize,
    ef_construction: usize,
}

impl HnswMetadataConfig {
    fn from_metadata(metadata: &HashMap<String, String>) -> Self {
        let m = metadata
            .get("m")
            .and_then(|v: &String| v.parse::<usize>().ok())
            .unwrap_or(16);
        let ef_construction = metadata
            .get("ef_construction")
            .and_then(|v: &String| v.parse::<usize>().ok())
            .unwrap_or(200);
        Self { m, ef_construction }
    }
}

impl SegmentedVectorField {
    fn search_active_segment(
        &self,
        query: &Vector,
        limit: usize,
        weight: f32,
    ) -> Result<Vec<FieldHit>> {
        let active_opt = self.active_segment.read();
        let writer = match active_opt.as_ref() {
            Some((_, w)) => w,
            None => return Ok(Vec::new()),
        };

        let vectors = writer.vectors();
        let mut candidates = Vec::with_capacity(vectors.len());

        for (doc_id, _field, vector) in vectors {
            let similarity = self.config.distance.similarity(&query.data, &vector.data)?;
            let distance = self.config.distance.distance(&query.data, &vector.data)?;
            candidates.push((*doc_id, similarity, distance, vector.metadata.clone()));
        }

        // Sort by similarity descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let hits = candidates
            .into_iter()
            .take(limit)
            .map(
                |(doc_id, similarity, distance, metadata): (
                    u64,
                    f32,
                    f32,
                    HashMap<String, String>,
                )| {
                    let vector_weight = metadata
                        .get(METADATA_WEIGHT)
                        .and_then(|raw: &String| raw.parse::<f32>().ok())
                        .unwrap_or(1.0);
                    FieldHit {
                        doc_id,
                        field: self.name.clone(),
                        score: similarity * weight * vector_weight,
                        distance,
                        metadata,
                    }
                },
            )
            .collect();

        Ok(hits)
    }

    fn search_managed_segments(
        &self,
        query: &Vector,
        limit: usize,
        weight: f32,
    ) -> Result<Vec<FieldHit>> {
        let mut all_hits = Vec::new();
        let segments = self.segment_manager.list_segments();

        for info in segments {
            // Load reader for segment
            let reader = HnswIndexReader::load(
                self.storage.as_ref(),
                &info.segment_id,
                self.config.distance,
            )?;
            let searcher = HnswSearcher::new(Arc::new(reader))?;

            let mut params = VectorIndexSearchParams::default();
            params.top_k = limit;

            let request = VectorIndexSearchRequest {
                query: query.clone(),
                params,
                field_name: Some(self.name.clone()),
            };

            let results = searcher.search(&request)?;
            for res in results.results {
                all_hits.push(FieldHit {
                    doc_id: res.doc_id,
                    field: self.name.clone(),
                    score: res.similarity * weight,
                    distance: res.distance,
                    metadata: res.metadata,
                });
            }
        }

        Ok(all_hits)
    }
}

impl VectorFieldReader for SegmentedVectorField {
    fn search(&self, request: FieldSearchInput) -> Result<FieldSearchResults> {
        if request.field != self.name {
            return Err(SarissaError::invalid_argument(format!(
                "field mismatch: expected '{}', got '{}'",
                self.name, request.field
            )));
        }

        if request.query_vectors.is_empty() {
            return Ok(FieldSearchResults::default());
        }

        let mut merged: HashMap<u64, FieldHit> = HashMap::new();

        for query in &request.query_vectors {
            let effective_weight = query.weight * query.vector.weight;
            let query_vec = query.vector.to_vector();

            // 1. Search Active
            let active_hits =
                self.search_active_segment(&query_vec, request.limit, effective_weight)?;
            for hit in active_hits {
                match merged.entry(hit.doc_id) {
                    Entry::Vacant(e) => {
                        e.insert(hit);
                    }
                    Entry::Occupied(mut e) => {
                        let entry = e.get_mut();
                        entry.score += hit.score;
                        entry.distance = entry.distance.min(hit.distance);
                    }
                }
            }

            // 2. Search Managed
            let managed_hits =
                self.search_managed_segments(&query_vec, request.limit, effective_weight)?;
            for hit in managed_hits {
                match merged.entry(hit.doc_id) {
                    Entry::Vacant(e) => {
                        e.insert(hit);
                    }
                    Entry::Occupied(mut e) => {
                        let entry = e.get_mut();
                        entry.score += hit.score;
                        entry.distance = entry.distance.min(hit.distance);
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
        let mut active_count = 0;
        if let Some((_, writer)) = self.active_segment.read().as_ref() {
            active_count = writer.vectors().len();
        }

        let manager_stats = self.segment_manager.stats();
        let managed_count = manager_stats.total_vectors;

        Ok(VectorFieldStats {
            vector_count: active_count + managed_count as usize,
            dimension: self.config.dimension,
        })
    }
}
