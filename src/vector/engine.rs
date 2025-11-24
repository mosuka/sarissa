//! Experimental doc-centric vector collection implementation.

use std::cmp::Ordering as CmpOrdering;
use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::fmt;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::error::{PlatypusError, Result};
use crate::storage::Storage;
use crate::storage::prefixed::PrefixedStorage;
use crate::vector::DistanceMetric;
use crate::vector::core::document::{DocumentVectors, FieldVectors, StoredVector, VectorRole};
use crate::vector::core::vector::Vector;
use crate::vector::field::{
    FieldHit, FieldSearchInput, FieldSearchResults, VectorField, VectorFieldReader,
    VectorFieldStats, VectorFieldWriter,
};
use crate::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
use crate::vector::index::field::{
    AdapterBackedVectorField, LegacyVectorFieldReader, LegacyVectorFieldWriter,
};
use crate::vector::index::flat::{
    reader::FlatVectorIndexReader, searcher::FlatVectorSearcher, writer::FlatIndexWriter,
};
use crate::vector::index::hnsw::{
    reader::HnswIndexReader, searcher::HnswSearcher, writer::HnswIndexWriter,
};
use crate::vector::index::ivf::{
    reader::IvfIndexReader, searcher::IvfSearcher, writer::IvfIndexWriter,
};
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// High-level collection combining multiple vector fields.
pub struct VectorEngine {
    config: Arc<VectorEngineConfig>,
    fields: HashMap<String, FieldHandle>,
    registry: Arc<DocumentVectorRegistry>,
    wal: Arc<VectorWal>,
    storage: Arc<dyn Storage>,
    documents: Arc<RwLock<HashMap<u64, DocumentVectors>>>,
    snapshot_wal_seq: AtomicU64,
}

#[derive(Clone)]
struct FieldHandle {
    field: Arc<dyn VectorField>,
    runtime: Arc<FieldRuntime>,
}

#[derive(Debug)]
struct FieldRuntime {
    default_reader: Arc<dyn VectorFieldReader>,
    current_reader: RwLock<Arc<dyn VectorFieldReader>>,
    writer: Arc<dyn VectorFieldWriter>,
}

const FIELD_INDEX_BASENAME: &str = "index";
const REGISTRY_NAMESPACE: &str = "vector_registry";
const REGISTRY_SNAPSHOT_FILE: &str = "registry.json";
const REGISTRY_WAL_FILE: &str = "wal.json";
const DOCUMENT_SNAPSHOT_FILE: &str = "documents.json";
const DOCUMENT_SNAPSHOT_TEMP_FILE: &str = "documents.tmp";
const COLLECTION_MANIFEST_FILE: &str = "manifest.json";
const COLLECTION_MANIFEST_VERSION: u32 = 1;
#[cfg(test)]
const WAL_COMPACTION_THRESHOLD: usize = 4;
#[cfg(not(test))]
const WAL_COMPACTION_THRESHOLD: usize = 64;

impl FieldRuntime {
    fn new(reader: Arc<dyn VectorFieldReader>, writer: Arc<dyn VectorFieldWriter>) -> Self {
        Self {
            current_reader: RwLock::new(reader.clone()),
            default_reader: reader,
            writer,
        }
    }

    fn from_field(field: &Arc<dyn VectorField>) -> Arc<Self> {
        Arc::new(Self::new(field.reader_handle(), field.writer_handle()))
    }

    fn reader(&self) -> Arc<dyn VectorFieldReader> {
        self.current_reader.read().clone()
    }

    fn writer(&self) -> Arc<dyn VectorFieldWriter> {
        self.writer.clone()
    }

    fn replace_reader(&self, reader: Arc<dyn VectorFieldReader>) -> Arc<dyn VectorFieldReader> {
        let mut guard = self.current_reader.write();
        std::mem::replace(&mut *guard, reader)
    }

    fn reset_reader(&self) -> Arc<dyn VectorFieldReader> {
        self.replace_reader(self.default_reader.clone())
    }
}

impl fmt::Debug for VectorEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorEngine")
            .field("config", &self.config)
            .field("field_count", &self.fields.len())
            .finish()
    }
}

impl VectorEngine {
    pub fn new(
        config: VectorEngineConfig,
        storage: Arc<dyn Storage>,
        registry: Option<Arc<DocumentVectorRegistry>>,
    ) -> Result<Self> {
        let should_load_state = registry.is_none();
        let registry = registry.unwrap_or_else(|| Arc::new(DocumentVectorRegistry::default()));
        let mut collection = Self {
            config: Arc::new(config),
            fields: HashMap::new(),
            registry,
            wal: Arc::new(VectorWal::default()),
            storage,
            documents: Arc::new(RwLock::new(HashMap::new())),
            snapshot_wal_seq: AtomicU64::new(0),
        };
        collection.instantiate_configured_fields()?;
        if should_load_state {
            collection.load_persisted_state()?;
        }
        Ok(collection)
    }

    pub fn config(&self) -> &VectorEngineConfig {
        self.config.as_ref()
    }

    /// Register a concrete field implementation. Each field name must be unique.
    pub fn register_field(&mut self, field: Arc<dyn VectorField>) -> Result<()> {
        let name = field.name().to_string();
        if self.fields.contains_key(&name) {
            return Err(PlatypusError::invalid_config(format!(
                "vector field '{name}' is already registered"
            )));
        }
        let runtime = FieldRuntime::from_field(&field);
        self.fields.insert(name, FieldHandle { field, runtime });
        Ok(())
    }

    /// Convenience helper to register a field backed by legacy adapters.
    pub fn register_adapter_field(
        &mut self,
        name: impl Into<String>,
        config: VectorFieldConfig,
        writer: Arc<dyn VectorFieldWriter>,
        reader: Arc<dyn VectorFieldReader>,
    ) -> Result<()> {
        let field: Arc<dyn VectorField> =
            Arc::new(AdapterBackedVectorField::new(name, config, writer, reader));
        self.register_field(field)
    }

    /// Replace the active reader for a field, returning the previously registered reader.
    pub fn replace_field_reader(
        &self,
        field_name: &str,
        reader: Arc<dyn VectorFieldReader>,
    ) -> Result<Arc<dyn VectorFieldReader>> {
        let field = self.fields.get(field_name).ok_or_else(|| {
            PlatypusError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;
        Ok(field.runtime.replace_reader(reader))
    }

    /// Reset a field reader back to the default runtime reader captured at registration time.
    pub fn reset_field_reader(&self, field_name: &str) -> Result<Arc<dyn VectorFieldReader>> {
        let field = self.fields.get(field_name).ok_or_else(|| {
            PlatypusError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;
        Ok(field.runtime.reset_reader())
    }

    /// Aggregate collection-level statistics across all registered vector fields.
    pub fn stats(&self) -> Result<VectorEngineStats> {
        let mut fields = HashMap::with_capacity(self.fields.len());
        for (name, field) in &self.fields {
            let stats = field.runtime.reader().stats()?;
            fields.insert(name.clone(), stats);
        }

        Ok(VectorEngineStats {
            document_count: self.registry.as_ref().document_count(),
            fields,
        })
    }

    /// Retrieve statistics for a specific field.
    pub fn field_stats(&self, field_name: &str) -> Result<VectorFieldStats> {
        let field = self.fields.get(field_name).ok_or_else(|| {
            PlatypusError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;
        field.runtime.reader().stats()
    }

    /// Build a legacy delegate index from the current in-memory store and swap the field reader.
    pub fn materialize_delegate_reader(
        &self,
        field_name: &str,
    ) -> Result<Arc<dyn VectorFieldReader>> {
        let handle = self.fields.get(field_name).ok_or_else(|| {
            PlatypusError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;

        let in_memory = handle
            .field
            .as_any()
            .downcast_ref::<InMemoryVectorField>()
            .ok_or_else(|| {
                PlatypusError::InvalidOperation(format!(
                    "field '{field_name}' does not support delegate materialization"
                ))
            })?;

        let vectors = in_memory.vector_tuples();
        self.write_field_delegate_index(field_name, in_memory.config(), vectors)?;
        let reader = self.load_delegate_reader(field_name, in_memory.config())?;
        let _ = self.replace_field_reader(field_name, reader.clone())?;
        Ok(reader)
    }

    pub fn upsert_document(&self, document: DocumentVectors) -> Result<RegistryVersion> {
        self.validate_document_fields(&document)?;
        let doc_id = document.doc_id;
        let entries = build_field_entries(&document);
        let version = self
            .registry
            .upsert(document.doc_id, &entries, document.metadata.clone())?;

        if let Err(err) = self.apply_field_updates(document.doc_id, version, &document.fields) {
            // Best-effort rollback to keep registry consistent with field state.
            let _ = self.registry.delete(doc_id);
            return Err(err);
        }

        let cached_document = document.clone();
        self.wal.append(WalPayload::Upsert { document })?;
        self.documents.write().insert(doc_id, cached_document);
        self.persist_state()?;
        self.maybe_compact_wal()?;
        Ok(version)
    }

    pub fn delete_document(&self, doc_id: u64) -> Result<()> {
        let entry = self
            .registry
            .get(doc_id)
            .ok_or_else(|| PlatypusError::not_found(format!("doc_id {doc_id}")))?;
        self.delete_fields_for_entry(doc_id, &entry)?;

        self.registry.delete(doc_id)?;
        self.wal.append(WalPayload::Delete { doc_id })?;
        self.documents.write().remove(&doc_id);
        self.persist_state()?;
        self.maybe_compact_wal()
    }

    pub fn search(&self, query: &VectorEngineSearchRequest) -> Result<VectorEngineSearchResults> {
        if query.query_vectors.is_empty() {
            return Err(PlatypusError::invalid_argument(
                "VectorEngineSearchRequest requires at least one query vector",
            ));
        }

        if query.limit == 0 {
            return Ok(VectorEngineSearchResults::default());
        }

        if query.overfetch < 1.0 {
            return Err(PlatypusError::invalid_argument(
                "VectorEngineSearchRequest overfetch must be >= 1.0",
            ));
        }

        if matches!(query.score_mode, VectorScoreMode::LateInteraction) {
            return Err(PlatypusError::invalid_argument(
                "VectorScoreMode::LateInteraction is not supported yet",
            ));
        }

        let target_fields = self.resolve_fields(query)?;
        let filter_matches = self.build_filter_matches(query, &target_fields);
        if let Some(matches) = &filter_matches {
            if matches.is_empty() {
                return Ok(VectorEngineSearchResults::default());
            }
        }
        let mut doc_hits: HashMap<u64, VectorEngineHit> = HashMap::new();
        let mut fields_with_queries = 0_usize;
        let field_limit = self.scaled_field_limit(query.limit, query.overfetch);

        for field_name in target_fields {
            let field = self
                .fields
                .get(&field_name)
                .ok_or_else(|| PlatypusError::not_found(format!("vector field '{field_name}'")))?;
            let matching_vectors = self.query_vectors_for_field(field.field.config(), query);
            if matching_vectors.is_empty() {
                continue;
            }

            fields_with_queries += 1;

            let field_query = FieldSearchInput {
                field: field_name.clone(),
                query_vectors: matching_vectors,
                limit: field_limit,
            };

            let field_results = field.runtime.reader().search(field_query)?;
            let field_weight = field.field.config().base_weight;

            self.merge_field_hits(
                &mut doc_hits,
                field_results.hits,
                field_weight,
                query.score_mode,
                filter_matches.as_ref(),
            )?;
        }

        if fields_with_queries == 0 {
            return Err(PlatypusError::invalid_argument(
                "no query vectors matched the requested fields",
            ));
        }

        let mut hits: Vec<VectorEngineHit> = doc_hits.into_values().collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(CmpOrdering::Equal));
        if hits.len() > query.limit {
            hits.truncate(query.limit);
        }

        Ok(VectorEngineSearchResults { hits })
    }

    fn resolve_fields(&self, query: &VectorEngineSearchRequest) -> Result<Vec<String>> {
        let mut candidates: Vec<String> = if let Some(selectors) = &query.fields {
            self.apply_field_selectors(selectors)?
        } else if !self.config.default_fields.is_empty() {
            self.config.default_fields.clone()
        } else {
            self.fields.keys().cloned().collect()
        };

        if candidates.is_empty() {
            return Err(PlatypusError::invalid_config(
                "VectorEngine has no fields configured",
            ));
        }

        let mut seen = HashSet::new();
        candidates.retain(|field| seen.insert(field.clone()));
        Ok(candidates)
    }

    fn build_filter_matches(
        &self,
        query: &VectorEngineSearchRequest,
        target_fields: &[String],
    ) -> Option<RegistryFilterMatches> {
        query
            .filter
            .as_ref()
            .filter(|filter| !filter.is_empty())
            .map(|filter| self.registry.filter_matches(filter, target_fields))
    }

    fn apply_field_selectors(&self, selectors: &[FieldSelector]) -> Result<Vec<String>> {
        if selectors.is_empty() {
            return Err(PlatypusError::invalid_argument(
                "VectorEngineSearchRequest fields selector list is empty",
            ));
        }

        let mut resolved = Vec::new();
        for selector in selectors {
            match selector {
                FieldSelector::Exact(name) => {
                    if self.fields.contains_key(name) {
                        resolved.push(name.clone());
                    } else {
                        return Err(PlatypusError::not_found(format!(
                            "vector field '{name}' is not registered"
                        )));
                    }
                }
                FieldSelector::Prefix(prefix) => {
                    let mut matched = false;
                    for field in self.fields.keys() {
                        if field.starts_with(prefix) {
                            resolved.push(field.clone());
                            matched = true;
                        }
                    }
                    if !matched {
                        return Err(PlatypusError::not_found(format!(
                            "no vector fields match prefix '{prefix}'"
                        )));
                    }
                }
                FieldSelector::Role(role) => {
                    let mut matched = false;
                    for (field_name, field_handle) in &self.fields {
                        if field_handle.field.config().role == *role {
                            resolved.push(field_name.clone());
                            matched = true;
                        }
                    }
                    if !matched {
                        return Err(PlatypusError::not_found(format!(
                            "no vector fields registered with role '{role:?}'"
                        )));
                    }
                }
            }
        }
        Ok(resolved)
    }

    fn scaled_field_limit(&self, limit: usize, overfetch: f32) -> usize {
        let scaled = (limit as f32 * overfetch).ceil() as usize;
        scaled.max(limit).max(1)
    }

    fn query_vectors_for_field(
        &self,
        config: &VectorFieldConfig,
        query: &VectorEngineSearchRequest,
    ) -> Vec<QueryVector> {
        query
            .query_vectors
            .iter()
            .cloned()
            .filter(|candidate| {
                candidate.vector.embedder_id == config.embedder_id
                    && candidate.vector.role == config.role
            })
            .collect()
    }

    fn merge_field_hits(
        &self,
        doc_hits: &mut HashMap<u64, VectorEngineHit>,
        hits: Vec<FieldHit>,
        field_weight: f32,
        score_mode: VectorScoreMode,
        filter_matches: Option<&RegistryFilterMatches>,
    ) -> Result<()> {
        for hit in hits {
            if let Some(matches) = filter_matches {
                if !matches.contains_doc(hit.doc_id) {
                    continue;
                }
                if !matches.field_allowed(hit.doc_id, &hit.field) {
                    continue;
                }
            }

            let weighted_score = hit.score * field_weight;
            let entry = doc_hits
                .entry(hit.doc_id)
                .or_insert_with(|| VectorEngineHit {
                    doc_id: hit.doc_id,
                    score: 0.0,
                    field_hits: Vec::new(),
                });

            match score_mode {
                VectorScoreMode::WeightedSum => {
                    entry.score += weighted_score;
                }
                VectorScoreMode::MaxSim => {
                    entry.score = entry.score.max(weighted_score);
                }
                VectorScoreMode::LateInteraction => {
                    return Err(PlatypusError::invalid_argument(
                        "VectorScoreMode::LateInteraction is not supported yet",
                    ));
                }
            }
            entry.field_hits.push(hit);
        }

        Ok(())
    }

    fn instantiate_configured_fields(&mut self) -> Result<()> {
        let configs: Vec<(String, VectorFieldConfig)> = self
            .config
            .fields
            .iter()
            .map(|(name, config)| (name.clone(), config.clone()))
            .collect();

        for (name, config) in configs {
            let delegate = self.build_delegate_writer(&name, &config)?;
            let field = Arc::new(InMemoryVectorField::new(name, config, delegate)?);
            self.register_field(field)?;
        }
        Ok(())
    }

    fn build_delegate_writer(
        &self,
        field_name: &str,
        config: &VectorFieldConfig,
    ) -> Result<Option<Arc<dyn VectorFieldWriter>>> {
        if config.dimension == 0 {
            return Ok(None);
        }

        let writer_config = VectorIndexWriterConfig::default();
        let storage = self.field_storage(field_name);
        let delegate: Arc<dyn VectorFieldWriter> = match config.index {
            VectorIndexKind::Flat => {
                let mut flat = FlatIndexConfig::default();
                flat.dimension = config.dimension;
                flat.distance_metric = config.distance;
                let writer =
                    FlatIndexWriter::with_storage(flat, writer_config.clone(), storage.clone())?;
                Arc::new(LegacyVectorFieldWriter::new(field_name.to_string(), writer))
            }
            VectorIndexKind::Hnsw => {
                let mut hnsw = HnswIndexConfig::default();
                hnsw.dimension = config.dimension;
                hnsw.distance_metric = config.distance;
                let writer =
                    HnswIndexWriter::with_storage(hnsw, writer_config.clone(), storage.clone())?;
                Arc::new(LegacyVectorFieldWriter::new(field_name.to_string(), writer))
            }
            VectorIndexKind::Ivf => {
                let mut ivf = IvfIndexConfig::default();
                ivf.dimension = config.dimension;
                ivf.distance_metric = config.distance;
                let writer = IvfIndexWriter::with_storage(ivf, writer_config, storage)?;
                Arc::new(LegacyVectorFieldWriter::new(field_name.to_string(), writer))
            }
        };
        Ok(Some(delegate))
    }

    fn write_field_delegate_index(
        &self,
        field_name: &str,
        config: &VectorFieldConfig,
        vectors: Vec<(u64, String, Vector)>,
    ) -> Result<()> {
        if config.dimension == 0 {
            return Err(PlatypusError::invalid_config(format!(
                "vector field '{field_name}' cannot materialize a zero-dimension index"
            )));
        }

        let storage = self.field_storage(field_name);
        let mut pending_vectors = Some(vectors);
        match config.index {
            VectorIndexKind::Flat => {
                let mut flat = FlatIndexConfig::default();
                flat.dimension = config.dimension;
                flat.distance_metric = config.distance;
                let mut writer = FlatIndexWriter::with_storage(
                    flat,
                    VectorIndexWriterConfig::default(),
                    storage.clone(),
                )?;
                let vectors = pending_vectors.take().unwrap_or_default();
                writer.build(vectors)?;
                writer.finalize()?;
                writer.write(FIELD_INDEX_BASENAME)?;
            }
            VectorIndexKind::Hnsw => {
                let mut hnsw = HnswIndexConfig::default();
                hnsw.dimension = config.dimension;
                hnsw.distance_metric = config.distance;
                let mut writer = HnswIndexWriter::with_storage(
                    hnsw,
                    VectorIndexWriterConfig::default(),
                    storage.clone(),
                )?;
                let vectors = pending_vectors.take().unwrap_or_default();
                writer.build(vectors)?;
                writer.finalize()?;
                writer.write(FIELD_INDEX_BASENAME)?;
            }
            VectorIndexKind::Ivf => {
                let mut ivf = IvfIndexConfig::default();
                ivf.dimension = config.dimension;
                ivf.distance_metric = config.distance;
                let mut writer = IvfIndexWriter::with_storage(
                    ivf,
                    VectorIndexWriterConfig::default(),
                    storage.clone(),
                )?;
                let vectors = pending_vectors.take().unwrap_or_default();
                writer.build(vectors)?;
                writer.finalize()?;
                writer.write(FIELD_INDEX_BASENAME)?;
            }
        }

        Ok(())
    }

    fn load_delegate_reader(
        &self,
        field_name: &str,
        config: &VectorFieldConfig,
    ) -> Result<Arc<dyn VectorFieldReader>> {
        let storage = self.field_storage(field_name);
        Ok(match config.index {
            VectorIndexKind::Flat => {
                let reader = Arc::new(FlatVectorIndexReader::load(
                    storage.clone(),
                    FIELD_INDEX_BASENAME,
                    config.distance,
                )?);
                let searcher = Arc::new(FlatVectorSearcher::new(reader.clone())?);
                Arc::new(LegacyVectorFieldReader::new(
                    field_name.to_string(),
                    searcher,
                    reader,
                ))
            }
            VectorIndexKind::Hnsw => {
                let reader = Arc::new(HnswIndexReader::load(
                    storage.clone(),
                    FIELD_INDEX_BASENAME,
                    config.distance,
                )?);
                let searcher = Arc::new(HnswSearcher::new(reader.clone())?);
                Arc::new(LegacyVectorFieldReader::new(
                    field_name.to_string(),
                    searcher,
                    reader,
                ))
            }
            VectorIndexKind::Ivf => {
                let reader = Arc::new(IvfIndexReader::load(
                    storage.clone(),
                    FIELD_INDEX_BASENAME,
                    config.distance,
                )?);
                let mut searcher = IvfSearcher::new(reader.clone())?;
                searcher.set_n_probe(4);
                let searcher = Arc::new(searcher);
                Arc::new(LegacyVectorFieldReader::new(
                    field_name.to_string(),
                    searcher,
                    reader,
                ))
            }
        })
    }

    fn field_storage(&self, field_name: &str) -> Arc<dyn Storage> {
        let prefix = Self::field_storage_prefix(field_name);
        Arc::new(PrefixedStorage::new(prefix, self.storage.clone())) as Arc<dyn Storage>
    }

    fn registry_storage(&self) -> Arc<dyn Storage> {
        Arc::new(PrefixedStorage::new(
            REGISTRY_NAMESPACE,
            self.storage.clone(),
        )) as Arc<dyn Storage>
    }

    fn field_storage_prefix(field_name: &str) -> String {
        let mut sanitized: String = field_name
            .chars()
            .map(|ch| match ch {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
                _ => '_',
            })
            .collect();
        if sanitized.is_empty() {
            sanitized.push_str("field");
        }
        format!("vector_fields/{sanitized}")
    }

    fn validate_document_fields(&self, document: &DocumentVectors) -> Result<()> {
        for field_name in document.fields.keys() {
            if !self.fields.contains_key(field_name) {
                return Err(PlatypusError::invalid_argument(format!(
                    "vector field '{field_name}' is not registered"
                )));
            }
        }
        Ok(())
    }

    fn delete_fields_for_entry(&self, doc_id: u64, entry: &DocumentEntry) -> Result<()> {
        for (field_name, field_entry) in &entry.fields {
            let field = self.fields.get(field_name).ok_or_else(|| {
                PlatypusError::not_found(format!(
                    "vector field '{field_name}' not registered during delete"
                ))
            })?;
            field
                .runtime
                .writer()
                .delete_document(doc_id, field_entry.version)?;
        }
        Ok(())
    }

    fn apply_field_updates(
        &self,
        doc_id: u64,
        version: RegistryVersion,
        fields: &HashMap<String, FieldVectors>,
    ) -> Result<()> {
        for (field_name, field_vectors) in fields {
            let field = self.fields.get(field_name).ok_or_else(|| {
                PlatypusError::not_found(format!("vector field '{field_name}' is not registered"))
            })?;
            field
                .runtime
                .writer()
                .add_field_vectors(doc_id, field_vectors, version)?;
        }
        Ok(())
    }

    fn load_persisted_state(&mut self) -> Result<()> {
        let storage = self.registry_storage();
        if storage.file_exists(REGISTRY_SNAPSHOT_FILE) {
            let mut input = storage.open_input(REGISTRY_SNAPSHOT_FILE)?;
            let mut buffer = Vec::new();
            input.read_to_end(&mut buffer)?;
            input.close()?;
            self.registry = Arc::new(DocumentVectorRegistry::from_snapshot(&buffer)?);
        }

        if storage.file_exists(REGISTRY_WAL_FILE) {
            let mut input = storage.open_input(REGISTRY_WAL_FILE)?;
            let mut buffer = Vec::new();
            input.read_to_end(&mut buffer)?;
            input.close()?;
            if !buffer.is_empty() {
                let records: Vec<WalRecord> = serde_json::from_slice(&buffer)?;
                self.wal = Arc::new(VectorWal::from_records(records));
            }
        }

        self.load_document_snapshot(storage.clone())?;
        self.load_collection_manifest(storage.clone())?;
        self.replay_wal_into_fields()?;
        self.persist_manifest()
    }

    fn load_document_snapshot(&self, storage: Arc<dyn Storage>) -> Result<()> {
        if !storage.file_exists(DOCUMENT_SNAPSHOT_FILE) {
            self.documents.write().clear();
            self.snapshot_wal_seq.store(0, Ordering::SeqCst);
            return Ok(());
        }

        let mut input = storage.open_input(DOCUMENT_SNAPSHOT_FILE)?;
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer)?;
        input.close()?;

        if buffer.is_empty() {
            self.documents.write().clear();
            self.snapshot_wal_seq.store(0, Ordering::SeqCst);
            return Ok(());
        }

        let snapshot = match serde_json::from_slice::<DocumentSnapshot>(&buffer) {
            Ok(snapshot) => snapshot,
            Err(primary_err) => {
                let docs: Vec<DocumentVectors> =
                    serde_json::from_slice(&buffer).map_err(|_| primary_err)?;
                DocumentSnapshot {
                    last_wal_seq: 0,
                    documents: docs,
                }
            }
        };
        let map = snapshot
            .documents
            .into_iter()
            .map(|doc| (doc.doc_id, doc))
            .collect();
        *self.documents.write() = map;
        self.snapshot_wal_seq
            .store(snapshot.last_wal_seq, Ordering::SeqCst);
        Ok(())
    }

    fn load_collection_manifest(&self, storage: Arc<dyn Storage>) -> Result<()> {
        if !storage.file_exists(COLLECTION_MANIFEST_FILE) {
            return Ok(());
        }

        let mut input = storage.open_input(COLLECTION_MANIFEST_FILE)?;
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer)?;
        input.close()?;
        if buffer.is_empty() {
            return Ok(());
        }

        let manifest: CollectionManifest = serde_json::from_slice(&buffer)?;
        if manifest.version != COLLECTION_MANIFEST_VERSION {
            return Err(PlatypusError::invalid_config(format!(
                "collection manifest version mismatch: expected {}, found {}",
                COLLECTION_MANIFEST_VERSION, manifest.version
            )));
        }

        let snapshot_seq = self.snapshot_wal_seq.load(Ordering::SeqCst);
        if manifest.snapshot_wal_seq != snapshot_seq {
            return Err(PlatypusError::invalid_config(format!(
                "collection manifest snapshot sequence {} does not match persisted snapshot {}",
                manifest.snapshot_wal_seq, snapshot_seq
            )));
        }

        if manifest.wal_last_seq < manifest.snapshot_wal_seq {
            return Err(PlatypusError::invalid_config(
                "collection manifest WAL sequence regressed",
            ));
        }

        Ok(())
    }

    fn replay_wal_into_fields(&self) -> Result<()> {
        let mut documents = self.documents.read().clone();
        self.apply_documents_to_fields(&documents)?;
        let mut records = self.wal.records();
        let start_seq = self
            .snapshot_wal_seq
            .load(Ordering::SeqCst)
            .saturating_add(1);
        let mut applied_seq = self.snapshot_wal_seq.load(Ordering::SeqCst);
        if records.is_empty() {
            *self.documents.write() = documents;
            return Ok(());
        }

        records.sort_by(|a, b| a.seq.cmp(&b.seq));
        for record in records.into_iter() {
            if record.seq < start_seq {
                continue;
            }
            applied_seq = record.seq;
            match record.payload {
                WalPayload::Upsert { document } => {
                    if document.fields.is_empty() {
                        documents.remove(&document.doc_id);
                        continue;
                    }
                    if let Some(entry) = self.registry.get(document.doc_id) {
                        self.apply_field_updates(document.doc_id, entry.version, &document.fields)?;
                    }
                    documents.insert(document.doc_id, document);
                }
                WalPayload::Delete { doc_id } => {
                    if let Some(entry) = self.registry.get(doc_id) {
                        self.delete_fields_for_entry(doc_id, &entry)?;
                    }
                    documents.remove(&doc_id);
                }
            }
        }

        if applied_seq > self.snapshot_wal_seq.load(Ordering::SeqCst) {
            self.snapshot_wal_seq.store(applied_seq, Ordering::SeqCst);
        }

        *self.documents.write() = documents;
        Ok(())
    }

    fn apply_documents_to_fields(&self, documents: &HashMap<u64, DocumentVectors>) -> Result<()> {
        for (doc_id, document) in documents.iter() {
            if let Some(entry) = self.registry.get(*doc_id) {
                if document.fields.is_empty() {
                    continue;
                }
                self.apply_field_updates(*doc_id, entry.version, &document.fields)?;
            }
        }
        Ok(())
    }

    fn persist_state(&self) -> Result<()> {
        self.persist_registry_snapshot()?;
        self.persist_document_snapshot()?;
        self.persist_wal()?;
        self.persist_manifest()
    }

    fn persist_registry_snapshot(&self) -> Result<()> {
        let storage = self.registry_storage();
        let snapshot = self.registry.snapshot()?;
        self.write_atomic(storage, REGISTRY_SNAPSHOT_FILE, &snapshot)
    }

    fn persist_document_snapshot(&self) -> Result<()> {
        let storage = self.registry_storage();
        let guard = self.documents.read();
        let documents: Vec<DocumentVectors> = guard.values().cloned().collect();
        drop(guard);
        let snapshot = DocumentSnapshot {
            last_wal_seq: self.wal.last_seq(),
            documents,
        };
        let serialized = serde_json::to_vec(&snapshot)?;

        if serialized.len() > 256 * 1024 {
            self.write_atomic(storage.clone(), DOCUMENT_SNAPSHOT_TEMP_FILE, &serialized)?;
            storage.delete_file(DOCUMENT_SNAPSHOT_FILE).ok();
            storage.rename_file(DOCUMENT_SNAPSHOT_TEMP_FILE, DOCUMENT_SNAPSHOT_FILE)?;
        } else {
            self.write_atomic(storage.clone(), DOCUMENT_SNAPSHOT_FILE, &serialized)?;
        }

        self.snapshot_wal_seq
            .store(snapshot.last_wal_seq, Ordering::SeqCst);
        Ok(())
    }

    fn persist_manifest(&self) -> Result<()> {
        let storage = self.registry_storage();
        let manifest = CollectionManifest {
            version: COLLECTION_MANIFEST_VERSION,
            snapshot_wal_seq: self.snapshot_wal_seq.load(Ordering::SeqCst),
            wal_last_seq: self.wal.last_seq(),
        };
        let serialized = serde_json::to_vec(&manifest)?;
        self.write_atomic(storage, COLLECTION_MANIFEST_FILE, &serialized)
    }

    fn persist_wal(&self) -> Result<()> {
        let storage = self.registry_storage();
        let records = self.wal.records();
        let serialized = serde_json::to_vec(&records)?;
        self.write_atomic(storage, REGISTRY_WAL_FILE, &serialized)
    }

    fn maybe_compact_wal(&self) -> Result<()> {
        if self.wal.len() > WAL_COMPACTION_THRESHOLD {
            self.compact_wal()?;
        }
        Ok(())
    }

    fn compact_wal(&self) -> Result<()> {
        let documents = self.documents.read().clone();
        if documents.is_empty() {
            self.wal.replace_records(Vec::new());
            return self.persist_wal();
        }

        let mut entries: Vec<(u64, DocumentVectors)> = documents.into_iter().collect();
        entries.sort_by(|(a_id, _), (b_id, _)| a_id.cmp(&b_id));
        let mut records = Vec::with_capacity(entries.len());
        for (idx, (_, document)) in entries.into_iter().enumerate() {
            records.push(WalRecord {
                seq: (idx as u64) + 1,
                payload: WalPayload::Upsert { document },
            });
        }
        self.wal.replace_records(records);
        self.persist_wal()
    }

    fn write_atomic(&self, storage: Arc<dyn Storage>, name: &str, bytes: &[u8]) -> Result<()> {
        let tmp_name = format!("{name}.tmp");
        let mut output = storage.create_output(&tmp_name)?;
        output.write_all(bytes)?;
        output.flush_and_sync()?;
        output.close()?;
        if storage.file_exists(name) {
            storage.delete_file(name)?;
        }
        storage.rename_file(&tmp_name, name)
    }
}

fn build_field_entries(document: &DocumentVectors) -> Vec<FieldEntry> {
    document
        .fields
        .iter()
        .map(|(name, field)| FieldEntry {
            field_name: name.clone(),
            version: 0,
            vector_count: field.vector_count(),
            weight: if field.weight == 0.0 {
                1.0
            } else {
                field.weight
            },
            metadata: field.metadata.clone(),
        })
        .collect()
}

/// Configuration for a single vector collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEngineConfig {
    pub fields: HashMap<String, VectorFieldConfig>,
    pub default_fields: Vec<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl VectorEngineConfig {
    pub fn validate(&self) -> Result<()> {
        for field in &self.default_fields {
            if !self.fields.contains_key(field) {
                return Err(PlatypusError::invalid_config(format!(
                    "default field '{field}' is not defined"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldConfig {
    pub dimension: usize,
    pub distance: DistanceMetric,
    pub index: VectorIndexKind,
    pub embedder_id: String,
    pub role: VectorRole,
    #[serde(default = "VectorFieldConfig::default_weight")]
    pub base_weight: f32,
}

impl VectorFieldConfig {
    fn default_weight() -> f32 {
        1.0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndexKind {
    Flat,
    Hnsw,
    Ivf,
}

/// Request model for collection-level search.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEngineSearchRequest {
    #[serde(default)]
    pub query_vectors: Vec<QueryVector>,
    #[serde(default)]
    pub fields: Option<Vec<FieldSelector>>,
    #[serde(default = "default_query_limit")]
    pub limit: usize,
    #[serde(default)]
    pub score_mode: VectorScoreMode,
    #[serde(default = "default_overfetch")]
    pub overfetch: f32,
    #[serde(default)]
    pub filter: Option<VectorEngineFilter>,
}

fn default_query_limit() -> usize {
    10
}

fn default_overfetch() -> f32 {
    1.0
}

impl Default for VectorEngineSearchRequest {
    fn default() -> Self {
        Self {
            query_vectors: Vec::new(),
            fields: None,
            limit: default_query_limit(),
            score_mode: VectorScoreMode::default(),
            overfetch: default_overfetch(),
            filter: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum FieldSelector {
    Exact(String),
    Prefix(String),
    Role(VectorRole),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorScoreMode {
    WeightedSum,
    MaxSim,
    LateInteraction,
}

impl Default for VectorScoreMode {
    fn default() -> Self {
        VectorScoreMode::WeightedSum
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetadataFilter {
    #[serde(default)]
    pub equals: HashMap<String, String>,
}

impl MetadataFilter {
    fn matches(&self, metadata: &HashMap<String, String>) -> bool {
        self.equals.iter().all(|(key, expected)| {
            metadata
                .get(key)
                .map(|actual| actual == expected)
                .unwrap_or(false)
        })
    }

    fn is_empty(&self) -> bool {
        self.equals.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorEngineFilter {
    #[serde(default)]
    pub document: MetadataFilter,
    #[serde(default)]
    pub field: MetadataFilter,
}

impl VectorEngineFilter {
    fn is_empty(&self) -> bool {
        self.document.is_empty() && self.field.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryVector {
    pub vector: StoredVector,
    #[serde(default = "QueryVector::default_weight")]
    pub weight: f32,
}

impl QueryVector {
    fn default_weight() -> f32 {
        1.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorEngineSearchResults {
    #[serde(default)]
    pub hits: Vec<VectorEngineHit>,
}

/// Aggregated statistics describing a collection and its fields.
#[derive(Debug, Clone, Default)]
pub struct VectorEngineStats {
    pub document_count: usize,
    pub fields: HashMap<String, VectorFieldStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEngineHit {
    pub doc_id: u64,
    pub score: f32,
    #[serde(default)]
    pub field_hits: Vec<FieldHit>,
}

#[derive(Debug, Default)]
pub struct RegistryFilterMatches {
    allowed_fields: HashMap<u64, HashSet<String>>,
}

impl RegistryFilterMatches {
    fn is_empty(&self) -> bool {
        self.allowed_fields.is_empty()
    }

    fn contains_doc(&self, doc_id: u64) -> bool {
        self.allowed_fields.contains_key(&doc_id)
    }

    fn field_allowed(&self, doc_id: u64, field: &str) -> bool {
        self.allowed_fields
            .get(&doc_id)
            .map(|fields| fields.contains(field))
            .unwrap_or(false)
    }
}

pub type RegistryVersion = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldEntry {
    pub field_name: String,
    pub version: RegistryVersion,
    pub vector_count: usize,
    pub weight: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentEntry {
    pub doc_id: u64,
    pub version: RegistryVersion,
    pub metadata: HashMap<String, String>,
    pub fields: HashMap<String, FieldEntry>,
}

#[derive(Debug, Default)]
pub struct DocumentVectorRegistry {
    entries: RwLock<HashMap<u64, DocumentEntry>>,
    next_version: AtomicU64,
}

impl DocumentVectorRegistry {
    pub fn upsert(
        &self,
        doc_id: u64,
        fields: &[FieldEntry],
        metadata: HashMap<String, String>,
    ) -> Result<RegistryVersion> {
        let version = self.next_version.fetch_add(1, Ordering::SeqCst) + 1;
        let mut map = HashMap::new();
        for entry in fields {
            let mut cloned = entry.clone();
            cloned.version = version;
            map.insert(cloned.field_name.clone(), cloned);
        }

        let doc_entry = DocumentEntry {
            doc_id,
            version,
            metadata,
            fields: map,
        };

        self.entries.write().insert(doc_id, doc_entry);
        Ok(version)
    }

    pub fn delete(&self, doc_id: u64) -> Result<()> {
        let mut guard = self.entries.write();
        guard
            .remove(&doc_id)
            .ok_or_else(|| PlatypusError::not_found(format!("doc_id {doc_id}")))?;
        Ok(())
    }

    pub fn get(&self, doc_id: u64) -> Option<DocumentEntry> {
        self.entries.read().get(&doc_id).cloned()
    }

    pub fn snapshot(&self) -> Result<Vec<u8>> {
        let guard = self.entries.read();
        serde_json::to_vec(&*guard).map_err(PlatypusError::from)
    }

    pub fn from_snapshot(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }

        let entries: HashMap<u64, DocumentEntry> = serde_json::from_slice(bytes)?;
        let max_version = entries
            .values()
            .map(|entry| entry.version)
            .max()
            .unwrap_or(0);
        Ok(Self {
            entries: RwLock::new(entries),
            next_version: AtomicU64::new(max_version),
        })
    }

    pub fn document_count(&self) -> usize {
        self.entries.read().len()
    }

    pub fn filter_matches(
        &self,
        filter: &VectorEngineFilter,
        target_fields: &[String],
    ) -> RegistryFilterMatches {
        let guard = self.entries.read();
        let mut allowed_fields: HashMap<u64, HashSet<String>> = HashMap::new();

        for entry in guard.values() {
            if !filter.document.is_empty() && !filter.document.matches(&entry.metadata) {
                continue;
            }

            let mut matched_fields: HashSet<String> = HashSet::new();
            for field_name in target_fields {
                if let Some(field_entry) = entry.fields.get(field_name) {
                    if filter.field.is_empty() || filter.field.matches(&field_entry.metadata) {
                        matched_fields.insert(field_name.clone());
                    }
                }
            }

            if matched_fields.is_empty() {
                continue;
            }

            allowed_fields.insert(entry.doc_id, matched_fields);
        }

        RegistryFilterMatches { allowed_fields }
    }
}

#[derive(Debug, Default)]
pub struct VectorWal {
    records: Mutex<Vec<WalRecord>>,
    next_seq: AtomicU64,
}

impl VectorWal {
    pub fn from_records(records: Vec<WalRecord>) -> Self {
        let max_seq = records.iter().map(|record| record.seq).max().unwrap_or(0);
        Self {
            records: Mutex::new(records),
            next_seq: AtomicU64::new(max_seq),
        }
    }

    pub fn records(&self) -> Vec<WalRecord> {
        self.records.lock().clone()
    }

    pub fn len(&self) -> usize {
        self.records.lock().len()
    }

    pub fn last_seq(&self) -> SeqNumber {
        self.next_seq.load(Ordering::SeqCst)
    }

    pub fn append(&self, payload: WalPayload) -> Result<SeqNumber> {
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst) + 1;
        let record = WalRecord { seq, payload };
        self.records.lock().push(record);
        Ok(seq)
    }

    pub fn replay<F: FnMut(&WalRecord)>(&self, from: SeqNumber, mut handler: F) {
        for record in self.records.lock().iter() {
            if record.seq >= from {
                handler(record);
            }
        }
    }

    pub fn replace_records(&self, records: Vec<WalRecord>) {
        let max_seq = records.iter().map(|record| record.seq).max().unwrap_or(0);
        let mut guard = self.records.lock();
        *guard = records;
        self.next_seq.store(max_seq, Ordering::SeqCst);
    }
}

pub type SeqNumber = u64;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DocumentSnapshot {
    #[serde(default)]
    last_wal_seq: SeqNumber,
    #[serde(default)]
    documents: Vec<DocumentVectors>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectionManifest {
    version: u32,
    snapshot_wal_seq: SeqNumber,
    wal_last_seq: SeqNumber,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    pub seq: SeqNumber,
    pub payload: WalPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalPayload {
    Upsert { document: DocumentVectors },
    Delete { doc_id: u64 },
}

#[derive(Debug)]
struct InMemoryVectorField {
    name: String,
    config: VectorFieldConfig,
    store: Arc<FieldStore>,
    writer: Arc<InMemoryFieldWriter>,
    reader: Arc<InMemoryFieldReader>,
}

impl InMemoryVectorField {
    fn new(
        name: String,
        config: VectorFieldConfig,
        delegate: Option<Arc<dyn VectorFieldWriter>>,
    ) -> Result<Self> {
        let store = Arc::new(FieldStore::default());
        let writer = Arc::new(InMemoryFieldWriter::new(
            name.clone(),
            config.clone(),
            store.clone(),
            delegate,
        ));
        let reader = Arc::new(InMemoryFieldReader::new(
            name.clone(),
            config.clone(),
            store.clone(),
        ));
        Ok(Self {
            name,
            config,
            store,
            writer,
            reader,
        })
    }

    fn vector_tuples(&self) -> Vec<(u64, String, Vector)> {
        self.store.vector_tuples(&self.name)
    }
}

impl VectorField for InMemoryVectorField {
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

#[derive(Debug)]
struct InMemoryFieldWriter {
    field_name: String,
    config: VectorFieldConfig,
    store: Arc<FieldStore>,
    delegate: Option<Arc<dyn VectorFieldWriter>>,
}

impl InMemoryFieldWriter {
    fn new(
        field_name: String,
        config: VectorFieldConfig,
        store: Arc<FieldStore>,
        delegate: Option<Arc<dyn VectorFieldWriter>>,
    ) -> Self {
        Self {
            field_name,
            config,
            store,
            delegate,
        }
    }

    fn convert_vectors(&self, field: &FieldVectors) -> Result<Vec<Vector>> {
        let mut vectors = Vec::with_capacity(field.vector_count());
        for stored in &field.vectors {
            let vector = stored.to_vector();
            if vector.dimension() != self.config.dimension {
                return Err(PlatypusError::invalid_argument(format!(
                    "vector dimension mismatch for field '{}': expected {}, got {}",
                    self.field_name,
                    self.config.dimension,
                    vector.dimension()
                )));
            }
            if !vector.is_valid() {
                return Err(PlatypusError::invalid_argument(format!(
                    "vector for field '{}' contains invalid values",
                    self.field_name
                )));
            }
            vectors.push(vector);
        }
        Ok(vectors)
    }
}

impl VectorFieldWriter for InMemoryFieldWriter {
    fn add_field_vectors(&self, doc_id: u64, field: &FieldVectors, version: u64) -> Result<()> {
        if field.vectors.is_empty() {
            self.store.remove(doc_id);
            if let Some(delegate) = &self.delegate {
                delegate.delete_document(doc_id, version)?;
            }
            return Ok(());
        }

        if let Some(delegate) = &self.delegate {
            delegate.add_field_vectors(doc_id, field, version)?;
        }

        let vectors = self.convert_vectors(field)?;
        self.store.replace(doc_id, FieldStoreEntry { vectors });
        Ok(())
    }

    fn delete_document(&self, doc_id: u64, version: u64) -> Result<()> {
        self.store.remove(doc_id);
        if let Some(delegate) = &self.delegate {
            delegate.delete_document(doc_id, version)?;
        }
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        if let Some(delegate) = &self.delegate {
            delegate.flush()?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct InMemoryFieldReader {
    field_name: String,
    config: VectorFieldConfig,
    store: Arc<FieldStore>,
}

impl InMemoryFieldReader {
    fn new(field_name: String, config: VectorFieldConfig, store: Arc<FieldStore>) -> Self {
        Self {
            field_name,
            config,
            store,
        }
    }
}

impl VectorFieldReader for InMemoryFieldReader {
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

        let snapshot = self.store.snapshot();
        let mut merged: HashMap<u64, FieldHit> = HashMap::new();

        for query in &request.query_vectors {
            let query_vector = query.vector.to_vector();
            if query_vector.dimension() != self.config.dimension {
                return Err(PlatypusError::invalid_argument(format!(
                    "query vector dimension mismatch for field '{}': expected {}, got {}",
                    self.field_name,
                    self.config.dimension,
                    query_vector.dimension()
                )));
            }
            let effective_weight = query.weight * query.vector.weight;
            if effective_weight == 0.0 {
                continue;
            }

            for (doc_id, entry) in &snapshot {
                for vector in &entry.vectors {
                    let similarity = self
                        .config
                        .distance
                        .similarity(&query_vector.data, &vector.data)?;
                    let distance = self
                        .config
                        .distance
                        .distance(&query_vector.data, &vector.data)?;
                    let weighted_score = similarity * effective_weight;
                    if weighted_score == 0.0 {
                        continue;
                    }

                    match merged.entry(*doc_id) {
                        Entry::Vacant(slot) => {
                            slot.insert(FieldHit {
                                doc_id: *doc_id,
                                field: self.field_name.clone(),
                                score: weighted_score,
                                distance,
                                metadata: vector.metadata.clone(),
                            });
                        }
                        Entry::Occupied(mut slot) => {
                            let hit = slot.get_mut();
                            hit.score += weighted_score;
                            hit.distance = hit.distance.min(distance);
                            if hit.metadata.is_empty() {
                                hit.metadata = vector.metadata.clone();
                            }
                        }
                    }
                }
            }
        }

        let mut hits: Vec<FieldHit> = merged.into_values().collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(CmpOrdering::Equal));
        if hits.len() > request.limit {
            hits.truncate(request.limit);
        }

        Ok(FieldSearchResults { hits })
    }

    fn stats(&self) -> Result<VectorFieldStats> {
        Ok(VectorFieldStats {
            vector_count: self.store.total_vectors(),
            dimension: self.config.dimension,
        })
    }
}

#[derive(Debug, Default)]
struct FieldStore {
    entries: RwLock<HashMap<u64, FieldStoreEntry>>,
}

impl FieldStore {
    fn replace(&self, doc_id: u64, entry: FieldStoreEntry) {
        self.entries.write().insert(doc_id, entry);
    }

    fn remove(&self, doc_id: u64) {
        self.entries.write().remove(&doc_id);
    }

    fn snapshot(&self) -> HashMap<u64, FieldStoreEntry> {
        self.entries.read().clone()
    }

    fn total_vectors(&self) -> usize {
        self.entries
            .read()
            .values()
            .map(|entry| entry.vectors.len())
            .sum()
    }

    fn vector_tuples(&self, field_name: &str) -> Vec<(u64, String, Vector)> {
        let guard = self.entries.read();
        let mut tuples = Vec::new();
        let name = field_name.to_string();
        for (doc_id, entry) in guard.iter() {
            for vector in &entry.vectors {
                tuples.push((*doc_id, name.clone(), vector.clone()));
            }
        }
        tuples
    }
}

#[derive(Debug, Clone)]
struct FieldStoreEntry {
    vectors: Vec<Vector>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::storage::prefixed::PrefixedStorage;
    use crate::vector::core::document::{DocumentVectors, FieldVectors, StoredVector, VectorRole};
    use crate::vector::field::{
        FieldSearchResults, VectorFieldReader, VectorFieldStats, VectorFieldWriter,
    };
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    #[derive(Debug)]
    struct MockFieldReader {
        hits: Vec<FieldHit>,
    }

    impl VectorFieldReader for MockFieldReader {
        fn search(&self, _request: FieldSearchInput) -> Result<FieldSearchResults> {
            Ok(FieldSearchResults {
                hits: self.hits.clone(),
            })
        }

        fn stats(&self) -> Result<VectorFieldStats> {
            Ok(VectorFieldStats {
                vector_count: self.hits.len(),
                dimension: 3,
            })
        }
    }

    #[derive(Debug, Default)]
    struct RecordingFieldWriter {
        additions: Arc<Mutex<Vec<(u64, usize, u64)>>>,
        deletions: Arc<Mutex<Vec<(u64, u64)>>>,
    }

    impl RecordingFieldWriter {
        fn new() -> Self {
            Self::default()
        }

        fn additions(&self) -> Arc<Mutex<Vec<(u64, usize, u64)>>> {
            Arc::clone(&self.additions)
        }

        fn deletions(&self) -> Arc<Mutex<Vec<(u64, u64)>>> {
            Arc::clone(&self.deletions)
        }
    }

    impl VectorFieldWriter for RecordingFieldWriter {
        fn add_field_vectors(&self, doc_id: u64, field: &FieldVectors, version: u64) -> Result<()> {
            self.additions
                .lock()
                .unwrap()
                .push((doc_id, field.vector_count(), version));
            Ok(())
        }

        fn delete_document(&self, doc_id: u64, version: u64) -> Result<()> {
            self.deletions.lock().unwrap().push((doc_id, version));
            Ok(())
        }

        fn flush(&self) -> Result<()> {
            Ok(())
        }
    }

    fn collection_with_field(
        hits: Vec<FieldHit>,
        weight: f32,
    ) -> (VectorEngine, Arc<RecordingFieldWriter>) {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            role: VectorRole::Text,
            base_weight: weight,
        };
        let config = VectorEngineConfig {
            fields: HashMap::new(),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default())) as Arc<dyn Storage>;
        let mut collection = VectorEngine::new(config, storage, None).expect("collection");
        let writer = Arc::new(RecordingFieldWriter::new());
        let writer_trait: Arc<dyn VectorFieldWriter> = writer.clone();
        let reader: Arc<dyn VectorFieldReader> = Arc::new(MockFieldReader { hits });
        collection
            .register_adapter_field("body", field_config, writer_trait, reader)
            .expect("register");
        (collection, writer)
    }

    fn field_vectors_with_metadata(metadata: Option<(&str, &str)>) -> FieldVectors {
        let mut vectors = FieldVectors::default();
        vectors.vectors.push(StoredVector::new(
            Arc::<[f32]>::from([1.0, 0.0, 0.0]),
            "mock".into(),
            VectorRole::Text,
        ));
        if let Some((key, value)) = metadata {
            vectors.metadata.insert(key.to_string(), value.to_string());
        }
        vectors
    }

    fn sample_query(limit: usize) -> VectorEngineSearchRequest {
        let mut query = VectorEngineSearchRequest::default();
        query.limit = limit;
        query.query_vectors.push(QueryVector {
            vector: StoredVector::new(
                Arc::<[f32]>::from([1.0, 0.0, 0.0]),
                "mock".into(),
                VectorRole::Text,
            ),
            weight: 1.0,
        });
        query
    }

    #[test]
    fn search_combines_field_scores() {
        let hits = vec![
            FieldHit {
                doc_id: 1,
                field: "body".into(),
                score: 0.8,
                distance: 0.2,
                metadata: HashMap::new(),
            },
            FieldHit {
                doc_id: 2,
                field: "body".into(),
                score: 0.5,
                distance: 0.5,
                metadata: HashMap::new(),
            },
        ];
        let (collection, _) = collection_with_field(hits, 2.0);
        let results = collection.search(&sample_query(5)).expect("search");
        assert_eq!(results.hits.len(), 2);
        assert_eq!(results.hits[0].doc_id, 1);
        assert!((results.hits[0].score - 1.6).abs() < f32::EPSILON);
    }

    #[test]
    fn search_filters_by_document_metadata() {
        let hits = vec![
            FieldHit {
                doc_id: 10,
                field: "body".into(),
                score: 0.9,
                distance: 0.1,
                metadata: HashMap::new(),
            },
            FieldHit {
                doc_id: 11,
                field: "body".into(),
                score: 0.8,
                distance: 0.2,
                metadata: HashMap::new(),
            },
        ];
        let (collection, _) = collection_with_field(hits, 1.0);

        let mut ja_doc = DocumentVectors::new(10);
        ja_doc.metadata.insert("lang".into(), "ja".into());
        ja_doc.add_field("body", field_vectors_with_metadata(None));
        collection.upsert_document(ja_doc).expect("upsert ja");

        let mut en_doc = DocumentVectors::new(11);
        en_doc.metadata.insert("lang".into(), "en".into());
        en_doc.add_field("body", field_vectors_with_metadata(None));
        collection.upsert_document(en_doc).expect("upsert en");

        let mut query = sample_query(5);
        query.filter = Some(VectorEngineFilter {
            document: MetadataFilter {
                equals: HashMap::from([(String::from("lang"), String::from("ja"))]),
            },
            ..Default::default()
        });

        let results = collection.search(&query).expect("search filtered");
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, 10);

        query.filter = Some(VectorEngineFilter {
            document: MetadataFilter {
                equals: HashMap::from([(String::from("lang"), String::from("fr"))]),
            },
            ..Default::default()
        });
        let empty = collection.search(&query).expect("no match");
        assert!(empty.hits.is_empty());
    }

    #[test]
    fn search_filters_by_field_metadata() {
        let hits = vec![
            FieldHit {
                doc_id: 1,
                field: "body".into(),
                score: 0.9,
                distance: 0.1,
                metadata: HashMap::new(),
            },
            FieldHit {
                doc_id: 2,
                field: "body".into(),
                score: 0.7,
                distance: 0.3,
                metadata: HashMap::new(),
            },
        ];
        let (collection, _) = collection_with_field(hits, 1.0);

        let mut title_doc = DocumentVectors::new(1);
        title_doc.add_field(
            "body",
            field_vectors_with_metadata(Some(("section", "title"))),
        );
        collection
            .upsert_document(title_doc)
            .expect("upsert title doc");

        let mut body_doc = DocumentVectors::new(2);
        body_doc.add_field(
            "body",
            field_vectors_with_metadata(Some(("section", "body"))),
        );
        collection
            .upsert_document(body_doc)
            .expect("upsert body doc");

        let mut query = sample_query(5);
        query.filter = Some(VectorEngineFilter {
            field: MetadataFilter {
                equals: HashMap::from([(String::from("section"), String::from("title"))]),
            },
            ..Default::default()
        });

        let results = collection.search(&query).expect("search filtered");
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, 1);
    }

    #[test]
    fn search_supports_role_selector() {
        let hits = vec![FieldHit {
            doc_id: 3,
            field: "body".into(),
            score: 0.9,
            distance: 0.1,
            metadata: HashMap::new(),
        }];
        let (collection, _) = collection_with_field(hits, 1.0);
        let mut query = sample_query(5);
        query.fields = Some(vec![FieldSelector::Role(VectorRole::Text)]);
        let results = collection.search(&query).expect("search");
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, 3);
    }

    #[test]
    fn search_errors_when_vectors_do_not_match_fields() {
        let (collection, _) = collection_with_field(vec![], 1.0);
        let mut query = VectorEngineSearchRequest::default();
        query.limit = 5;
        query.query_vectors.push(QueryVector {
            vector: StoredVector::new(
                Arc::<[f32]>::from([1.0, 0.0, 0.0]),
                "other".into(),
                VectorRole::Text,
            ),
            weight: 1.0,
        });

        let err = collection.search(&query).expect_err("should fail");
        assert!(
            err.to_string()
                .contains("no query vectors matched the requested fields")
        );
    }

    #[test]
    fn score_mode_max_sim_uses_highest_score() {
        let hits = vec![
            FieldHit {
                doc_id: 1,
                field: "body".into(),
                score: 0.2,
                distance: 0.8,
                metadata: HashMap::new(),
            },
            FieldHit {
                doc_id: 1,
                field: "body".into(),
                score: 0.9,
                distance: 0.1,
                metadata: HashMap::new(),
            },
        ];
        let (collection, _) = collection_with_field(hits, 2.0);
        let mut query = sample_query(5);
        query.score_mode = VectorScoreMode::MaxSim;
        let results = collection.search(&query).expect("search");
        assert_eq!(results.hits.len(), 1);
        assert!((results.hits[0].score - 1.8).abs() < f32::EPSILON);
    }

    #[test]
    fn search_errors_when_field_missing() {
        let hits = vec![];
        let (collection, _) = collection_with_field(hits, 1.0);
        let mut query = sample_query(5);
        query.fields = Some(vec![FieldSelector::Exact("unknown".into())]);
        let err = collection.search(&query).expect_err("should fail");
        assert!(matches!(err, PlatypusError::Other(_)));
        assert!(err.to_string().contains("vector field"));
    }

    #[test]
    fn collection_reports_stats() {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            role: VectorRole::Text,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default())) as Arc<dyn Storage>;
        let collection = VectorEngine::new(config, storage, None).expect("collection");
        let initial = collection.stats().expect("stats");
        assert_eq!(initial.document_count, 0);
        let body_stats = initial.fields.get("body").expect("body stats");
        assert_eq!(body_stats.vector_count, 0);
        assert_eq!(body_stats.dimension, 3);

        let mut doc = DocumentVectors::new(5);
        let mut vectors = FieldVectors::default();
        vectors.vectors.push(StoredVector::new(
            Arc::<[f32]>::from([0.5, 0.1, 0.3]),
            "mock".into(),
            VectorRole::Text,
        ));
        doc.add_field("body", vectors);
        collection.upsert_document(doc).expect("upsert");

        let updated = collection.stats().expect("stats");
        assert_eq!(updated.document_count, 1);
        let body_stats = updated.fields.get("body").expect("body stats");
        assert_eq!(body_stats.vector_count, 1);
    }

    #[test]
    fn materialize_delegate_reader_builds_persistent_index() {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            role: VectorRole::Text,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let collection = VectorEngine::new(config, storage.clone(), None).expect("collection");

        let mut doc = DocumentVectors::new(1);
        let mut vectors = FieldVectors::default();
        vectors.vectors.push(StoredVector::new(
            Arc::<[f32]>::from([1.0, 0.0, 0.0]),
            "mock".into(),
            VectorRole::Text,
        ));
        doc.add_field("body", vectors);
        collection.upsert_document(doc).expect("upsert");

        collection
            .materialize_delegate_reader("body")
            .expect("materialize");

        let files = storage.list_files().expect("list files");
        assert!(
            files
                .iter()
                .any(|name| name.ends_with("vector_fields/body/index.flat"))
        );

        let results = collection.search(&sample_query(5)).expect("search");
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, 1);
    }

    #[test]
    fn registry_snapshot_persists_across_instances() {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            role: VectorRole::Text,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let collection =
                VectorEngine::new(config.clone(), storage.clone(), None).expect("collection");
            let mut doc = DocumentVectors::new(10);
            let mut vectors = FieldVectors::default();
            vectors.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([0.2, 0.3, 0.4]),
                "mock".into(),
                VectorRole::Text,
            ));
            doc.add_field("body", vectors);
            collection.upsert_document(doc).expect("upsert");
        }

        let files = storage.list_files().expect("list files");
        assert!(
            files
                .iter()
                .any(|name| name.ends_with("vector_registry/registry.json"))
        );
        assert!(
            files
                .iter()
                .any(|name| name.ends_with("vector_registry/wal.json"))
        );

        let collection = VectorEngine::new(config, storage.clone(), None).expect("collection");
        let stats = collection.stats().expect("stats");
        assert_eq!(stats.document_count, 1);
        let results = collection
            .search(&sample_query(5))
            .expect("search after reload");
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, 10);
    }

    #[test]
    fn wal_replay_restores_live_documents_only() {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            role: VectorRole::Text,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let collection =
                VectorEngine::new(config.clone(), storage.clone(), None).expect("collection");
            let mut doc_one = DocumentVectors::new(1);
            let mut vectors_one = FieldVectors::default();
            vectors_one.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([1.0, 0.0, 0.0]),
                "mock".into(),
                VectorRole::Text,
            ));
            doc_one.add_field("body", vectors_one);
            collection.upsert_document(doc_one).expect("upsert doc one");

            let mut doc_two = DocumentVectors::new(2);
            let mut vectors_two = FieldVectors::default();
            vectors_two.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([0.8, 0.0, 0.2]),
                "mock".into(),
                VectorRole::Text,
            ));
            doc_two.add_field("body", vectors_two);
            collection.upsert_document(doc_two).expect("upsert doc two");

            collection.delete_document(1).expect("delete doc one");
        }

        let collection = VectorEngine::new(config, storage.clone(), None).expect("collection");
        let stats = collection.stats().expect("stats");
        assert_eq!(stats.document_count, 1);
        let results = collection
            .search(&sample_query(5))
            .expect("search after replay");
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, 2);
    }

    #[test]
    fn wal_compaction_drops_tombstones() {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            role: VectorRole::Text,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let collection = VectorEngine::new(config, storage, None).expect("collection");

        for doc_id in 0..6_u64 {
            let mut doc = DocumentVectors::new(doc_id);
            let mut vectors = FieldVectors::default();
            vectors.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([1.0, 0.0, 0.0]),
                "mock".into(),
                VectorRole::Text,
            ));
            doc.add_field("body", vectors);
            collection.upsert_document(doc).expect("upsert");
        }

        collection.delete_document(0).expect("delete doc zero");

        let records = collection.wal.records();
        assert_eq!(records.len(), 5);
        for (expected_seq, record) in records.iter().enumerate() {
            assert_eq!(record.seq, (expected_seq as u64) + 1);
            match &record.payload {
                WalPayload::Upsert { document } => {
                    assert_ne!(document.doc_id, 0);
                }
                _ => panic!("compacted WAL should only contain upserts"),
            }
        }
    }

    #[test]
    fn document_snapshot_restores_without_wal() {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            role: VectorRole::Text,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let collection =
                VectorEngine::new(config.clone(), storage.clone(), None).expect("collection");
            let mut doc = DocumentVectors::new(77);
            let mut vectors = FieldVectors::default();
            vectors.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([0.4, 0.5, 0.6]),
                "mock".into(),
                VectorRole::Text,
            ));
            doc.add_field("body", vectors);
            collection.upsert_document(doc).expect("upsert");
        }

        storage
            .delete_file("vector_registry/wal.json")
            .expect("delete wal");

        let collection = VectorEngine::new(config, storage.clone(), None).expect("collection");
        let stats = collection.stats().expect("stats");
        assert_eq!(stats.document_count, 1);
        let results = collection
            .search(&sample_query(5))
            .expect("search after snapshot");
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, 77);
    }

    #[test]
    fn manifest_mismatch_returns_error() {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            role: VectorRole::Text,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let collection =
                VectorEngine::new(config.clone(), storage.clone(), None).expect("collection");
            let mut doc = DocumentVectors::new(1);
            let mut vectors = FieldVectors::default();
            vectors.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([1.0, 0.0, 0.0]),
                "mock".into(),
                VectorRole::Text,
            ));
            doc.add_field("body", vectors);
            collection.upsert_document(doc).expect("upsert");
        }

        let registry_storage: Arc<dyn Storage> =
            Arc::new(PrefixedStorage::new(REGISTRY_NAMESPACE, storage.clone()));
        let manifest = CollectionManifest {
            version: COLLECTION_MANIFEST_VERSION,
            snapshot_wal_seq: 9999,
            wal_last_seq: 9999,
        };
        let serialized = serde_json::to_vec(&manifest).expect("serialize manifest");
        let mut output = registry_storage
            .create_output(COLLECTION_MANIFEST_FILE)
            .expect("create manifest");
        output.write_all(&serialized).expect("write manifest");
        output.flush_and_sync().expect("flush manifest");
        output.close().expect("close manifest");

        let err = VectorEngine::new(config, storage.clone(), None).expect_err("should fail");
        assert!(
            err.to_string()
                .contains("collection manifest snapshot sequence")
        );
    }

    #[test]
    fn replace_and_reset_field_reader() {
        let base_hits = vec![FieldHit {
            doc_id: 1,
            field: "body".into(),
            score: 0.8,
            distance: 0.2,
            metadata: HashMap::new(),
        }];
        let replacement_hits = vec![FieldHit {
            doc_id: 42,
            field: "body".into(),
            score: 0.9,
            distance: 0.1,
            metadata: HashMap::new(),
        }];

        let (collection, _) = collection_with_field(base_hits.clone(), 1.0);
        let replacement_reader: Arc<dyn VectorFieldReader> = Arc::new(MockFieldReader {
            hits: replacement_hits.clone(),
        });

        collection
            .replace_field_reader("body", replacement_reader)
            .expect("replace");

        let query = sample_query(5);
        let replaced_results = collection.search(&query).expect("search after replace");
        assert_eq!(replaced_results.hits.len(), 1);
        assert_eq!(replaced_results.hits[0].doc_id, 42);

        collection.reset_field_reader("body").expect("reset");
        let restored_results = collection.search(&query).expect("search after reset");
        assert_eq!(restored_results.hits.len(), 1);
        assert_eq!(restored_results.hits[0].doc_id, 1);
    }

    #[test]
    fn upsert_routes_vectors_to_field_writer() {
        let (collection, writer) = collection_with_field(vec![], 1.0);
        let mut doc = DocumentVectors::new(99);
        let mut vectors = FieldVectors::default();
        vectors.vectors.push(StoredVector::new(
            Arc::<[f32]>::from([0.1, 0.2, 0.3]),
            "mock".into(),
            VectorRole::Text,
        ));
        doc.add_field("body", vectors);

        let version = collection.upsert_document(doc).expect("upsert");
        let additions = writer.additions().lock().unwrap().clone();
        assert_eq!(additions.len(), 1);
        assert_eq!(additions[0].0, 99);
        assert_eq!(additions[0].1, 1);
        assert_eq!(additions[0].2, version);
    }

    #[test]
    fn delete_invokes_field_writer() {
        let (collection, writer) = collection_with_field(vec![], 1.0);
        let mut doc = DocumentVectors::new(7);
        doc.add_field("body", FieldVectors::default());
        collection.upsert_document(doc).expect("upsert");

        collection.delete_document(7).expect("delete");
        let deletions = writer.deletions().lock().unwrap().clone();
        assert_eq!(deletions.len(), 1);
        assert_eq!(deletions[0].0, 7);
    }
}
