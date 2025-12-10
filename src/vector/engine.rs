//! VectorEngine: ドキュメント中心のベクトルコレクション実装
//!
//! このモジュールは複数のベクトルフィールドを管理する高レベルのコレクションエンジンを提供する。
//!
//! # モジュール構成
//!
//! - [`config`] - 設定型（VectorEngineConfig, VectorFieldConfig, VectorIndexKind）
//! - [`embedder`] - 埋め込みレジストリとエグゼキュータ
//! - [`filter`] - メタデータフィルタリング
//! - [`memory`] - インメモリフィールド実装
//! - [`registry`] - ドキュメントベクトルレジストリ
//! - [`request`] - 検索リクエスト型
//! - [`response`] - 検索レスポンス型
//! - [`snapshot`] - スナップショット永続化
//! - [`wal`] - Write-Ahead Logging
//!
//! # 使用例
//!
//! ```ignore
//! use platypus::vector::engine::{VectorEngine, VectorEngineConfig, VectorFieldConfig};
//!
//! let config = VectorEngineConfig::default();
//! let engine = VectorEngine::new(config, storage, None)?;
//! ```

pub mod config;
pub mod embedder;
pub mod filter;
pub mod memory;
pub mod registry;
pub mod request;
pub mod response;
pub mod snapshot;
pub mod wal;

use std::cmp::Ordering as CmpOrdering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::embedding::image_embedder::ImageEmbedder;
use crate::embedding::text_embedder::TextEmbedder;
use crate::error::{PlatypusError, Result};
use crate::storage::Storage;
use crate::storage::prefixed::PrefixedStorage;
use crate::vector::core::document::{
    DocumentPayload, DocumentVector, FieldPayload, FieldVectors, PayloadSource, SegmentPayload,
    StoredVector, VectorType,
};
use crate::vector::core::vector::Vector;
use crate::vector::field::{
    FieldHit, FieldSearchInput, VectorField, VectorFieldReader, VectorFieldStats,
    VectorFieldWriter,
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

#[cfg(feature = "embeddings-multimodal")]
use crate::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
#[cfg(feature = "embeddings-candle")]
use crate::embedding::candle_text_embedder::CandleTextEmbedder;
#[cfg(feature = "embeddings-openai")]
use crate::embedding::openai_text_embedder::OpenAITextEmbedder;

pub use config::{VectorEmbedderConfig, VectorEmbedderProvider, VectorEngineConfig, VectorFieldConfig, VectorIndexKind};
pub use filter::{MetadataFilter, VectorEngineFilter};
pub use registry::{DocumentVectorRegistry, RegistryVersion};
pub use request::{FieldSelector, QueryVector, VectorEngineSearchRequest, VectorScoreMode};
pub use response::{VectorEngineHit, VectorEngineSearchResults, VectorEngineStats};

use embedder::{EmbedderExecutor, VectorEmbedderRegistry};
use filter::RegistryFilterMatches;
use memory::{FieldHandle, FieldRuntime, InMemoryVectorField};
use registry::{DocumentEntry, FieldEntry};
use snapshot::{
    CollectionManifest, DocumentSnapshot, SnapshotDocument, COLLECTION_MANIFEST_FILE,
    COLLECTION_MANIFEST_VERSION, DOCUMENT_SNAPSHOT_FILE, DOCUMENT_SNAPSHOT_TEMP_FILE,
    FIELD_INDEX_BASENAME, REGISTRY_NAMESPACE, REGISTRY_SNAPSHOT_FILE, REGISTRY_WAL_FILE,
};
use wal::{VectorWal, WalPayload, WalRecord, WAL_COMPACTION_THRESHOLD};

/// High-level collection combining multiple vector fields.
pub struct VectorEngine {
    config: Arc<VectorEngineConfig>,
    fields: HashMap<String, FieldHandle>,
    registry: Arc<DocumentVectorRegistry>,
    embedder_registry: Arc<VectorEmbedderRegistry>,
    embedder_executor: Mutex<Option<Arc<EmbedderExecutor>>>,
    wal: Arc<VectorWal>,
    storage: Arc<dyn Storage>,
    documents: Arc<RwLock<HashMap<u64, DocumentVector>>>,
    snapshot_wal_seq: AtomicU64,
    next_doc_id: AtomicU64,
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
        let embedder_registry = Arc::new(VectorEmbedderRegistry::new(config.embedders.clone()));
        let should_load_state = registry.is_none();
        let registry = registry.unwrap_or_else(|| Arc::new(DocumentVectorRegistry::default()));
        let mut collection = Self {
            config: Arc::new(config),
            fields: HashMap::new(),
            registry,
            embedder_registry,
            embedder_executor: Mutex::new(None),
            wal: Arc::new(VectorWal::default()),
            storage,
            documents: Arc::new(RwLock::new(HashMap::new())),
            snapshot_wal_seq: AtomicU64::new(0),
            next_doc_id: AtomicU64::new(0),
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

    /// Register an externally constructed embedder instance for `VectorEmbedderProvider::External`.
    pub fn register_embedder_instance(
        &self,
        embedder_id: impl Into<String>,
        embedder: Arc<dyn TextEmbedder>,
    ) -> Result<()> {
        self.embedder_registry
            .register_external(embedder_id.into(), embedder)
    }

    /// Register an externally constructed embedder that supports both text and image inputs.
    pub fn register_multimodal_embedder_instance(
        &self,
        embedder_id: impl Into<String>,
        text_embedder: Arc<dyn TextEmbedder>,
        image_embedder: Arc<dyn ImageEmbedder>,
    ) -> Result<()> {
        self.embedder_registry.register_external_with_image(
            embedder_id.into(),
            text_embedder,
            Some(image_embedder),
        )
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

    /// Add a document with automatically assigned doc_id.
    pub fn add_document(&self, document: DocumentVector) -> Result<u64> {
        let doc_id = self.next_doc_id.fetch_add(1, Ordering::SeqCst);
        self.upsert_document(doc_id, document)?;
        Ok(doc_id)
    }

    /// Add a raw payload document with automatically assigned doc_id (embedding occurs inside).
    pub fn add_document_payload(&self, payload: DocumentPayload) -> Result<u64> {
        let doc_id = self.next_doc_id.fetch_add(1, Ordering::SeqCst);
        self.upsert_document_payload(doc_id, payload)?;
        Ok(doc_id)
    }

    pub fn upsert_document(
        &self,
        doc_id: u64,
        document: DocumentVector,
    ) -> Result<RegistryVersion> {
        self.validate_document_fields(&document)?;
        let entries = build_field_entries(&document);
        let version = self
            .registry
            .upsert(doc_id, &entries, document.metadata.clone())?;

        if let Err(err) = self.apply_field_updates(doc_id, version, &document.fields) {
            // Best-effort rollback to keep registry consistent with field state.
            let _ = self.registry.delete(doc_id);
            return Err(err);
        }

        let cached_document = document.clone();
        self.wal.append(WalPayload::Upsert { doc_id, document })?;
        self.documents.write().insert(doc_id, cached_document);
        self.persist_state()?;
        self.bump_next_doc_id(doc_id);
        self.maybe_compact_wal()?;
        Ok(version)
    }

    /// Upsert a document represented by raw payloads that require embedding.
    pub fn upsert_document_payload(
        &self,
        doc_id: u64,
        payload: DocumentPayload,
    ) -> Result<RegistryVersion> {
        let document = self.embed_document_payload(doc_id, payload)?;
        self.upsert_document(doc_id, document)
    }

    fn embed_document_payload(&self, _doc_id: u64, payload: DocumentPayload) -> Result<DocumentVector> {
        let mut document = DocumentVector::new();
        document.metadata = payload.metadata;

        for (field_name, field_payload) in payload.fields.into_iter() {
            let vectors = self.embed_field_payload(&field_name, field_payload)?;
            if !vectors.vectors.is_empty() || !vectors.metadata.is_empty() {
                document.fields.insert(field_name, vectors);
            }
        }

        Ok(document)
    }

    fn embed_field_payload(&self, field_name: &str, payload: FieldPayload) -> Result<FieldVectors> {
        let handle = self.fields.get(field_name).ok_or_else(|| {
            PlatypusError::invalid_argument(format!(
                "vector field '{field_name}' is not registered"
            ))
        })?;
        let field_config = handle.field.config().clone();
        if payload.is_empty() {
            let empty = FieldVectors {
                metadata: payload.metadata,
                ..FieldVectors::default()
            };
            return Ok(empty);
        }
        struct ResolvedEmbedders {
            name: String,
            executor: Arc<EmbedderExecutor>,
            text: Option<Arc<dyn TextEmbedder>>,
            image: Option<Arc<dyn ImageEmbedder>>,
        }

        let needs_text = payload
            .segments
            .iter()
            .any(|segment| matches!(segment.source, PayloadSource::Text { .. }));
        let needs_image = payload.segments.iter().any(|segment| {
            matches!(
                segment.source,
                PayloadSource::Bytes { .. } | PayloadSource::Uri { .. }
            )
        });

        let embedder_handles = if needs_text || needs_image {
            let embedder_name = field_config.embedder.as_ref().ok_or_else(|| {
                PlatypusError::invalid_config(format!(
                    "vector field '{field_name}' must specify 'embedder' to ingest raw payloads"
                ))
            })?;
            let executor = self.ensure_embedder_executor()?;
            let text = if needs_text {
                Some(self.embedder_registry.resolve_text(embedder_name)?)
            } else {
                None
            };
            let image = if needs_image {
                Some(self.embedder_registry.resolve_image(embedder_name)?)
            } else {
                None
            };
            Some(ResolvedEmbedders {
                name: embedder_name.clone(),
                executor,
                text,
                image,
            })
        } else {
            None
        };

        let mut field_vectors = FieldVectors {
            metadata: payload.metadata,
            ..FieldVectors::default()
        };

        for segment in payload.segments.into_iter() {
            let SegmentPayload {
                source,
                vector_type: segment_vector_type,
                weight: segment_weight,
                metadata,
            } = segment;
            let vector_type = match segment_vector_type {
                VectorType::Generic => field_config.vector_type.clone(),
                explicit => explicit,
            };
            let weight = if segment_weight <= 0.0 {
                1.0
            } else {
                segment_weight
            };
            match source {
                PayloadSource::Text { value } => {
                    let handles = embedder_handles.as_ref().ok_or_else(|| {
                        PlatypusError::invalid_config(format!(
                            "vector field '{field_name}' must specify 'embedder' to ingest text segments"
                        ))
                    })?;
                    let embedder = handles.text.as_ref().ok_or_else(|| {
                        PlatypusError::invalid_config(format!(
                            "embedder '{}' does not support text segments",
                            handles.name
                        ))
                    })?;
                    let embedder_name = handles.name.clone();
                    let executor = handles.executor.clone();
                    let embedder_clone = Arc::clone(embedder);
                    let text_value = value;
                    let vector =
                        executor.run(async move { embedder_clone.embed(&text_value).await })?;
                    vector.validate_dimension(field_config.dimension)?;
                    if !vector.is_valid() {
                        return Err(PlatypusError::InvalidOperation(format!(
                            "embedder '{}' produced invalid values for field '{}'",
                            embedder_name, field_name
                        )));
                    }
                    let mut stored: StoredVector = vector.into();
                    stored.embedder_id = field_config.embedder_id.clone();
                    stored.vector_type = vector_type;
                    stored.weight = weight;
                    stored.attributes.extend(metadata);
                    field_vectors.vectors.push(stored);
                }
                PayloadSource::Bytes { bytes, mime } => {
                    let handles = embedder_handles.as_ref().ok_or_else(|| {
                        PlatypusError::invalid_config(format!(
                            "vector field '{field_name}' must specify 'embedder' to ingest binary segments"
                        ))
                    })?;
                    let embedder = handles.image.as_ref().ok_or_else(|| {
                        PlatypusError::invalid_config(format!(
                            "embedder '{}' does not support binary segments",
                            handles.name
                        ))
                    })?;
                    let embedder_name = handles.name.clone();
                    let executor = handles.executor.clone();
                    let embedder_clone = Arc::clone(embedder);
                    let payload = bytes.clone();
                    let mime_hint = mime.clone();
                    let vector = executor.run(async move {
                        embedder_clone
                            .embed_bytes(payload.as_ref(), mime_hint.as_deref())
                            .await
                    })?;
                    vector.validate_dimension(field_config.dimension)?;
                    if !vector.is_valid() {
                        return Err(PlatypusError::InvalidOperation(format!(
                            "embedder '{}' produced invalid values for field '{}'",
                            embedder_name, field_name
                        )));
                    }
                    let mut stored: StoredVector = vector.into();
                    stored.embedder_id = field_config.embedder_id.clone();
                    stored.vector_type = vector_type;
                    stored.weight = weight;
                    stored.attributes.extend(metadata);
                    field_vectors.vectors.push(stored);
                }
                PayloadSource::Uri { uri, media_hint } => {
                    let handles = embedder_handles.as_ref().ok_or_else(|| {
                        PlatypusError::invalid_config(format!(
                            "vector field '{field_name}' must specify 'embedder' to ingest URI segments"
                        ))
                    })?;
                    let embedder = handles.image.as_ref().ok_or_else(|| {
                        PlatypusError::invalid_config(format!(
                            "embedder '{}' does not support URI segments",
                            handles.name
                        ))
                    })?;
                    let embedder_name = handles.name.clone();
                    let executor = handles.executor.clone();
                    let embedder_clone = Arc::clone(embedder);
                    let uri_value = uri;
                    let hint = media_hint.clone();
                    let vector = executor.run(async move {
                        embedder_clone
                            .embed_uri(uri_value.as_str(), hint.as_deref())
                            .await
                    })?;
                    vector.validate_dimension(field_config.dimension)?;
                    if !vector.is_valid() {
                        return Err(PlatypusError::InvalidOperation(format!(
                            "embedder '{}' produced invalid values for field '{}'",
                            embedder_name, field_name
                        )));
                    }
                    let mut stored: StoredVector = vector.into();
                    stored.embedder_id = field_config.embedder_id.clone();
                    stored.vector_type = vector_type;
                    stored.weight = weight;
                    stored.attributes.extend(metadata);
                    field_vectors.vectors.push(stored);
                }
                PayloadSource::Vector { data, embedder_id } => {
                    if embedder_id != field_config.embedder_id {
                        return Err(PlatypusError::invalid_argument(format!(
                            "vector field '{field_name}' only accepts embedder_id '{}' but got '{}'",
                            field_config.embedder_id, embedder_id
                        )));
                    }
                    if data.len() != field_config.dimension {
                        return Err(PlatypusError::invalid_argument(format!(
                            "vector field '{field_name}' expects dimension {} but received {}",
                            field_config.dimension,
                            data.len()
                        )));
                    }
                    let mut stored = StoredVector::new(
                        data.clone(),
                        field_config.embedder_id.clone(),
                        vector_type,
                    );
                    stored.weight = weight;
                    stored.attributes.extend(metadata);
                    field_vectors.vectors.push(stored);
                }
            }
        }

        Ok(field_vectors)
    }

    /// Embed a raw payload for use as query vectors without mutating the index.
    pub fn embed_query_field_payload(
        &self,
        field_name: &str,
        payload: FieldPayload,
    ) -> Result<Vec<QueryVector>> {
        let vectors = self.embed_field_payload(field_name, payload)?;
        Ok(vectors
            .vectors
            .into_iter()
            .map(|vector| QueryVector {
                vector,
                weight: 1.0,
            })
            .collect())
    }

    fn ensure_embedder_executor(&self) -> Result<Arc<EmbedderExecutor>> {
        let mut guard = self.embedder_executor.lock();
        if let Some(executor) = guard.as_ref() {
            return Ok(executor.clone());
        }
        let executor = Arc::new(EmbedderExecutor::new()?);
        *guard = Some(executor.clone());
        Ok(executor)
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
        if filter_matches
            .as_ref()
            .is_some_and(|matches| matches.is_empty())
        {
            return Ok(VectorEngineSearchResults::default());
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
                FieldSelector::VectorType(vector_type) => {
                    let mut matched = false;
                    for (field_name, field_handle) in &self.fields {
                        if field_handle.field.config().vector_type == *vector_type {
                            resolved.push(field_name.clone());
                            matched = true;
                        }
                    }
                    if !matched {
                        return Err(PlatypusError::not_found(format!(
                            "no vector fields registered with vector type '{vector_type:?}'"
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
            .filter(|candidate| {
                candidate.vector.embedder_id == config.embedder_id
                    && candidate.vector.vector_type == config.vector_type
            })
            .cloned()
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
                let flat = FlatIndexConfig {
                    dimension: config.dimension,
                    distance_metric: config.distance,
                    ..FlatIndexConfig::default()
                };
                let writer =
                    FlatIndexWriter::with_storage(flat, writer_config.clone(), storage.clone())?;
                Arc::new(LegacyVectorFieldWriter::new(field_name.to_string(), writer))
            }
            VectorIndexKind::Hnsw => {
                let hnsw = HnswIndexConfig {
                    dimension: config.dimension,
                    distance_metric: config.distance,
                    ..HnswIndexConfig::default()
                };
                let writer =
                    HnswIndexWriter::with_storage(hnsw, writer_config.clone(), storage.clone())?;
                Arc::new(LegacyVectorFieldWriter::new(field_name.to_string(), writer))
            }
            VectorIndexKind::Ivf => {
                let ivf = IvfIndexConfig {
                    dimension: config.dimension,
                    distance_metric: config.distance,
                    ..IvfIndexConfig::default()
                };
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
                let flat = FlatIndexConfig {
                    dimension: config.dimension,
                    distance_metric: config.distance,
                    ..FlatIndexConfig::default()
                };
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
                let hnsw = HnswIndexConfig {
                    dimension: config.dimension,
                    distance_metric: config.distance,
                    ..HnswIndexConfig::default()
                };
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
                let ivf = IvfIndexConfig {
                    dimension: config.dimension,
                    distance_metric: config.distance,
                    ..IvfIndexConfig::default()
                };
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

    fn validate_document_fields(&self, document: &DocumentVector) -> Result<()> {
        for field_name in document.fields.keys() {
            if !self.fields.contains_key(field_name) {
                return Err(PlatypusError::invalid_argument(format!(
                    "vector field '{field_name}' is not registered"
                )));
            }
        }
        Ok(())
    }

    fn bump_next_doc_id(&self, doc_id: u64) {
        let _ = self.next_doc_id.fetch_update(
            Ordering::SeqCst,
            Ordering::SeqCst,
            |current| {
                if doc_id >= current {
                    Some(doc_id.saturating_add(1))
                } else {
                    None
                }
            },
        );
    }

    fn recompute_next_doc_id(&self) {
        let max_id = self
            .documents
            .read()
            .keys()
            .copied()
            .max()
            .unwrap_or(0);
        self.bump_next_doc_id(max_id);
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
        self.recompute_next_doc_id();
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

        #[derive(serde::Deserialize)]
        struct LegacySnapshotDocument {
            doc_id: u64,
            #[serde(default)]
            fields: HashMap<String, FieldVectors>,
            #[serde(default)]
            metadata: HashMap<String, String>,
        }

        let snapshot = match serde_json::from_slice::<DocumentSnapshot>(&buffer) {
            Ok(snapshot) => snapshot,
            Err(primary_err) => {
                let docs: Vec<LegacySnapshotDocument> =
                    serde_json::from_slice(&buffer).map_err(|_| primary_err)?;
                let converted = docs
                    .into_iter()
                    .map(|legacy| SnapshotDocument {
                        doc_id: legacy.doc_id,
                        document: DocumentVector {
                            fields: legacy.fields,
                            metadata: legacy.metadata,
                        },
                    })
                    .collect();
                DocumentSnapshot {
                    last_wal_seq: 0,
                    documents: converted,
                }
            }
        };
        let map = snapshot
            .documents
            .into_iter()
            .map(|doc| (doc.doc_id, doc.document))
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
                WalPayload::Upsert { doc_id, document } => {
                    if document.fields.is_empty() {
                        documents.remove(&doc_id);
                        continue;
                    }
                    if let Some(entry) = self.registry.get(doc_id) {
                        self.apply_field_updates(doc_id, entry.version, &document.fields)?;
                    }
                    documents.insert(doc_id, document);
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

    fn apply_documents_to_fields(&self, documents: &HashMap<u64, DocumentVector>) -> Result<()> {
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
        let documents: Vec<SnapshotDocument> = guard
            .iter()
            .map(|(doc_id, document)| SnapshotDocument {
                doc_id: *doc_id,
                document: document.clone(),
            })
            .collect();
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

        let mut entries: Vec<(u64, DocumentVector)> = documents.into_iter().collect();
        entries.sort_by(|(a_id, _), (b_id, _)| a_id.cmp(b_id));
        let mut records = Vec::with_capacity(entries.len());
        for (idx, (doc_id, document)) in entries.into_iter().enumerate() {
            records.push(WalRecord {
                seq: (idx as u64) + 1,
                payload: WalPayload::Upsert { doc_id, document },
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

fn build_field_entries(document: &DocumentVector) -> Vec<FieldEntry> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::storage::prefixed::PrefixedStorage;
    use crate::vector::DistanceMetric;
    use crate::vector::core::document::{
        DocumentPayload, DocumentVector, FieldPayload, FieldVectors, PayloadSource,
        SegmentPayload, StoredVector, VectorType,
    };
    use crate::vector::core::vector::Vector;
    use crate::vector::core::document::FieldVectors as CoreFieldVectors;
    use crate::vector::field::{
        FieldSearchResults, VectorFieldReader, VectorFieldStats, VectorFieldWriter,
    };
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::{Arc, Mutex};
    use tempfile::NamedTempFile;

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
        fn add_field_vectors(&self, doc_id: u64, field: &CoreFieldVectors, version: u64) -> Result<()> {
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
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: weight,
        };
        let config = VectorEngineConfig {
            fields: HashMap::new(),
            embedders: HashMap::new(),
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
            VectorType::Text,
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
                VectorType::Text,
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

        let mut ja_doc = DocumentVector::new();
        ja_doc.metadata.insert("lang".into(), "ja".into());
        ja_doc.add_field("body", field_vectors_with_metadata(None));
        collection.upsert_document(10, ja_doc).expect("upsert ja");

        let mut en_doc = DocumentVector::new();
        en_doc.metadata.insert("lang".into(), "en".into());
        en_doc.add_field("body", field_vectors_with_metadata(None));
        collection.upsert_document(11, en_doc).expect("upsert en");

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

        let mut title_doc = DocumentVector::new();
        title_doc.add_field(
            "body",
            field_vectors_with_metadata(Some(("section", "title"))),
        );
        collection
            .upsert_document(1, title_doc)
            .expect("upsert title doc");

        let mut body_doc = DocumentVector::new();
        body_doc.add_field(
            "body",
            field_vectors_with_metadata(Some(("section", "body"))),
        );
        collection
            .upsert_document(2, body_doc)
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
    fn search_supports_vector_type_selector() {
        let hits = vec![FieldHit {
            doc_id: 3,
            field: "body".into(),
            score: 0.9,
            distance: 0.1,
            metadata: HashMap::new(),
        }];
        let (collection, _) = collection_with_field(hits, 1.0);
        let mut query = sample_query(5);
        query.fields = Some(vec![FieldSelector::VectorType(VectorType::Text)]);
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
                VectorType::Text,
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
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            embedders: HashMap::new(),
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

        let mut doc = DocumentVector::new();
        let mut vectors = FieldVectors::default();
        vectors.vectors.push(StoredVector::new(
            Arc::<[f32]>::from([0.5, 0.1, 0.3]),
            "mock".into(),
            VectorType::Text,
        ));
        doc.add_field("body", vectors);
        collection.upsert_document(5, doc).expect("upsert");

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
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            embedders: HashMap::new(),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let collection = VectorEngine::new(config, storage.clone(), None).expect("collection");

        let mut doc = DocumentVector::new();
        let mut vectors = FieldVectors::default();
        vectors.vectors.push(StoredVector::new(
            Arc::<[f32]>::from([1.0, 0.0, 0.0]),
            "mock".into(),
            VectorType::Text,
        ));
        doc.add_field("body", vectors);
        collection.upsert_document(1, doc).expect("upsert");

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
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            embedders: HashMap::new(),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let collection =
                VectorEngine::new(config.clone(), storage.clone(), None).expect("collection");
            let mut doc = DocumentVector::new();
            let mut vectors = FieldVectors::default();
            vectors.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([0.2, 0.3, 0.4]),
                "mock".into(),
                VectorType::Text,
            ));
            doc.add_field("body", vectors);
            collection.upsert_document(10, doc).expect("upsert");
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
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            embedders: HashMap::new(),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let collection =
                VectorEngine::new(config.clone(), storage.clone(), None).expect("collection");
            let mut doc_one = DocumentVector::new();
            let mut vectors_one = FieldVectors::default();
            vectors_one.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([1.0, 0.0, 0.0]),
                "mock".into(),
                VectorType::Text,
            ));
            doc_one.add_field("body", vectors_one);
            collection.upsert_document(1, doc_one).expect("upsert doc one");

            let mut doc_two = DocumentVector::new();
            let mut vectors_two = FieldVectors::default();
            vectors_two.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([0.8, 0.0, 0.2]),
                "mock".into(),
                VectorType::Text,
            ));
            doc_two.add_field("body", vectors_two);
            collection.upsert_document(2, doc_two).expect("upsert doc two");

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
    fn upsert_document_payload_embeds_vectors() {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            vector_type: VectorType::Text,
            embedder: Some("mock_embedder".into()),
            base_weight: 1.0,
        };
        let embedders = HashMap::from([(
            String::from("mock_embedder"),
            VectorEmbedderConfig {
                provider: config::VectorEmbedderProvider::External,
                model: "mock".into(),
                options: HashMap::new(),
            },
        )]);
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            embedders,
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = VectorEngine::new(config, storage, None).expect("engine");
        engine
            .register_embedder_instance("mock_embedder", Arc::new(MockTextEmbedder::new(3)))
            .expect("register embedder");

        let mut field_payload = FieldPayload::default();
        field_payload.add_text_segment("hello world");
        let mut document_payload = DocumentPayload::new();
        document_payload.add_field("body", field_payload);

        let version = engine
            .upsert_document_payload(1, document_payload)
            .expect("upsert raw payload");
        assert!(version > 0);

        let stats = engine.stats().expect("stats");
        assert_eq!(stats.document_count, 1);
        let body_stats = stats.fields.get("body").expect("body stats");
        assert_eq!(body_stats.vector_count, 1);
    }

    #[test]
    fn embed_query_payload_accepts_bytes_segments() {
        let engine = engine_with_multimodal_embedder();
        let mut payload = FieldPayload::default();
        payload.add_segment(SegmentPayload::new(
            PayloadSource::Bytes {
                bytes: Arc::<[u8]>::from(vec![1_u8, 2, 3, 4]),
                mime: Some("image/png".into()),
            },
            VectorType::Image,
        ));

        let vectors = engine
            .embed_query_field_payload("body", payload)
            .expect("embed bytes");
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].vector.vector_type, VectorType::Image);
        assert_eq!(vectors[0].vector.embedder_id, "mock-multimodal");
    }

    #[test]
    fn embed_query_payload_accepts_uri_segments() {
        let engine = engine_with_multimodal_embedder();
        let mut payload = FieldPayload::default();
        let mut image_file = NamedTempFile::new().expect("temp file");
        image_file
            .write_all(&[9_u8, 8, 7, 6])
            .expect("write temp image");
        let uri = image_file.path().to_string_lossy().to_string();
        payload.add_segment(SegmentPayload::new(
            PayloadSource::Uri {
                uri,
                media_hint: Some("image/png".into()),
            },
            VectorType::Image,
        ));

        let vectors = engine
            .embed_query_field_payload("body", payload)
            .expect("embed uri");
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].vector.vector_type, VectorType::Image);
    }

    #[test]
    fn wal_compaction_drops_tombstones() {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            embedders: HashMap::new(),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let collection = VectorEngine::new(config, storage, None).expect("collection");

        for doc_id in 0..6_u64 {
            let mut doc = DocumentVector::new();
            let mut vectors = FieldVectors::default();
            vectors.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([1.0, 0.0, 0.0]),
                "mock".into(),
                VectorType::Text,
            ));
            doc.add_field("body", vectors);
            collection.upsert_document(doc_id, doc).expect("upsert");
        }

        collection.delete_document(0).expect("delete doc zero");

        let records = collection.wal.records();
        assert_eq!(records.len(), 5);
        for (expected_seq, record) in records.iter().enumerate() {
            assert_eq!(record.seq, (expected_seq as u64) + 1);
            match &record.payload {
                WalPayload::Upsert { doc_id, .. } => {
                    assert_ne!(*doc_id, 0);
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
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            embedders: HashMap::new(),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let collection =
                VectorEngine::new(config.clone(), storage.clone(), None).expect("collection");
            let mut doc = DocumentVector::new();
            let mut vectors = FieldVectors::default();
            vectors.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([0.4, 0.5, 0.6]),
                "mock".into(),
                VectorType::Text,
            ));
            doc.add_field("body", vectors);
            collection.upsert_document(77, doc).expect("upsert");
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
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.0,
        };
        let config = VectorEngineConfig {
            fields: HashMap::from([(String::from("body"), field_config.clone())]),
            embedders: HashMap::new(),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let collection =
                VectorEngine::new(config.clone(), storage.clone(), None).expect("collection");
            let mut doc = DocumentVector::new();
            let mut vectors = FieldVectors::default();
            vectors.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([1.0, 0.0, 0.0]),
                "mock".into(),
                VectorType::Text,
            ));
            doc.add_field("body", vectors);
            collection.upsert_document(1, doc).expect("upsert");
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
        let mut doc = DocumentVector::new();
        let mut vectors = FieldVectors::default();
        vectors.vectors.push(StoredVector::new(
            Arc::<[f32]>::from([0.1, 0.2, 0.3]),
            "mock".into(),
            VectorType::Text,
        ));
        doc.add_field("body", vectors);

        let version = collection.upsert_document(99, doc).expect("upsert");
        let additions = writer.additions().lock().unwrap().clone();
        assert_eq!(additions.len(), 1);
        assert_eq!(additions[0].0, 99);
        assert_eq!(additions[0].1, 1);
        assert_eq!(additions[0].2, version);
    }

    #[test]
    fn delete_invokes_field_writer() {
        let (collection, writer) = collection_with_field(vec![], 1.0);
        let mut doc = DocumentVector::new();
        doc.add_field("body", FieldVectors::default());
        collection.upsert_document(7, doc).expect("upsert");

        collection.delete_document(7).expect("delete");
        let deletions = writer.deletions().lock().unwrap().clone();
        assert_eq!(deletions.len(), 1);
        assert_eq!(deletions[0].0, 7);
    }

    fn engine_with_multimodal_embedder() -> VectorEngine {
        let mut fields = HashMap::new();
        fields.insert(
            "body".into(),
            VectorFieldConfig {
                dimension: 3,
                distance: DistanceMetric::Cosine,
                index: VectorIndexKind::Flat,
                embedder_id: "mock-multimodal".into(),
                vector_type: VectorType::Image,
                embedder: Some("mock_multi".into()),
                base_weight: 1.0,
            },
        );

        let embedders = HashMap::from([(
            "mock_multi".into(),
            VectorEmbedderConfig {
                provider: config::VectorEmbedderProvider::External,
                model: "mock".into(),
                options: HashMap::new(),
            },
        )]);

        let config = VectorEngineConfig {
            fields,
            embedders,
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = VectorEngine::new(config, storage, None).expect("engine");
        let embedder = Arc::new(MockMultimodalEmbedder::new(3));
        engine
            .register_multimodal_embedder_instance("mock_multi", embedder.clone(), embedder)
            .expect("register multimodal");
        engine
    }

    #[derive(Debug)]
    struct MockMultimodalEmbedder {
        dimension: usize,
    }

    impl MockMultimodalEmbedder {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }

        fn vector_from_value(&self, value: f32) -> Vector {
            Vector::new(vec![value; self.dimension])
        }
    }

    #[async_trait]
    impl TextEmbedder for MockMultimodalEmbedder {
        async fn embed(&self, text: &str) -> Result<Vector> {
            let aggregate = text.bytes().map(|b| b as f32).sum::<f32>();
            let divisor = text.len().max(1) as f32;
            Ok(self.vector_from_value(aggregate / divisor))
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock-multimodal"
        }
    }

    #[async_trait]
    impl ImageEmbedder for MockMultimodalEmbedder {
        async fn embed(&self, image_path: &str) -> Result<Vector> {
            let bytes = std::fs::read(image_path)?;
            let aggregate = bytes.iter().map(|b| *b as f32).sum::<f32>();
            let divisor = bytes.len().max(1) as f32;
            Ok(self.vector_from_value(aggregate / divisor))
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock-multimodal"
        }
    }

    #[derive(Debug)]
    struct MockTextEmbedder {
        dimension: usize,
    }

    impl MockTextEmbedder {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl TextEmbedder for MockTextEmbedder {
        async fn embed(&self, text: &str) -> Result<Vector> {
            let value = text.len() as f32;
            Ok(Vector::new(vec![value; self.dimension]))
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock-text-embedder"
        }
    }
}
