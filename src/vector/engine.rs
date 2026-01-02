//! VectorEngine: High-level vector search engine.
//!
//! This module provides a unified interface for vector indexing and search,
//! analogous to `LexicalEngine` for lexical search.
//!
//! # Module Structure
//!
//! - [`config`] - Configuration types (VectorIndexConfig, VectorFieldConfig, VectorIndexKind)
//! - [`embedder`] - Embedding utilities
//! - [`filter`] - Metadata filtering
//! - [`memory`] - In-memory field implementation
//! - [`registry`] - Document vector registry
//! - [`request`] - Search request types
//! - [`response`] - Search response types
//! - [`snapshot`] - Snapshot persistence
//! - [`wal`] - Write-Ahead Logging

pub mod config;
pub mod embedder;
pub mod filter;
pub mod memory;
pub mod query;
pub mod registry;
pub mod request;
pub mod response;
pub mod snapshot;

use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};

use crate::embedding::embedder::{EmbedInput, Embedder};
use crate::embedding::per_field::PerFieldEmbedder;
use crate::error::{Result, SarissaError};
use crate::storage::Storage;
use crate::storage::prefixed::PrefixedStorage;
use crate::vector::core::document::{
    DocumentPayload, DocumentVector, Payload, PayloadSource, StoredVector,
};
use crate::vector::core::vector::Vector;
use crate::vector::field::{
    FieldHit, FieldSearchInput, VectorField, VectorFieldReader, VectorFieldStats, VectorFieldWriter,
};
use crate::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
use crate::vector::index::field::{AdapterBackedVectorField, LegacyVectorFieldWriter};
use crate::vector::index::flat::{
    field_reader::FlatFieldReader, reader::FlatVectorIndexReader, writer::FlatIndexWriter,
};
use crate::vector::index::hnsw::segment::manager::{SegmentManager, SegmentManagerConfig};
use crate::vector::index::hnsw::{
    field_reader::HnswFieldReader, reader::HnswIndexReader, writer::HnswIndexWriter,
};
use crate::vector::index::ivf::{
    field_reader::IvfFieldReader, reader::IvfIndexReader, writer::IvfIndexWriter,
};
use crate::vector::index::segmented_field::SegmentedVectorField;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

use self::embedder::{EmbedderExecutor, VectorEmbedderRegistry};
use self::filter::RegistryFilterMatches;
use self::memory::{FieldHandle, FieldRuntime, InMemoryVectorField};
use self::registry::{DocumentEntry, DocumentVectorRegistry, FieldEntry};
use self::request::{FieldSelector, QueryVector, VectorScoreMode, VectorSearchRequest};
use self::response::{VectorHit, VectorSearchResults, VectorStats};
use self::snapshot::{
    COLLECTION_MANIFEST_FILE, COLLECTION_MANIFEST_VERSION, CollectionManifest,
    DOCUMENT_SNAPSHOT_FILE, DOCUMENT_SNAPSHOT_TEMP_FILE, DocumentSnapshot, FIELD_INDEX_BASENAME,
    REGISTRY_NAMESPACE, REGISTRY_SNAPSHOT_FILE, SnapshotDocument,
};
use crate::vector::wal::{WalEntry, WalManager};
use config::{VectorFieldConfig, VectorIndexConfig, VectorIndexKind};

const WAL_COMPACTION_THRESHOLD: usize = 1000;

/// A high-level vector search engine that provides both indexing and searching.
///
/// The `VectorEngine` provides a simplified, unified interface for all vector operations,
/// managing multiple vector fields, persistence, and search.
pub struct VectorEngine {
    config: Arc<VectorIndexConfig>,
    field_configs: Arc<RwLock<HashMap<String, VectorFieldConfig>>>,
    fields: Arc<RwLock<HashMap<String, FieldHandle>>>,
    registry: Arc<DocumentVectorRegistry>,
    embedder_registry: Arc<VectorEmbedderRegistry>,
    embedder_executor: Mutex<Option<Arc<EmbedderExecutor>>>,
    wal: Arc<WalManager>,
    storage: Arc<dyn Storage>,
    documents: Arc<RwLock<HashMap<u64, DocumentVector>>>,
    snapshot_wal_seq: AtomicU64,
    next_doc_id: AtomicU64,
    closed: AtomicU64, // 0 = open, 1 = closed
}

impl fmt::Debug for VectorEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorEngine")
            .field("config", &self.config)
            .field("field_count", &self.fields.read().len())
            .finish()
    }
}

impl VectorEngine {
    /// Create a new vector engine with the given storage and configuration.
    pub fn new(storage: Arc<dyn Storage>, config: VectorIndexConfig) -> Result<Self> {
        let embedder_registry = Arc::new(VectorEmbedderRegistry::new());
        let registry = Arc::new(DocumentVectorRegistry::default());
        let field_configs = Arc::new(RwLock::new(config.fields.clone()));

        // Store the embedder from config before moving config into Arc
        let config_embedder = config.embedder.clone();

        let mut engine = Self {
            config: Arc::new(config),
            field_configs: field_configs.clone(),
            fields: Arc::new(RwLock::new(HashMap::new())),
            registry,
            embedder_registry,
            embedder_executor: Mutex::new(None),
            wal: Arc::new(WalManager::new(storage.clone(), "vector_engine.wal")?),
            storage,
            documents: Arc::new(RwLock::new(HashMap::new())),
            snapshot_wal_seq: AtomicU64::new(0),
            next_doc_id: AtomicU64::new(0),
            closed: AtomicU64::new(0),
        };
        engine.load_persisted_state()?;

        // Register embedder instances from the config (after fields are instantiated)
        engine.register_embedder_from_config(config_embedder)?;

        Ok(engine)
    }

    /// Register embedder instances from the Embedder trait object.
    fn register_embedder_from_config(&self, embedder: Arc<dyn Embedder>) -> Result<()> {
        let configs = self.field_configs.read().clone();
        if let Some(per_field) = embedder.as_any().downcast_ref::<PerFieldEmbedder>() {
            for field_name in configs.keys() {
                let field_embedder = per_field.get_embedder(field_name).clone();
                self.embedder_registry
                    .register(field_name.clone(), field_embedder.clone());
            }
        } else {
            for field_name in configs.keys() {
                self.embedder_registry
                    .register(field_name.clone(), embedder.clone());
            }
        }

        Ok(())
    }

    /// Register a concrete field implementation. Each field name must be unique.
    pub fn register_field_impl(&self, field: Arc<dyn VectorField>) -> Result<()> {
        let name = field.name().to_string();
        let mut fields = self.fields.write();
        if fields.contains_key(&name) {
            return Err(SarissaError::invalid_config(format!(
                "vector field '{name}' is already registered"
            )));
        }
        let runtime = FieldRuntime::from_field(&field);
        fields.insert(name, FieldHandle { field, runtime });
        Ok(())
    }

    /// Convenience helper to register a field backed by legacy adapters.
    pub fn register_adapter_field(
        &self,
        name: impl Into<String>,
        config: VectorFieldConfig,
        writer: Arc<dyn VectorFieldWriter>,
        reader: Arc<dyn VectorFieldReader>,
    ) -> Result<()> {
        let field: Arc<dyn VectorField> =
            Arc::new(AdapterBackedVectorField::new(name, config, writer, reader));
        self.register_field_impl(field)
    }

    // =========================================================================
    // Internal methods
    // =========================================================================

    fn embed_document_payload_internal(
        &self,
        _doc_id: u64,
        payload: DocumentPayload,
    ) -> Result<DocumentVector> {
        // Ensure fields are registered (implicit schema generation if enabled)
        for (field_name, field_payload) in payload.fields.iter() {
            self.ensure_field_for_payload(field_name, field_payload)?;
        }

        let mut document = DocumentVector::new();
        document.metadata = payload.metadata;

        for (field_name, field_payload) in payload.fields.into_iter() {
            let vector = self.embed_payload(&field_name, field_payload)?;
            document.fields.insert(field_name, vector);
        }

        Ok(document)
    }

    fn ensure_field_for_payload(&self, field_name: &str, payload: &Payload) -> Result<()> {
        // Fast path: already registered
        if self.fields.read().contains_key(field_name) {
            return Ok(());
        }

        if !self.config.implicit_schema {
            return Err(SarissaError::invalid_argument(format!(
                "vector field '{field_name}' is not registered"
            )));
        }

        let field_config = self.build_field_config_for_payload(field_name, payload)?;

        // Persist in config cache
        self.field_configs
            .write()
            .insert(field_name.to_string(), field_config.clone());

        // Build field runtime
        let field = self.create_vector_field(field_name.to_string(), field_config)?;
        self.register_field_impl(field)?;

        // Register embedder for this field
        self.register_embedder_for_field(field_name, self.config.embedder.clone())?;

        // Persist manifest to record new field configuration
        self.persist_manifest()?;
        Ok(())
    }

    fn build_field_config_for_payload(
        &self,
        field_name: &str,
        payload: &Payload,
    ) -> Result<VectorFieldConfig> {
        let dimension = match &payload.source {
            PayloadSource::Text { .. } | PayloadSource::Bytes { .. } => {
                self.config.default_dimension.ok_or_else(|| {
                    SarissaError::invalid_config(
                        "implicit schema requires default_dimension to be set",
                    )
                })?
            }
            PayloadSource::Vector { data } => data.len(),
        };

        if dimension == 0 {
            return Err(SarissaError::invalid_config(format!(
                "cannot register field '{field_name}' with zero dimension"
            )));
        }

        Ok(VectorFieldConfig {
            dimension,
            distance: self.config.default_distance,
            index: self.config.default_index_kind,
            metadata: HashMap::new(),
            base_weight: self.config.default_base_weight,
        })
    }

    fn register_embedder_for_field(
        &self,
        field_name: &str,
        embedder: Arc<dyn Embedder>,
    ) -> Result<()> {
        if let Some(per_field) = embedder.as_any().downcast_ref::<PerFieldEmbedder>() {
            let field_embedder = per_field.get_embedder(field_name).clone();
            self.embedder_registry
                .register(field_name.to_string(), field_embedder);
        } else {
            self.embedder_registry
                .register(field_name.to_string(), embedder);
        }
        Ok(())
    }

    /// Embeds a single `Payload` into a `StoredVector`.
    fn embed_payload(&self, field_name: &str, payload: Payload) -> Result<StoredVector> {
        let fields = self.fields.read();
        let handle = fields.get(field_name).ok_or_else(|| {
            SarissaError::invalid_argument(format!("vector field '{field_name}' is not registered"))
        })?;
        let field_config = handle.field.config().clone();
        drop(fields);

        let Payload { source } = payload;

        match source {
            PayloadSource::Text { value } => {
                let executor = self.ensure_embedder_executor()?;
                let embedder = self.embedder_registry.resolve(field_name)?;

                if !embedder.supports_text() {
                    return Err(SarissaError::invalid_config(format!(
                        "embedder '{}' does not support text embedding",
                        field_name
                    )));
                }

                let embedder_name_owned = field_name.to_string();
                let text_value = value;
                let vector = executor
                    .run(async move { embedder.embed(&EmbedInput::Text(&text_value)).await })?;
                vector.validate_dimension(field_config.dimension)?;
                if !vector.is_valid() {
                    return Err(SarissaError::InvalidOperation(format!(
                        "embedder '{}' produced invalid values for field '{}'",
                        embedder_name_owned, field_name
                    )));
                }
                let stored: StoredVector = vector.into();
                Ok(stored)
            }
            PayloadSource::Bytes { bytes, mime } => {
                let executor = self.ensure_embedder_executor()?;
                let embedder = self.embedder_registry.resolve(field_name)?;

                if !embedder.supports_image() {
                    return Err(SarissaError::invalid_config(format!(
                        "embedder '{}' does not support image embedding",
                        field_name
                    )));
                }

                let embedder_name_owned = field_name.to_string();
                let payload_bytes = bytes.clone();
                let mime_hint = mime.clone();
                let vector = executor.run(async move {
                    embedder
                        .embed(&EmbedInput::Bytes(&payload_bytes, mime_hint.as_deref()))
                        .await
                })?;
                vector.validate_dimension(field_config.dimension)?;
                if !vector.is_valid() {
                    return Err(SarissaError::InvalidOperation(format!(
                        "embedder '{}' produced invalid values for field '{}'",
                        embedder_name_owned, field_name
                    )));
                }
                let stored: StoredVector = vector.into();

                Ok(stored)
            }
            PayloadSource::Vector { data } => {
                if data.len() != field_config.dimension {
                    return Err(SarissaError::invalid_argument(format!(
                        "vector field '{field_name}' expects dimension {} but received {}",
                        field_config.dimension,
                        data.len()
                    )));
                }
                Ok(StoredVector::new(data.clone()))
            }
        }
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

    fn instantiate_configured_fields(&mut self) -> Result<()> {
        let configs: Vec<(String, VectorFieldConfig)> = self
            .field_configs
            .read()
            .iter()
            .map(|(name, config)| (name.clone(), config.clone()))
            .collect();

        for (name, config) in configs {
            // Skip if already registered
            if self.fields.read().contains_key(&name) {
                continue;
            }
            let field = self.create_vector_field(name, config)?;
            self.register_field_impl(field)?;
        }
        Ok(())
    }

    fn create_vector_field(
        &self,
        name: String,
        config: VectorFieldConfig,
    ) -> Result<Arc<dyn VectorField>> {
        match config.index {
            VectorIndexKind::Hnsw => {
                let storage = self.field_storage(&name);
                let manager_config = SegmentManagerConfig::default();
                let segment_manager =
                    Arc::new(SegmentManager::new(manager_config, storage.clone())?);

                Ok(Arc::new(SegmentedVectorField::create(
                    name,
                    config,
                    segment_manager,
                    storage,
                )?))
            }
            _ => {
                let delegate = self.build_delegate_writer(&name, &config)?;
                Ok(Arc::new(InMemoryVectorField::new(name, config, delegate)?))
            }
        }
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
                let writer = FlatIndexWriter::with_storage(
                    flat,
                    writer_config.clone(),
                    "vectors.index",
                    storage.clone(),
                )?;
                Arc::new(LegacyVectorFieldWriter::new(field_name.to_string(), writer))
            }
            VectorIndexKind::Hnsw => {
                let hnsw = HnswIndexConfig {
                    dimension: config.dimension,
                    distance_metric: config.distance,
                    ..HnswIndexConfig::default()
                };
                let writer = HnswIndexWriter::with_storage(
                    hnsw,
                    writer_config.clone(),
                    "vectors.index",
                    storage.clone(),
                )?;
                Arc::new(LegacyVectorFieldWriter::new(field_name.to_string(), writer))
            }
            VectorIndexKind::Ivf => {
                let ivf = IvfIndexConfig {
                    dimension: config.dimension,
                    distance_metric: config.distance,
                    ..IvfIndexConfig::default()
                };
                let writer =
                    IvfIndexWriter::with_storage(ivf, writer_config, "vectors.index", storage)?;
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
            return Err(SarissaError::invalid_config(format!(
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
                    FIELD_INDEX_BASENAME,
                    storage.clone(),
                )?;
                let vectors = pending_vectors.take().unwrap_or_default();
                writer.build(vectors)?;
                writer.finalize()?;
                writer.write()?;
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
                    FIELD_INDEX_BASENAME,
                    storage.clone(),
                )?;
                let vectors = pending_vectors.take().unwrap_or_default();
                writer.build(vectors)?;
                writer.finalize()?;
                writer.write()?;
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
                    FIELD_INDEX_BASENAME,
                    storage.clone(),
                )?;
                let vectors = pending_vectors.take().unwrap_or_default();
                writer.build(vectors)?;
                writer.finalize()?;
                writer.write()?;
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
                let flat_config = crate::vector::index::config::FlatIndexConfig {
                    dimension: config.dimension,
                    distance_metric: config.distance,
                    loading_mode: crate::vector::index::config::IndexLoadingMode::default(),
                    ..Default::default()
                };
                let reader = Arc::new(FlatVectorIndexReader::load(
                    &*storage,
                    FIELD_INDEX_BASENAME,
                    flat_config.distance_metric,
                )?);
                Arc::new(FlatFieldReader::new(field_name.to_string(), reader))
            }
            VectorIndexKind::Hnsw => {
                let reader = Arc::new(HnswIndexReader::load(
                    &*storage,
                    FIELD_INDEX_BASENAME,
                    config.distance,
                )?);
                Arc::new(HnswFieldReader::new(field_name.to_string(), reader))
            }
            VectorIndexKind::Ivf => {
                let reader = Arc::new(IvfIndexReader::load(
                    &*storage,
                    FIELD_INDEX_BASENAME,
                    config.distance,
                )?);
                Arc::new(IvfFieldReader::with_n_probe(
                    field_name.to_string(),
                    reader,
                    4,
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
        let fields = self.fields.read();
        for field_name in document.fields.keys() {
            if !fields.contains_key(field_name) {
                return Err(SarissaError::invalid_argument(format!(
                    "vector field '{field_name}' is not registered"
                )));
            }
        }
        Ok(())
    }

    fn bump_next_doc_id(&self, doc_id: u64) {
        let _ = self
            .next_doc_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if doc_id >= current {
                    Some(doc_id.saturating_add(1))
                } else {
                    None
                }
            });
    }

    fn recompute_next_doc_id(&self) {
        let max_id = self.documents.read().keys().copied().max().unwrap_or(0);
        self.bump_next_doc_id(max_id);
    }

    fn delete_fields_for_entry(&self, doc_id: u64, entry: &DocumentEntry) -> Result<()> {
        let fields = self.fields.read();
        for (field_name, field_entry) in &entry.fields {
            let field = fields.get(field_name).ok_or_else(|| {
                SarissaError::not_found(format!(
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
        version: u64,
        fields_data: &HashMap<String, StoredVector>,
    ) -> Result<()> {
        let fields = self.fields.read();
        for (field_name, stored_vector) in fields_data {
            let field = fields.get(field_name).ok_or_else(|| {
                SarissaError::not_found(format!("vector field '{field_name}' is not registered"))
            })?;
            field
                .runtime
                .writer()
                .add_stored_vector(doc_id, stored_vector, version)?;
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

        self.load_document_snapshot(storage.clone())?;
        self.load_collection_manifest(storage.clone())?;
        // Instantiate fields after manifest load so that persisted implicit fields are registered
        self.instantiate_configured_fields()?;
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

        // Legacy format with FieldVectors (multiple vectors per field).
        #[derive(serde::Deserialize)]
        struct LegacyFieldVectors {
            #[serde(default)]
            vectors: Vec<StoredVector>,
        }

        #[derive(serde::Deserialize)]
        struct LegacySnapshotDocument {
            doc_id: u64,
            #[serde(default)]
            fields: HashMap<String, LegacyFieldVectors>,
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
                    .map(|legacy| {
                        // Convert FieldVectors to StoredVector (take first vector only).
                        let fields = legacy
                            .fields
                            .into_iter()
                            .filter_map(|(name, fv)| {
                                fv.vectors.into_iter().next().map(|v| (name, v))
                            })
                            .collect();
                        SnapshotDocument {
                            doc_id: legacy.doc_id,
                            document: DocumentVector {
                                fields,
                                metadata: legacy.metadata,
                            },
                        }
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
            return Err(SarissaError::invalid_config(format!(
                "collection manifest version mismatch: expected {}, found {}",
                COLLECTION_MANIFEST_VERSION, manifest.version
            )));
        }

        let snapshot_seq = self.snapshot_wal_seq.load(Ordering::SeqCst);
        if manifest.snapshot_wal_seq != snapshot_seq {
            return Err(SarissaError::invalid_config(format!(
                "collection manifest snapshot sequence {} does not match persisted snapshot {}",
                manifest.snapshot_wal_seq, snapshot_seq
            )));
        }

        if manifest.wal_last_seq < manifest.snapshot_wal_seq {
            return Err(SarissaError::invalid_config(
                "collection manifest WAL sequence regressed",
            ));
        }

        if !manifest.field_configs.is_empty() {
            *self.field_configs.write() = manifest.field_configs.clone();
        }

        Ok(())
    }

    fn replay_wal_into_fields(&self) -> Result<()> {
        let mut documents = self.documents.read().clone();
        self.apply_documents_to_fields(&documents)?;

        // Read records (this also updates internal next_seq based on finding)
        let mut records = self.wal.read_all()?;

        let mut applied_seq = self.snapshot_wal_seq.load(Ordering::SeqCst);
        let start_seq = applied_seq.saturating_add(1);

        // Ensure WAL manager knows about the sequence number from snapshot
        let current_wal_seq = self.wal.last_seq();
        if applied_seq > current_wal_seq {
            self.wal.set_next_seq(applied_seq + 1);
        }

        if records.is_empty() {
            // If WAL is empty but we have documents, ensure they are in sync?
            // Assuming snapshot is source of truth if WAL is empty.
            *self.documents.write() = documents;
            return Ok(());
        }

        records.sort_by(|a, b| a.seq.cmp(&b.seq));
        for record in records.into_iter() {
            if record.seq < start_seq {
                continue;
            }
            applied_seq = record.seq;
            match record.entry {
                WalEntry::Upsert { doc_id, document } => {
                    // logic..
                    if document.fields.is_empty() {
                        documents.remove(&doc_id);
                        continue;
                    }
                    if let Some(entry) = self.registry.get(doc_id) {
                        self.apply_field_updates(doc_id, entry.version, &document.fields)?;
                    }
                    documents.insert(doc_id, document);
                }
                WalEntry::Delete { doc_id } => {
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

    pub fn flush_vectors(&self) -> Result<()> {
        let fields = self.fields.read();
        for field_entry in fields.values() {
            field_entry.runtime.writer().flush()?;
        }
        Ok(())
    }

    fn persist_state(&self) -> Result<()> {
        self.persist_registry_snapshot()?;
        self.persist_document_snapshot()?;
        // WAL is self-persisting
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
            field_configs: self.field_configs.read().clone(),
        };
        let serialized = serde_json::to_vec(&manifest)?;
        self.write_atomic(storage, COLLECTION_MANIFEST_FILE, &serialized)
    }

    fn maybe_compact_wal(&self) -> Result<()> {
        // Simple compaction strategy: if WAL file is too large?
        // For now, if we just persisted a snapshot, we can truncate the WAL safely.
        // But maybe_compact_wal is called after persist_state.
        // So we can unconditionally truncate?
        // To be safe, let's only truncate if we have many ops.
        // But tracking op count requires atomic counter.
        // Let's just truncate after every persist for now if easy, or use random sampling?
        // Actually, let's leave it as a TODO or implementing a simple counter if valuable.
        // Given the requirement to "Unify WAL", correct behavior is robust conservation logic.
        // If snapshot is saved, WAL *can* be truncated.
        // So just truncate.
        self.compact_wal()
    }

    fn compact_wal(&self) -> Result<()> {
        self.wal.truncate()
    }

    /// Upsert a document (internal implementation).
    fn upsert_document_internal(&self, doc_id: u64, document: DocumentVector) -> Result<u64> {
        self.validate_document_fields(&document)?;
        let entries = self::registry::build_field_entries(&document);
        let version = self
            .registry
            .upsert(doc_id, &entries, document.metadata.clone())?;

        // Update fields (in memory/segments)
        if let Err(err) = self.apply_field_updates(doc_id, version, &document.fields) {
            let _ = self.registry.delete(doc_id);
            return Err(err);
        }

        // Then update documents map and WAL
        self.documents.write().insert(doc_id, document.clone());
        self.wal.append(&WalEntry::Upsert { doc_id, document })?;

        self.persist_state()?;
        self.bump_next_doc_id(doc_id);
        Ok(version)
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

    // =========================================================================
    // Public API
    // =========================================================================

    /// Get the index configuration.
    pub fn config(&self) -> &VectorIndexConfig {
        self.config.as_ref()
    }

    /// Get the embedder for this engine.
    pub fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(self.config.get_embedder())
    }

    /// Add a document to the collection.
    ///
    /// Returns the assigned document ID.
    pub fn add_vectors(&self, doc: DocumentVector) -> Result<u64> {
        let doc_id = self.next_doc_id.fetch_add(1, Ordering::SeqCst);
        self.upsert_document_internal(doc_id, doc)?;
        Ok(doc_id)
    }

    /// Add a document from payload (will be embedded if configured).
    ///
    /// Returns the assigned document ID.
    pub fn add_payloads(&self, payload: DocumentPayload) -> Result<u64> {
        let doc_id = self.next_doc_id.fetch_add(1, Ordering::SeqCst);
        self.upsert_document_payload(doc_id, payload)?;
        Ok(doc_id)
    }

    /// Add multiple vectors with automatically assigned doc_ids.
    pub fn add_vectors_batch(
        &self,
        docs: impl IntoIterator<Item = DocumentVector>,
    ) -> Result<Vec<u64>> {
        docs.into_iter().map(|doc| self.add_vectors(doc)).collect()
    }

    /// Add multiple payloads with automatically assigned doc_ids.
    pub fn add_payloads_batch(
        &self,
        payloads: impl IntoIterator<Item = DocumentPayload>,
    ) -> Result<Vec<u64>> {
        payloads
            .into_iter()
            .map(|payload| self.add_payloads(payload))
            .collect()
    }

    /// Upsert a document with a specific document ID.
    pub fn upsert_vectors(&self, doc_id: u64, doc: DocumentVector) -> Result<()> {
        self.upsert_document_internal(doc_id, doc)?;
        Ok(())
    }

    /// Upsert a document from payload (will be embedded if configured).
    pub fn upsert_payloads(&self, doc_id: u64, payload: DocumentPayload) -> Result<()> {
        self.upsert_document_payload(doc_id, payload)
    }

    /// Upsert a document from payload (internal helper).
    fn upsert_document_payload(&self, doc_id: u64, payload: DocumentPayload) -> Result<()> {
        let document = self.embed_document_payload_internal(doc_id, payload)?;
        self.upsert_document_internal(doc_id, document)?;
        Ok(())
    }

    /// Delete a document by ID.
    pub fn delete_vectors(&self, doc_id: u64) -> Result<()> {
        let entry = self
            .registry
            .get(doc_id)
            .ok_or_else(|| SarissaError::not_found(format!("doc_id {doc_id}")))?;
        self.delete_fields_for_entry(doc_id, &entry)?;

        self.registry.delete(doc_id)?;
        self.wal.append(&WalEntry::Delete { doc_id })?;
        self.documents.write().remove(&doc_id);
        // WAL is durable on append, so we don't need full persist_state here
        // But we might want to update snapshots periodically? For now, keep it simple.
        self.persist_state()?; // Still need to update registry/doc snapshots if we want them in sync
        // self.maybe_compact_wal() // Refactor later
        Ok(())
    }

    /// Embed a document payload into vectors.
    pub fn embed_document_payload(&self, payload: DocumentPayload) -> Result<DocumentVector> {
        self.embed_document_payload_internal(0, payload)
    }

    /// Embed a payload for query.
    pub fn embed_query_payload(&self, field_name: &str, payload: Payload) -> Result<QueryVector> {
        let vector = self.embed_payload(field_name, payload)?;
        Ok(QueryVector {
            vector,
            weight: 1.0,
            fields: None, // Will be set by caller if needed
        })
    }

    /// Register an external field implementation.
    pub fn register_field(&self, _name: String, field: Box<dyn VectorField>) -> Result<()> {
        let field_arc: Arc<dyn VectorField> = Arc::from(field);
        self.register_field_impl(field_arc)
    }

    /// Get statistics for a specific field.
    pub fn field_stats(&self, field_name: &str) -> Result<VectorFieldStats> {
        let fields = self.fields.read();
        let field = fields.get(field_name).ok_or_else(|| {
            SarissaError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;
        field.runtime.reader().stats()
    }

    /// Replace the reader for a specific field.
    pub fn replace_field_reader(
        &self,
        field_name: &str,
        reader: Box<dyn VectorFieldReader>,
    ) -> Result<()> {
        let fields = self.fields.read();
        let field = fields.get(field_name).ok_or_else(|| {
            SarissaError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;
        let reader_arc: Arc<dyn VectorFieldReader> = Arc::from(reader);
        field.runtime.replace_reader(reader_arc);
        Ok(())
    }

    /// Reset the reader for a specific field to default.
    pub fn reset_field_reader(&self, field_name: &str) -> Result<()> {
        let fields = self.fields.read();
        let field = fields.get(field_name).ok_or_else(|| {
            SarissaError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;
        field.runtime.reset_reader();
        Ok(())
    }

    /// Materialize the delegate reader for a field (build persistent index).
    pub fn materialize_delegate_reader(&self, field_name: &str) -> Result<()> {
        let fields = self.fields.read();
        let handle = fields.get(field_name).ok_or_else(|| {
            SarissaError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;

        let in_memory = handle
            .field
            .as_any()
            .downcast_ref::<InMemoryVectorField>()
            .ok_or_else(|| {
                SarissaError::InvalidOperation(format!(
                    "field '{field_name}' does not support delegate materialization"
                ))
            })?;

        let vectors = in_memory.vector_tuples();
        let config = in_memory.config().clone();
        drop(fields);

        self.write_field_delegate_index(field_name, &config, vectors)?;
        let reader = self.load_delegate_reader(field_name, &config)?;

        let fields = self.fields.read();
        let handle = fields.get(field_name).ok_or_else(|| {
            SarissaError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;
        handle.runtime.replace_reader(reader);
        Ok(())
    }

    /// Create a searcher for this engine.
    pub fn searcher(&self) -> Result<Box<dyn crate::vector::search::searcher::VectorSearcher>> {
        Ok(Box::new(VectorEngineSearcher::from_engine_ref(self)))
    }

    /// Execute a search query.
    pub fn search(&self, mut request: VectorSearchRequest) -> Result<VectorSearchResults> {
        // Embed query_payloads and add to query_vectors.
        for query_payload in std::mem::take(&mut request.query_payloads) {
            let mut qv = self.embed_query_payload(&query_payload.field, query_payload.payload)?;
            qv.weight = query_payload.weight;
            request.query_vectors.push(qv);
        }

        let searcher = self.searcher()?;
        searcher.search(&request)
    }

    /// Count documents matching the search criteria.
    pub fn count(&self, mut request: VectorSearchRequest) -> Result<u64> {
        // Embed query_payloads and add to query_vectors.
        for query_payload in std::mem::take(&mut request.query_payloads) {
            let mut qv = self.embed_query_payload(&query_payload.field, query_payload.payload)?;
            qv.weight = query_payload.weight;
            request.query_vectors.push(qv);
        }

        let searcher = self.searcher()?;
        searcher.count(&request)
    }

    /// Commit pending changes (persist state).
    pub fn commit(&self) -> Result<()> {
        self.persist_state()
    }

    /// Get collection statistics.
    pub fn stats(&self) -> Result<VectorStats> {
        let fields = self.fields.read();
        let mut field_stats = HashMap::with_capacity(fields.len());
        for (name, field) in fields.iter() {
            let stats = field.runtime.reader().stats()?;
            field_stats.insert(name.clone(), stats);
        }

        Ok(VectorStats {
            document_count: self.registry.document_count(),
            fields: field_stats,
        })
    }

    /// Get the storage backend.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// Close the collection and release resources.
    pub fn close(&self) -> Result<()> {
        self.closed.store(1, Ordering::SeqCst);
        Ok(())
    }

    /// Check if the collection is closed.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst) == 1
    }

    /// Optimize the vector index.
    ///
    /// This triggers optimization (e.g., segment merging, index rebuild) for all registered fields.
    pub fn optimize(&self) -> Result<()> {
        let fields = self.fields.read();

        for (_field_name, field_entry) in fields.iter() {
            field_entry.field.optimize()?;
        }

        Ok(())
    }
}

/// Searcher implementation for [`VectorEngine`].
#[derive(Debug)]
pub struct VectorEngineSearcher {
    config: Arc<VectorIndexConfig>,
    fields: Arc<RwLock<HashMap<String, FieldHandle>>>,
    registry: Arc<DocumentVectorRegistry>,
    documents: Arc<RwLock<HashMap<u64, DocumentVector>>>,
}

impl VectorEngineSearcher {
    /// Create a new searcher from an engine reference.
    pub fn from_engine_ref(engine: &VectorEngine) -> Self {
        Self {
            config: Arc::clone(&engine.config),
            fields: Arc::clone(&engine.fields),
            registry: Arc::clone(&engine.registry),
            documents: Arc::clone(&engine.documents),
        }
    }

    /// Resolve which fields to search based on the request.
    fn resolve_fields(&self, request: &VectorSearchRequest) -> Result<Vec<String>> {
        match &request.fields {
            Some(selectors) => self.apply_field_selectors(selectors),
            None => Ok(self.config.default_fields.clone()),
        }
    }

    /// Apply field selectors to determine which fields to search.
    fn apply_field_selectors(&self, selectors: &[FieldSelector]) -> Result<Vec<String>> {
        let fields = self.fields.read();
        let mut result = Vec::new();

        for selector in selectors {
            match selector {
                FieldSelector::Exact(name) => {
                    if fields.contains_key(name) {
                        if !result.contains(name) {
                            result.push(name.clone());
                        }
                    } else {
                        return Err(SarissaError::not_found(format!(
                            "vector field '{name}' is not registered",
                        )));
                    }
                }
                FieldSelector::Prefix(prefix) => {
                    for field_name in fields.keys() {
                        if field_name.starts_with(prefix) && !result.contains(field_name) {
                            result.push(field_name.clone());
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Build filter matches based on request filters.
    fn build_filter_matches(
        &self,
        request: &VectorSearchRequest,
        target_fields: &[String],
    ) -> Option<RegistryFilterMatches> {
        request
            .filter
            .as_ref()
            .filter(|filter| !filter.is_empty())
            .map(|filter| self.registry.filter_matches(filter, target_fields))
    }

    /// Get the scaled field limit based on overfetch factor.
    fn scaled_field_limit(&self, limit: usize, overfetch: f32) -> usize {
        ((limit as f32) * overfetch).ceil() as usize
    }

    /// Get query vectors that match a specific field.
    fn query_vectors_for_field(
        &self,
        field_name: &str,
        _config: &VectorFieldConfig,
        request: &VectorSearchRequest,
    ) -> Vec<QueryVector> {
        request
            .query_vectors
            .iter()
            .filter(|candidate| {
                if let Some(fields) = &candidate.fields {
                    if !fields.contains(&field_name.to_string()) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect()
    }

    /// Merge field hits into document hits.
    fn merge_field_hits(
        &self,
        doc_hits: &mut HashMap<u64, VectorHit>,
        hits: Vec<FieldHit>,
        field_weight: f32,
        score_mode: VectorScoreMode,
        filter_matches: Option<&RegistryFilterMatches>,
    ) -> Result<()> {
        let doc_ids: Vec<u64> = hits.iter().map(|h| h.doc_id).collect();
        let existing_ids = self.registry.filter_existing(&doc_ids);

        for hit in hits {
            if !existing_ids.contains(&hit.doc_id) {
                continue;
            }

            if let Some(matches) = filter_matches {
                if !matches.contains_doc(hit.doc_id) {
                    continue;
                }
                if !matches.field_allowed(hit.doc_id, &hit.field) {
                    continue;
                }
            }

            let weighted_score = hit.score * field_weight;
            let entry = doc_hits.entry(hit.doc_id).or_insert_with(|| VectorHit {
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
                    return Err(SarissaError::invalid_argument(
                        "VectorScoreMode::LateInteraction is not supported yet",
                    ));
                }
            }
            entry.field_hits.push(hit);
        }

        Ok(())
    }
}

impl crate::vector::search::searcher::VectorSearcher for VectorEngineSearcher {
    fn search(&self, request: &VectorSearchRequest) -> Result<VectorSearchResults> {
        if request.query_vectors.is_empty() {
            return Err(SarissaError::invalid_argument(
                "VectorSearchRequest requires at least one query vector",
            ));
        }

        if request.limit == 0 {
            return Ok(VectorSearchResults::default());
        }

        if request.overfetch < 1.0 {
            return Err(SarissaError::invalid_argument(
                "VectorSearchRequest overfetch must be >= 1.0",
            ));
        }

        if matches!(request.score_mode, VectorScoreMode::LateInteraction) {
            return Err(SarissaError::invalid_argument(
                "VectorScoreMode::LateInteraction is not supported yet",
            ));
        }

        let target_fields = self.resolve_fields(request)?;
        let filter_matches = self.build_filter_matches(request, &target_fields);
        if filter_matches
            .as_ref()
            .is_some_and(|matches| matches.is_empty())
        {
            return Ok(VectorSearchResults::default());
        }

        let mut doc_hits: HashMap<u64, VectorHit> = HashMap::new();
        let mut fields_with_queries = 0_usize;
        let field_limit = self.scaled_field_limit(request.limit, request.overfetch);

        let fields = self.fields.read();
        for field_name in target_fields {
            let field = fields
                .get(&field_name)
                .ok_or_else(|| SarissaError::not_found(format!("vector field '{field_name}'")))?;
            let matching_vectors =
                self.query_vectors_for_field(&field_name, field.field.config(), request);
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
                request.score_mode,
                filter_matches.as_ref(),
            )?;
        }
        drop(fields);

        if fields_with_queries == 0 {
            return Err(SarissaError::invalid_argument(
                "no query vectors matched the requested fields",
            ));
        }

        let mut hits: Vec<VectorHit> = doc_hits.into_values().collect();

        if request.min_score > 0.0 {
            hits.retain(|hit| hit.score >= request.min_score);
        }

        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(CmpOrdering::Equal));
        if hits.len() > request.limit {
            hits.truncate(request.limit);
        }

        Ok(VectorSearchResults { hits })
    }

    fn count(&self, request: &VectorSearchRequest) -> Result<u64> {
        if request.query_vectors.is_empty() {
            let documents = self.documents.read();
            return Ok(documents.len() as u64);
        }

        let count_request = VectorSearchRequest {
            query_vectors: request.query_vectors.clone(),
            query_payloads: request.query_payloads.clone(),
            fields: request.fields.clone(),
            limit: usize::MAX,
            score_mode: request.score_mode,
            overfetch: 1.0,
            filter: request.filter.clone(),
            min_score: request.min_score,
        };

        let results = self.search(&count_request)?;
        Ok(results.hits.len() as u64)
    }
}

fn build_field_entries(document: &DocumentVector) -> Vec<FieldEntry> {
    document
        .fields
        .iter()
        .map(|(name, vector)| FieldEntry {
            field_name: name.clone(),
            version: 0,
            vector_count: 1, // Always 1 in flattened model
            weight: vector.weight,
            metadata: vector.attributes.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::DistanceMetric;
    use crate::vector::core::document::StoredVector;
    use crate::vector::engine::config::{VectorFieldConfig, VectorIndexKind};
    use crate::vector::engine::request::QueryVector;
    use std::collections::HashMap;

    fn sample_config() -> VectorIndexConfig {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            metadata: HashMap::new(),
            base_weight: 1.0,
        };
        use crate::embedding::precomputed::PrecomputedEmbedder;

        VectorIndexConfig {
            fields: HashMap::from([("body".into(), field_config)]),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
            default_distance: DistanceMetric::Cosine,
            default_dimension: None,
            default_index_kind: VectorIndexKind::Flat,
            default_base_weight: 1.0,
            implicit_schema: false,
            embedder: Arc::new(PrecomputedEmbedder::new()),
        }
    }

    fn sample_query(limit: usize) -> VectorSearchRequest {
        let mut query = VectorSearchRequest::default();
        query.limit = limit;
        query.query_vectors.push(QueryVector {
            vector: StoredVector::new(Arc::<[f32]>::from([1.0, 0.0, 0.0])),
            weight: 1.0,
            fields: None,
        });
        query
    }

    fn create_engine(config: VectorIndexConfig, storage: Arc<dyn Storage>) -> VectorEngine {
        VectorEngine::new(storage, config).expect("engine")
    }

    #[test]
    fn engine_creation_works() {
        let config = sample_config();
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = create_engine(config, storage);

        let stats = engine.stats().expect("stats");
        assert_eq!(stats.document_count, 0);
        assert!(stats.fields.contains_key("body"));
    }

    #[test]
    fn engine_add_and_search() {
        let config = sample_config();
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = create_engine(config, storage);

        let mut doc = DocumentVector::new();
        doc.set_field(
            "body",
            StoredVector::new(Arc::<[f32]>::from([1.0, 0.0, 0.0])),
        );

        let doc_id = engine.add_vectors(doc).expect("add vectors");

        let stats = engine.stats().expect("stats");
        assert_eq!(stats.document_count, 1);

        let results = engine.search(sample_query(5)).expect("search");
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, doc_id);
    }

    #[test]
    fn engine_upsert_and_delete() {
        let config = sample_config();
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = create_engine(config, storage);

        let mut doc = DocumentVector::new();
        doc.set_field(
            "body",
            StoredVector::new(Arc::<[f32]>::from([0.5, 0.5, 0.0])),
        );

        engine.upsert_vectors(42, doc).expect("upsert");
        let stats = engine.stats().expect("stats");
        assert_eq!(stats.document_count, 1);

        engine.delete_vectors(42).expect("delete");
        let stats = engine.stats().expect("stats");
        assert_eq!(stats.document_count, 0);
    }

    #[test]
    fn engine_persistence_across_instances() {
        let config = sample_config();
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let engine = create_engine(config.clone(), storage.clone());
            let mut doc = DocumentVector::new();
            doc.set_field(
                "body",
                StoredVector::new(Arc::<[f32]>::from([1.0, 0.0, 0.0])),
            );
            engine.upsert_vectors(10, doc).expect("upsert");
        }

        let engine = create_engine(config, storage);
        let stats = engine.stats().expect("stats");
        assert_eq!(stats.document_count, 1);

        let results = engine.search(sample_query(5)).expect("search");
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, 10);
    }
}
