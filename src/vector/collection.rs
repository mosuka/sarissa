//! Vector collection module.
//!
//! This module provides `VectorCollection`, a document-centric vector
//! collection that manages multiple vector fields with different configurations.
//!
//! This is analogous to `InvertedIndex` in the lexical module, providing
//! the concrete implementation that `VectorEngine` uses internally.

use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};

use crate::embedding::embedder::{EmbedInput, Embedder};
use crate::embedding::per_field::PerFieldEmbedder;
use crate::error::{PlatypusError, Result};
use crate::storage::Storage;
use crate::storage::prefixed::PrefixedStorage;
use crate::vector::core::document::{
    DocumentPayload, DocumentVector, Payload, PayloadSource, StoredVector, VectorType,
};
use crate::vector::core::vector::Vector;
use crate::vector::engine::config::{VectorFieldConfig, VectorIndexConfig, VectorIndexKind};
use crate::vector::engine::embedder::{EmbedderExecutor, VectorEmbedderRegistry};
use crate::vector::engine::filter::RegistryFilterMatches;
use crate::vector::engine::memory::{FieldHandle, FieldRuntime, InMemoryVectorField};
use crate::vector::engine::registry::{DocumentEntry, DocumentVectorRegistry, FieldEntry};
use crate::vector::engine::request::{
    FieldSelector, QueryVector, VectorScoreMode, VectorSearchRequest,
};
use crate::vector::engine::response::{VectorHit, VectorSearchResults, VectorStats};
use crate::vector::engine::snapshot::{
    COLLECTION_MANIFEST_FILE, COLLECTION_MANIFEST_VERSION, CollectionManifest,
    DOCUMENT_SNAPSHOT_FILE, DOCUMENT_SNAPSHOT_TEMP_FILE, DocumentSnapshot, FIELD_INDEX_BASENAME,
    REGISTRY_NAMESPACE, REGISTRY_SNAPSHOT_FILE, REGISTRY_WAL_FILE, SnapshotDocument,
};
use crate::vector::engine::wal::{VectorWal, WAL_COMPACTION_THRESHOLD, WalPayload, WalRecord};
use crate::vector::field::{
    FieldHit, FieldSearchInput, VectorField, VectorFieldReader, VectorFieldStats, VectorFieldWriter,
};
use crate::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
use crate::vector::index::field::{AdapterBackedVectorField, LegacyVectorFieldWriter};
use crate::vector::index::flat::{
    field_reader::FlatFieldReader, reader::FlatVectorIndexReader, writer::FlatIndexWriter,
};
use crate::vector::index::hnsw::{
    field_reader::HnswFieldReader, reader::HnswIndexReader, writer::HnswIndexWriter,
};
use crate::vector::index::ivf::{
    field_reader::IvfFieldReader, reader::IvfIndexReader, writer::IvfIndexWriter,
};
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// A document-centric vector collection with multiple vector fields.
///
/// This is the primary vector collection implementation, supporting:
/// - Multiple vector fields with different configurations
/// - Document-level and field-level metadata filtering
/// - WAL-based persistence and recovery
/// - Configurable embedder integration
///
/// This is analogous to `InvertedIndex` in the lexical module, providing
/// the concrete implementation that `VectorEngine` uses internally.
pub struct VectorCollection {
    config: Arc<VectorIndexConfig>,
    field_configs: Arc<RwLock<HashMap<String, VectorFieldConfig>>>,
    fields: Arc<RwLock<HashMap<String, FieldHandle>>>,
    registry: Arc<DocumentVectorRegistry>,
    embedder_registry: Arc<VectorEmbedderRegistry>,
    embedder_executor: Mutex<Option<Arc<EmbedderExecutor>>>,
    wal: Arc<VectorWal>,
    storage: Arc<dyn Storage>,
    documents: Arc<RwLock<HashMap<u64, DocumentVector>>>,
    snapshot_wal_seq: AtomicU64,
    next_doc_id: AtomicU64,
    closed: AtomicU64, // 0 = open, 1 = closed
}

impl fmt::Debug for VectorCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorCollection")
            .field("config", &self.config)
            .field("field_count", &self.fields.read().len())
            .finish()
    }
}

impl VectorCollection {
    /// Create a new multi-field vector collection.
    ///
    /// The embedder instances are automatically registered from the
    /// `VectorIndexConfig.embedder` field.
    pub fn new(
        config: VectorIndexConfig,
        storage: Arc<dyn Storage>,
        initial_doc_id: Option<u64>,
    ) -> Result<Self> {
        let embedder_registry = Arc::new(VectorEmbedderRegistry::new());
        let registry = Arc::new(DocumentVectorRegistry::default());
        let field_configs = Arc::new(RwLock::new(config.fields.clone()));

        // Store the embedder from config before moving config into Arc
        let config_embedder = config.embedder.clone();

        let mut collection = Self {
            config: Arc::new(config),
            field_configs: field_configs.clone(),
            fields: Arc::new(RwLock::new(HashMap::new())),
            registry,
            embedder_registry,
            embedder_executor: Mutex::new(None),
            wal: Arc::new(VectorWal::default()),
            storage,
            documents: Arc::new(RwLock::new(HashMap::new())),
            snapshot_wal_seq: AtomicU64::new(0),
            next_doc_id: AtomicU64::new(initial_doc_id.unwrap_or(0)),
            closed: AtomicU64::new(0),
        };
        collection.load_persisted_state()?;

        // Register embedder instances from the config (after fields are instantiated)
        collection.register_embedder_from_config(config_embedder)?;

        Ok(collection)
    }

    /// Register embedder instances from the Embedder trait object.
    ///
    /// This extracts embedders for each configured field and registers them
    /// with the embedder registry.
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
            return Err(PlatypusError::invalid_config(format!(
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
            return Err(PlatypusError::invalid_argument(format!(
                "vector field '{field_name}' is not registered"
            )));
        }

        let field_config = self.build_field_config_for_payload(field_name, payload)?;

        // Persist in config cache
        self.field_configs
            .write()
            .insert(field_name.to_string(), field_config.clone());

        // Build field runtime
        let delegate = self.build_delegate_writer(field_name, &field_config)?;
        let field = Arc::new(InMemoryVectorField::new(
            field_name.to_string(),
            field_config,
            delegate,
        )?);
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
        let vector_type = match payload.vector_type.clone() {
            VectorType::Generic => self.config.default_vector_type.clone(),
            vt => vt,
        };

        let (dimension, source_tag) = match &payload.source {
            PayloadSource::Text { .. }
            | PayloadSource::Bytes { .. }
            | PayloadSource::Uri { .. } => {
                let dim = self
                    .config
                    .default_dimension
                    .ok_or_else(|| PlatypusError::invalid_config(
                        "implicit schema requires default_dimension to be set",
                    ))?;
                (dim, field_name.to_string())
            }
            PayloadSource::Vector { data, source_tag } => (data.len(), source_tag.clone()),
        };

        if dimension == 0 {
            return Err(PlatypusError::invalid_config(format!(
                "cannot register field '{field_name}' with zero dimension"
            )));
        }

        Ok(VectorFieldConfig {
            dimension,
            distance: self.config.default_distance,
            index: self.config.default_index_kind,
            source_tag,
            vector_type,
            base_weight: self.config.default_base_weight,
        })
    }

    fn resolve_embedder_for_field(&self, field_name: &str) -> Arc<dyn Embedder> {
        let embedder = self.config.embedder.clone();
        if let Some(per_field) = embedder.as_any().downcast_ref::<PerFieldEmbedder>() {
            per_field.get_embedder(field_name).clone()
        } else {
            embedder
        }
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
    ///
    /// This is the new flattened embedding API where 1 payload produces 1 vector.
    fn embed_payload(&self, field_name: &str, payload: Payload) -> Result<StoredVector> {
        let fields = self.fields.read();
        let handle = fields.get(field_name).ok_or_else(|| {
            PlatypusError::invalid_argument(format!(
                "vector field '{field_name}' is not registered"
            ))
        })?;
        let field_config = handle.field.config().clone();
        drop(fields);

        let Payload {
            source,
            vector_type: payload_vector_type,
        } = payload;

        let vector_type = match payload_vector_type {
            VectorType::Generic => field_config.vector_type.clone(),
            explicit => explicit,
        };

        match source {
            PayloadSource::Text { value } => {
                let executor = self.ensure_embedder_executor()?;
                let embedder = self.embedder_registry.resolve(field_name)?;

                if !embedder.supports_text() {
                    return Err(PlatypusError::invalid_config(format!(
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
                    return Err(PlatypusError::InvalidOperation(format!(
                        "embedder '{}' produced invalid values for field '{}'",
                        embedder_name_owned, field_name
                    )));
                }
                let mut stored: StoredVector = vector.into();
                stored.source_tag = field_config.source_tag.clone();
                stored.vector_type = vector_type;
                Ok(stored)
            }
            PayloadSource::Bytes { bytes, mime } => {
                let executor = self.ensure_embedder_executor()?;
                let embedder = self.embedder_registry.resolve(field_name)?;

                if !embedder.supports_image() {
                    return Err(PlatypusError::invalid_config(format!(
                        "embedder '{}' does not support image embedding",
                        field_name
                    )));
                }

                let embedder_name_owned = field_name.to_string();
                let payload_bytes = bytes.clone();
                let mime_hint = mime.clone();
                let vector = executor.run(async move {
                    embedder
                        .embed(&EmbedInput::ImageBytes(
                            &payload_bytes,
                            mime_hint.as_deref(),
                        ))
                        .await
                })?;
                vector.validate_dimension(field_config.dimension)?;
                if !vector.is_valid() {
                    return Err(PlatypusError::InvalidOperation(format!(
                        "embedder '{}' produced invalid values for field '{}'",
                        embedder_name_owned, field_name
                    )));
                }
                let mut stored: StoredVector = vector.into();
                stored.source_tag = field_config.source_tag.clone();
                stored.vector_type = vector_type;
                Ok(stored)
            }
            PayloadSource::Uri { uri, media_hint: _ } => {
                let executor = self.ensure_embedder_executor()?;
                let embedder = self.embedder_registry.resolve(field_name)?;

                if !embedder.supports_image() {
                    return Err(PlatypusError::invalid_config(format!(
                        "embedder '{}' does not support image embedding",
                        field_name
                    )));
                }

                let embedder_name_owned = field_name.to_string();
                let uri_value = uri;
                let vector = executor
                    .run(async move { embedder.embed(&EmbedInput::ImageUri(&uri_value)).await })?;
                vector.validate_dimension(field_config.dimension)?;
                if !vector.is_valid() {
                    return Err(PlatypusError::InvalidOperation(format!(
                        "embedder '{}' produced invalid values for field '{}'",
                        embedder_name_owned, field_name
                    )));
                }
                let mut stored: StoredVector = vector.into();
                stored.source_tag = field_config.source_tag.clone();
                stored.vector_type = vector_type;
                Ok(stored)
            }
            PayloadSource::Vector { data, source_tag } => {
                if source_tag != field_config.source_tag {
                    return Err(PlatypusError::invalid_argument(format!(
                        "vector field '{field_name}' only accepts source_tag '{}' but got '{}'",
                        field_config.source_tag, source_tag
                    )));
                }
                if data.len() != field_config.dimension {
                    return Err(PlatypusError::invalid_argument(format!(
                        "vector field '{field_name}' expects dimension {} but received {}",
                        field_config.dimension,
                        data.len()
                    )));
                }
                Ok(StoredVector::new(
                    data.clone(),
                    field_config.source_tag.clone(),
                    vector_type,
                ))
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

    fn upsert_document_internal(&self, doc_id: u64, document: DocumentVector) -> Result<u64> {
        self.validate_document_fields(&document)?;
        let entries = build_field_entries(&document);
        let version = self
            .registry
            .upsert(doc_id, &entries, document.metadata.clone())?;

        if let Err(err) = self.apply_field_updates(doc_id, version, &document.fields) {
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
            let delegate = self.build_delegate_writer(&name, &config)?;
            let field = Arc::new(InMemoryVectorField::new(name, config, delegate)?);
            self.register_field_impl(field)?;
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
                Arc::new(FlatFieldReader::new(field_name.to_string(), reader))
            }
            VectorIndexKind::Hnsw => {
                let reader = Arc::new(HnswIndexReader::load(
                    storage.clone(),
                    FIELD_INDEX_BASENAME,
                    config.distance,
                )?);
                Arc::new(HnswFieldReader::new(field_name.to_string(), reader))
            }
            VectorIndexKind::Ivf => {
                let reader = Arc::new(IvfIndexReader::load(
                    storage.clone(),
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
                return Err(PlatypusError::invalid_argument(format!(
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
        version: u64,
        fields_data: &HashMap<String, StoredVector>,
    ) -> Result<()> {
        let fields = self.fields.read();
        for (field_name, stored_vector) in fields_data {
            let field = fields.get(field_name).ok_or_else(|| {
                PlatypusError::not_found(format!("vector field '{field_name}' is not registered"))
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

        if !manifest.field_configs.is_empty() {
            *self.field_configs.write() = manifest.field_configs.clone();
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
            field_configs: self.field_configs.read().clone(),
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

// =========================================================================
// Public API
// =========================================================================

impl VectorCollection {
    /// Get the collection configuration.
    pub fn config(&self) -> &VectorIndexConfig {
        self.config.as_ref()
    }

    /// Add a document to the collection.
    ///
    /// Returns the assigned document ID.
    pub fn add_document(&self, doc: DocumentVector) -> Result<u64> {
        let doc_id = self.next_doc_id.fetch_add(1, Ordering::SeqCst);
        self.upsert_document(doc_id, doc)?;
        Ok(doc_id)
    }

    /// Add a document from payload (will be embedded if configured).
    ///
    /// Returns the assigned document ID.
    pub fn add_document_payload(&self, payload: DocumentPayload) -> Result<u64> {
        let doc_id = self.next_doc_id.fetch_add(1, Ordering::SeqCst);
        self.upsert_document_payload(doc_id, payload)?;
        Ok(doc_id)
    }

    /// Upsert a document with a specific document ID.
    pub fn upsert_document(&self, doc_id: u64, doc: DocumentVector) -> Result<()> {
        self.upsert_document_internal(doc_id, doc)?;
        Ok(())
    }

    /// Upsert a document from payload (will be embedded if configured).
    pub fn upsert_document_payload(&self, doc_id: u64, payload: DocumentPayload) -> Result<()> {
        let document = self.embed_document_payload_internal(doc_id, payload)?;
        self.upsert_document_internal(doc_id, document)?;
        Ok(())
    }

    /// Delete a document by ID.
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

    /// Embed a document payload into vectors.
    pub fn embed_document_payload(&self, payload: DocumentPayload) -> Result<DocumentVector> {
        self.embed_document_payload_internal(0, payload)
    }

    /// Embed a payload for query.
    ///
    /// Returns a `QueryVector` with the embedded vector and default weight 1.0.
    pub fn embed_query_payload(&self, field_name: &str, payload: Payload) -> Result<QueryVector> {
        let vector = self.embed_payload(field_name, payload)?;
        Ok(QueryVector {
            vector,
            weight: 1.0,
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
            PlatypusError::not_found(format!("vector field '{field_name}' is not registered"))
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
            PlatypusError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;
        let reader_arc: Arc<dyn VectorFieldReader> = Arc::from(reader);
        field.runtime.replace_reader(reader_arc);
        Ok(())
    }

    /// Reset the reader for a specific field to default.
    pub fn reset_field_reader(&self, field_name: &str) -> Result<()> {
        let fields = self.fields.read();
        let field = fields.get(field_name).ok_or_else(|| {
            PlatypusError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;
        field.runtime.reset_reader();
        Ok(())
    }

    /// Materialize the delegate reader for a field (build persistent index).
    pub fn materialize_delegate_reader(&self, field_name: &str) -> Result<()> {
        let fields = self.fields.read();
        let handle = fields.get(field_name).ok_or_else(|| {
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
        let config = in_memory.config().clone();
        drop(fields);

        self.write_field_delegate_index(field_name, &config, vectors)?;
        let reader = self.load_delegate_reader(field_name, &config)?;

        let fields = self.fields.read();
        let handle = fields.get(field_name).ok_or_else(|| {
            PlatypusError::not_found(format!("vector field '{field_name}' is not registered"))
        })?;
        handle.runtime.replace_reader(reader);
        Ok(())
    }

    /// Create a searcher for this collection.
    ///
    /// Returns a boxed [`VectorSearcher`](crate::vector::search::searcher::VectorSearcher)
    /// capable of executing search and count operations.
    pub fn searcher(&self) -> Result<Box<dyn crate::vector::search::searcher::VectorSearcher>> {
        Ok(Box::new(VectorCollectionSearcher::from_collection_ref(
            self,
        )))
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
}

/// Searcher implementation for [`VectorCollection`].
///
/// This struct implements the [`VectorSearcher`](crate::vector::search::searcher::VectorSearcher)
/// trait, providing high-level search functionality for vector collections.
/// It handles multi-field queries, score aggregation, filtering, and result ranking.
///
/// # Architecture
///
/// The searcher holds shared references (via `Arc`) to the collection's internal components:
/// - `config`: Collection configuration for field resolution
/// - `fields`: Registered vector fields for searching
/// - `registry`: Document registry for filtering
/// - `documents`: Document storage for counting
///
/// This design allows creating a searcher from `&VectorCollection` without
/// requiring `Arc<VectorCollection>`, which is important for the `searcher()`
/// method that only receives `&self`.
///
/// # Example
///
/// ```ignore
/// let collection: VectorCollection = /* ... */;
/// let searcher = VectorCollectionSearcher::from_collection_ref(&collection);
/// let results = searcher.search(&request)?;
/// ```
#[derive(Debug)]
pub struct VectorCollectionSearcher {
    config: Arc<VectorIndexConfig>,
    fields: Arc<RwLock<HashMap<String, FieldHandle>>>,
    registry: Arc<DocumentVectorRegistry>,
    documents: Arc<RwLock<HashMap<u64, DocumentVector>>>,
}

impl VectorCollectionSearcher {
    /// Create a new searcher from a collection reference.
    ///
    /// This method clones the `Arc` references to the collection's internal components,
    /// allowing the searcher to operate independently while sharing the same underlying data.
    pub fn from_collection_ref(collection: &VectorCollection) -> Self {
        Self {
            config: Arc::clone(&collection.config),
            fields: Arc::clone(&collection.fields),
            registry: Arc::clone(&collection.registry),
            documents: Arc::clone(&collection.documents),
        }
    }

    /// Resolve which fields to search based on the request.
    fn resolve_fields(&self, request: &VectorSearchRequest) -> Result<Vec<String>> {
        match &request.fields {
            Some(selectors) => self.apply_field_selectors(selectors),
            None => {
                // Use default fields from config.
                Ok(self.config.default_fields.clone())
            }
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
                        return Err(PlatypusError::not_found(format!(
                            "vector field '{}' is not registered",
                            name
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
                FieldSelector::VectorType(vector_type) => {
                    for (field_name, handle) in fields.iter() {
                        if &handle.field.config().vector_type == vector_type
                            && !result.contains(field_name)
                        {
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

    /// Get query vectors that match a specific field configuration.
    fn query_vectors_for_field(
        &self,
        config: &VectorFieldConfig,
        request: &VectorSearchRequest,
    ) -> Vec<QueryVector> {
        request
            .query_vectors
            .iter()
            .filter(|candidate| {
                candidate.vector.source_tag == config.source_tag
                    && candidate.vector.vector_type == config.vector_type
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
                    return Err(PlatypusError::invalid_argument(
                        "VectorScoreMode::LateInteraction is not supported yet",
                    ));
                }
            }
            entry.field_hits.push(hit);
        }

        Ok(())
    }
}

impl crate::vector::search::searcher::VectorSearcher for VectorCollectionSearcher {
    fn search(&self, request: &VectorSearchRequest) -> Result<VectorSearchResults> {
        if request.query_vectors.is_empty() {
            return Err(PlatypusError::invalid_argument(
                "VectorSearchRequest requires at least one query vector",
            ));
        }

        if request.limit == 0 {
            return Ok(VectorSearchResults::default());
        }

        if request.overfetch < 1.0 {
            return Err(PlatypusError::invalid_argument(
                "VectorSearchRequest overfetch must be >= 1.0",
            ));
        }

        if matches!(request.score_mode, VectorScoreMode::LateInteraction) {
            return Err(PlatypusError::invalid_argument(
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
                .ok_or_else(|| PlatypusError::not_found(format!("vector field '{field_name}'")))?;
            let matching_vectors = self.query_vectors_for_field(field.field.config(), request);
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
            return Err(PlatypusError::invalid_argument(
                "no query vectors matched the requested fields",
            ));
        }

        let mut hits: Vec<VectorHit> = doc_hits.into_values().collect();

        // Apply min_score filtering if specified.
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
        // If no query vectors, return total document count.
        if request.query_vectors.is_empty() {
            let documents = self.documents.read();
            return Ok(documents.len() as u64);
        }

        // For queries with vectors, perform a search with no limit and count results.
        // Create a modified request with unlimited results.
        let count_request = VectorSearchRequest {
            query_vectors: request.query_vectors.clone(),
            query_payloads: request.query_payloads.clone(),
            fields: request.fields.clone(),
            limit: usize::MAX,
            score_mode: request.score_mode,
            overfetch: 1.0, // No overfetch needed for counting.
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
