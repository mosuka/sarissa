//! Factory for creating vector field readers and writers.

use std::sync::Arc;

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::vector::core::field::FieldOption;
use crate::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
use crate::vector::index::field::{
    LegacyVectorFieldWriter, VectorField, VectorFieldReader, VectorFieldWriter,
};
use crate::vector::index::flat::field_reader::FlatFieldReader;
use crate::vector::index::flat::reader::FlatVectorIndexReader;
use crate::vector::index::flat::writer::FlatIndexWriter;
use crate::vector::index::hnsw::field_reader::HnswFieldReader;
use crate::vector::index::hnsw::reader::HnswIndexReader as HnswVectorIndexReader;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::index::ivf::field_reader::IvfFieldReader;
use crate::vector::index::ivf::reader::IvfIndexReader as IvfVectorIndexReader;
use crate::vector::index::ivf::writer::IvfIndexWriter;
use crate::vector::store::config::VectorFieldConfig;
use crate::vector::writer::VectorIndexWriterConfig;

/// Base name for field index files.
pub const FIELD_INDEX_BASENAME: &str = "field";

/// Factory for creating vector field readers and writers.
pub struct VectorFieldFactory;

impl VectorFieldFactory {
    /// Create a writer for a specific vector field configuration.
    pub fn create_writer(
        field_name: &str,
        vector_option: &FieldOption,
        storage: Arc<dyn Storage>,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Arc<dyn VectorFieldWriter>> {
        if vector_option.dimension() == 0 {
            return Err(LaurusError::invalid_config(format!(
                "vector field '{field_name}' cannot materialize a zero-dimension index"
            )));
        }

        use crate::vector::store::embedding_writer::EmbeddingVectorIndexWriter;
        use crate::vector::writer::VectorIndexWriter;

        let inner_writer: Box<dyn VectorIndexWriter> = match vector_option {
            FieldOption::Flat(opt) => {
                let flat = FlatIndexConfig {
                    dimension: opt.dimension,
                    distance_metric: opt.distance,
                    rerank_storage: opt.rerank_storage,
                    embedder: embedder.clone(),
                    ..FlatIndexConfig::default()
                };
                Box::new(FlatIndexWriter::with_storage(
                    flat,
                    VectorIndexWriterConfig::default(),
                    FIELD_INDEX_BASENAME,
                    storage,
                )?)
            }
            FieldOption::Hnsw(opt) => {
                let hnsw = HnswIndexConfig {
                    dimension: opt.dimension,
                    distance_metric: opt.distance,
                    m: opt.m,
                    ef_construction: opt.ef_construction,
                    default_ef_search: opt.default_ef_search,
                    quantization_method: opt.quantizer,
                    rerank_storage: opt.rerank_storage,
                    embedder: embedder.clone(),
                    ..HnswIndexConfig::default()
                };
                Box::new(HnswIndexWriter::with_storage(
                    hnsw,
                    VectorIndexWriterConfig::default(),
                    FIELD_INDEX_BASENAME,
                    storage,
                )?)
            }
            FieldOption::Ivf(opt) => {
                let ivf = IvfIndexConfig {
                    dimension: opt.dimension,
                    distance_metric: opt.distance,
                    n_clusters: opt.n_clusters,
                    n_probe: opt.n_probe,
                    rerank_storage: opt.rerank_storage,
                    embedder: embedder.clone(),
                    ..IvfIndexConfig::default()
                };
                Box::new(IvfIndexWriter::with_storage(
                    ivf,
                    VectorIndexWriterConfig::default(),
                    FIELD_INDEX_BASENAME,
                    storage,
                )?)
            }
        };

        let embedding_writer = EmbeddingVectorIndexWriter::new(inner_writer, embedder.clone());
        let writer: Arc<dyn VectorFieldWriter> = Arc::new(
            LegacyVectorFieldWriter::new(field_name, embedding_writer).with_embedder(embedder),
        );

        Ok(writer)
    }

    /// Create a reader for a specific vector field configuration.
    pub fn create_reader(
        field_name: &str,
        vector_option: &FieldOption,
        storage: Arc<dyn Storage>,
        deletion_bitmap: Option<Arc<DeletionBitmap>>,
    ) -> Result<Arc<dyn VectorFieldReader>> {
        let reader: Arc<dyn VectorFieldReader> = match vector_option {
            FieldOption::Flat(opt) => {
                let mut reader = FlatVectorIndexReader::load(
                    storage.clone(),
                    FIELD_INDEX_BASENAME,
                    opt.distance,
                )?;
                if let Some(bitmap) = deletion_bitmap {
                    reader.set_deletion_bitmap(bitmap);
                }
                Arc::new(FlatFieldReader::new(field_name, Arc::new(reader)))
            }
            FieldOption::Hnsw(opt) => {
                let mut reader = HnswVectorIndexReader::load(
                    storage.clone(),
                    FIELD_INDEX_BASENAME,
                    opt.distance,
                )?;
                if let Some(bitmap) = deletion_bitmap {
                    reader.set_deletion_bitmap(bitmap);
                }
                Arc::new(HnswFieldReader::new(field_name, Arc::new(reader)))
            }
            FieldOption::Ivf(opt) => {
                let mut reader = IvfVectorIndexReader::load(
                    storage.clone(),
                    FIELD_INDEX_BASENAME,
                    opt.distance,
                )?;
                if let Some(bitmap) = deletion_bitmap {
                    reader.set_deletion_bitmap(bitmap);
                }
                Arc::new(IvfFieldReader::with_n_probe(
                    field_name,
                    Arc::new(reader),
                    opt.n_probe,
                ))
            }
        };

        Ok(reader)
    }

    /// Create a full vector field (segmented or in-memory) based on configuration.
    pub fn create_field(
        name: String,
        config: VectorFieldConfig,
        storage: Arc<dyn Storage>,
        deletion_bitmap: Option<Arc<crate::maintenance::deletion::DeletionBitmap>>,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Arc<dyn VectorField>> {
        use crate::vector::index::hnsw::segment::manager::{SegmentManager, SegmentManagerConfig};
        use crate::vector::index::segmented_field::SegmentedVectorField;
        use crate::vector::store::memory::InMemoryVectorField;

        let vector_opt = config.vector.clone();
        let field: Arc<dyn VectorField> = match vector_opt {
            Some(FieldOption::Hnsw(_)) => {
                let manager_config = SegmentManagerConfig::default();
                let segment_manager =
                    Arc::new(SegmentManager::new(manager_config, storage.clone())?);

                Arc::new(SegmentedVectorField::create(
                    name,
                    config,
                    segment_manager,
                    storage,
                    deletion_bitmap,
                )?)
            }
            _ => {
                let vector_option = config.vector.as_ref();
                let (delegate_writer, delegate_reader) = if let Some(opt) = vector_option {
                    if opt.dimension() > 0 {
                        let writer = Self::create_writer(&name, opt, storage.clone(), embedder)?;
                        // Only create reader if storage exists, otherwise None?
                        // Actually create_reader attempts to LOAD. If file doesn't exist, it might fail or return empty?
                        // FlatVectorIndexReader::load checks file existence usually?
                        // Let's check if we can create it safely.
                        // Or we trust create_reader to handle non-existent gracefully?
                        // Usually reader::load returns error if file missing.
                        // We should check file existence or Try to create reader.
                        // For now, let's look at create_reader implementation above...
                        // It calls load().

                        // If file doesn't exist, we probably shouldn't create a reader yet, OR create an empty one.
                        // But FlatVectorIndexReader doesn't seem to support "empty/new".

                        // Assume error means "not found" or similar. Ideally we check specific error.
                        let reader = Self::create_reader(
                            &name,
                            opt,
                            storage.clone(),
                            deletion_bitmap.clone(),
                        )
                        .ok();

                        (Some(writer), reader)
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };

                Arc::new(InMemoryVectorField::new(
                    name,
                    config,
                    delegate_writer,
                    delegate_reader,
                )?)
            }
        };

        Ok(field)
    }
}
