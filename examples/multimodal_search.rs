//! End-to-end multimodal vector search example.
//!
//! Demonstrates how to mix text and image payloads, register a custom
//! multimodal embedder, and run a blended query against both fields.
//!
//! Run with `cargo run --example multimodal_search`.

use std::any::Any;
use std::collections::HashMap;
use std::fs;
#[cfg(feature = "embeddings-multimodal")]
use std::io::Cursor;
use std::io::Write;
use std::sync::Arc;

use async_trait::async_trait;
#[cfg(feature = "embeddings-multimodal")]
use platypus::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
use platypus::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
use platypus::embedding::per_field::PerFieldEmbedder;
use platypus::error::{PlatypusError, Result};
use platypus::storage::Storage;
use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use platypus::vector::DistanceMetric;
use platypus::vector::core::document::{
    DocumentPayload, FieldPayload, PayloadSource, SegmentPayload, VectorType,
};
use platypus::vector::core::vector::Vector;
use platypus::vector::engine::{
    FieldSelector, QueryPayload, VectorEngine, VectorFieldConfig, VectorIndexConfig,
    VectorIndexKind, VectorScoreMode, VectorSearchRequest,
};
use tempfile::{Builder, NamedTempFile};

const DEMO_TEXT_DIM: usize = 4;
const DEMO_IMAGE_DIM: usize = 3;
const MULTIMODAL_EMBEDDER_ID: &str = "demo-multimodal";
const TEXT_FIELD: &str = "body_embedding";
const IMAGE_FIELD: &str = "image_embedding";
#[cfg(feature = "embeddings-multimodal")]
const DEFAULT_CANDLE_MODEL: &str = "openai/clip-vit-base-patch32";

fn main() -> Result<()> {
    let embedder_choice = select_embedder()?;
    let text_dim = embedder_choice.text_embedder.dimension();
    let image_dim = embedder_choice.image_embedder.dimension();

    println!("1) Configure an in-memory VectorEngine with text and image fields\n");
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default())) as Arc<dyn Storage>;

    // Create PerFieldEmbedder with text embedder as default and image embedder for IMAGE_FIELD
    let mut per_field_embedder = PerFieldEmbedder::new(Arc::clone(&embedder_choice.text_embedder));
    per_field_embedder.add_embedder(TEXT_FIELD, Arc::clone(&embedder_choice.text_embedder));
    per_field_embedder.add_embedder(IMAGE_FIELD, Arc::clone(&embedder_choice.image_embedder));

    // Build config using the new embedder field
    // Note: `embedder` field is the lookup key that PerFieldEmbedder uses
    let config = VectorIndexConfig::builder()
        .field(
            TEXT_FIELD,
            VectorFieldConfig {
                dimension: text_dim,
                distance: DistanceMetric::Cosine,
                index: VectorIndexKind::Flat,
                embedder_id: MULTIMODAL_EMBEDDER_ID.into(),
                vector_type: VectorType::Text,
                embedder: Some(TEXT_FIELD.into()),
                base_weight: 1.0,
            },
        )
        .field(
            IMAGE_FIELD,
            VectorFieldConfig {
                dimension: image_dim,
                distance: DistanceMetric::Cosine,
                index: VectorIndexKind::Flat,
                embedder_id: MULTIMODAL_EMBEDDER_ID.into(),
                vector_type: VectorType::Image,
                embedder: Some(IMAGE_FIELD.into()),
                base_weight: 1.0,
            },
        )
        .default_field(TEXT_FIELD)
        .default_field(IMAGE_FIELD)
        .embedder(per_field_embedder)
        .build()?;

    let engine = VectorEngine::new(storage, config)?;

    println!("2) Upsert documents containing text and image segments\n");
    let mut doc1 = DocumentPayload::new();
    doc1.metadata.insert("category".into(), "notebook".into());
    doc1.add_field(TEXT_FIELD, text_payload("Rust notebook overview"));
    let doc1_image = embedder_choice.sample_image_bytes(3)?;
    doc1.add_field(IMAGE_FIELD, image_bytes_payload(doc1_image, "rye-paper"));

    let mut doc2 = DocumentPayload::new();
    doc2.metadata.insert("category".into(), "camera".into());
    doc2.add_field(
        TEXT_FIELD,
        text_payload("Mirrorless camera quickstart guide"),
    );
    let doc2_image = embedder_choice.sample_image_bytes(9)?;
    let (image_payload_doc2, temp_file_doc2) = image_uri_payload(doc2_image, "studio");
    doc2.add_field(IMAGE_FIELD, image_payload_doc2);

    engine.upsert_document_payload(1, doc1)?;
    engine.upsert_document_payload(2, doc2)?;
    // Ensure the temporary file lives until ingestion finishes.
    drop(temp_file_doc2);

    println!("   -> Docs indexed: {}\n", engine.stats()?.document_count);

    println!("3) Build one text query + one image query\n");
    let mut query = VectorSearchRequest::default();
    query.limit = 2;
    query.fields = Some(vec![
        FieldSelector::Exact(TEXT_FIELD.into()),
        FieldSelector::Exact(IMAGE_FIELD.into()),
    ]);
    query.score_mode = VectorScoreMode::WeightedSum;

    let mut text_query = FieldPayload::default();
    text_query.add_text_segment("portable developer setup");
    query
        .query_payloads
        .push(QueryPayload::new(TEXT_FIELD, text_query));

    let query_image = embedder_choice.sample_image_bytes(12)?;
    let (image_query_payload, _temp_query_file) = image_uri_payload(query_image, "query");
    query
        .query_payloads
        .push(QueryPayload::new(IMAGE_FIELD, image_query_payload));
    // Note: _temp_query_file must remain alive until search() completes (it reads the file).

    println!("4) Execute the blended search\n");
    let results = engine.search(query)?;
    for (rank, hit) in results.hits.iter().enumerate() {
        println!("{}. doc {} • score {:.3}", rank + 1, hit.doc_id, hit.score);
        for field_hit in &hit.field_hits {
            println!(
                "   field {:<15} distance {:.3} score {:.3}",
                field_hit.field, field_hit.distance, field_hit.score
            );
        }
    }

    Ok(())
}

struct EmbedderChoice {
    model_label: String,
    text_embedder: Arc<dyn Embedder>,
    image_embedder: Arc<dyn Embedder>,
    real_images: bool,
}

fn select_embedder() -> Result<EmbedderChoice> {
    let use_candle = std::env::args().any(|arg| arg == "--use-candle");
    if use_candle {
        #[cfg(feature = "embeddings-multimodal")]
        {
            let model = std::env::var("PLATYPUS_CANDLE_MODEL")
                .unwrap_or_else(|_| DEFAULT_CANDLE_MODEL.to_string());
            println!(
                "   -> Using Candle CLIP embedder '{}' (enable caching via HF_HOME)",
                model
            );
            let embedder: Arc<dyn Embedder> =
                Arc::new(CandleMultimodalEmbedder::new(model.as_str())?);
            return Ok(EmbedderChoice {
                model_label: model,
                text_embedder: embedder.clone(),
                image_embedder: embedder,
                real_images: true,
            });
        }
        #[cfg(not(feature = "embeddings-multimodal"))]
        {
            return Err(PlatypusError::invalid_argument(
                "--use-candle requires the 'embeddings-multimodal' feature",
            ));
        }
    }

    println!("   -> Using built-in demo embedder (no external models)");
    let text_embedder: Arc<dyn Embedder> = Arc::new(DemoTextEmbedder::new(DEMO_TEXT_DIM));
    let image_embedder: Arc<dyn Embedder> = Arc::new(DemoImageEmbedder::new(DEMO_IMAGE_DIM));
    Ok(EmbedderChoice {
        model_label: "demo-multimodal".into(),
        text_embedder,
        image_embedder,
        real_images: false,
    })
}

impl EmbedderChoice {
    fn sample_image_bytes(&self, seed: u8) -> Result<Vec<u8>> {
        if self.real_images {
            #[cfg(feature = "embeddings-multimodal")]
            {
                use image::codecs::png::PngEncoder;
                use image::{ExtendedColorType, ImageBuffer, ImageEncoder, Rgb};
                let img = ImageBuffer::from_fn(16, 16, |x, y| {
                    let base = seed as u32 + x + (y * 3);
                    Rgb([
                        (base & 0xFF) as u8,
                        (base.wrapping_mul(5) & 0xFF) as u8,
                        (base.wrapping_mul(11) & 0xFF) as u8,
                    ])
                });
                let mut cursor = Cursor::new(Vec::new());
                PngEncoder::new(&mut cursor)
                    .write_image(img.as_raw(), 16, 16, ExtendedColorType::Rgb8)
                    .map_err(|err| {
                        PlatypusError::internal(format!(
                            "failed to encode sample image payload: {err}"
                        ))
                    })?;
                return Ok(cursor.into_inner());
            }
            #[cfg(not(feature = "embeddings-multimodal"))]
            {
                unreachable!("real image generation only active with multimodal feature");
            }
        }

        Ok((0..16)
            .map(|idx| seed.wrapping_mul(7).wrapping_add(idx))
            .collect())
    }
}

fn text_payload(value: &str) -> FieldPayload {
    let mut payload = FieldPayload::default();
    payload.add_text_segment(value);
    payload
}

fn image_bytes_payload(bytes: Vec<u8>, label: &str) -> FieldPayload {
    let mut payload = FieldPayload::default();
    payload.add_segment(
        SegmentPayload::new(
            PayloadSource::Bytes {
                bytes: Arc::<[u8]>::from(bytes),
                mime: Some("image/png".into()),
            },
            VectorType::Image,
        )
        .with_metadata(HashMap::from([(String::from("variant"), label.into())])),
    );
    payload
}

fn image_uri_payload(bytes: Vec<u8>, label: &str) -> (FieldPayload, NamedTempFile) {
    let mut temp_file = Builder::new().suffix(".png").tempfile().expect("temp file");
    temp_file
        .write_all(&bytes)
        .expect("write sample image payload");
    let uri = temp_file.path().to_string_lossy().to_string();

    let mut payload = FieldPayload::default();
    payload.add_segment(
        SegmentPayload::new(
            PayloadSource::Uri {
                uri,
                media_hint: Some("image/png".into()),
            },
            VectorType::Image,
        )
        .with_metadata(HashMap::from([(String::from("variant"), label.into())])),
    );
    (payload, temp_file)
}

/// Demo text embedder for examples.
#[derive(Debug)]
struct DemoTextEmbedder {
    dim: usize,
}

impl DemoTextEmbedder {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn vector_from_bytes(&self, bytes: &[u8]) -> Vector {
        if bytes.is_empty() {
            return Vector::new(vec![0.0; self.dim]);
        }
        let mut data = vec![0.0_f32; self.dim];
        for (idx, byte) in bytes.iter().enumerate() {
            let bucket = idx % self.dim;
            data[bucket] += (*byte as f32) / 255.0;
        }
        Vector::new(data)
    }
}

#[async_trait]
impl Embedder for DemoTextEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(text) => Ok(self.vector_from_bytes(text.as_bytes())),
            _ => Err(PlatypusError::invalid_argument(
                "DemoTextEmbedder only supports text input",
            )),
        }
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    fn name(&self) -> &str {
        "demo-text"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Demo image embedder for examples.
#[derive(Debug)]
struct DemoImageEmbedder {
    dim: usize,
}

impl DemoImageEmbedder {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn vector_from_bytes(&self, bytes: &[u8]) -> Vector {
        if bytes.is_empty() {
            return Vector::new(vec![0.0; self.dim]);
        }
        let mut data = vec![0.0_f32; self.dim];
        for (idx, byte) in bytes.iter().enumerate() {
            let bucket = idx % self.dim;
            data[bucket] += (*byte as f32) / 255.0;
        }
        Vector::new(data)
    }
}

#[async_trait]
impl Embedder for DemoImageEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::ImagePath(path) => {
                let bytes = fs::read(path)?;
                Ok(self.vector_from_bytes(&bytes))
            }
            EmbedInput::ImageBytes(bytes, _) => Ok(self.vector_from_bytes(bytes)),
            EmbedInput::ImageUri(uri) => {
                // For demo purposes, treat URI as file path
                let bytes = fs::read(uri)?;
                Ok(self.vector_from_bytes(&bytes))
            }
            EmbedInput::Text(_) => Err(PlatypusError::invalid_argument(
                "DemoImageEmbedder only supports image input",
            )),
        }
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Image]
    }

    fn name(&self) -> &str {
        "demo-image"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
