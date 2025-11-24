//! Step-by-step doc-centric vector search example using automatic embeddings.
//!
//! Run with `cargo run --example vector_search`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use platypus::embedding::text_embedder::TextEmbedder;
use platypus::error::Result;
use platypus::storage::Storage;
use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use platypus::vector::DistanceMetric;
use platypus::vector::core::document::{DocumentPayload, FieldPayload, RawTextSegment, VectorRole};
use platypus::vector::core::vector::Vector;
use platypus::vector::engine::{
    FieldSelector, VectorEmbedderConfig, VectorEmbedderProvider, VectorEngine, VectorEngineConfig,
    VectorEngineFilter, VectorEngineSearchRequest, VectorFieldConfig, VectorIndexKind,
    VectorScoreMode,
};

const DIM: usize = 4;
const EMBEDDER_ID: &str = "demo-encoder";
const EMBEDDER_CONFIG_ID: &str = "demo_text_embedder";
const TITLE_FIELD: &str = "title_embedding";
const BODY_FIELD: &str = "body_embedding";

fn main() -> Result<()> {
    println!("1) Configure storage + VectorEngine fields with an embedder registry\n");
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default())) as Arc<dyn Storage>;

    let mut field_configs = HashMap::new();
    field_configs.insert(
        TITLE_FIELD.to_string(),
        VectorFieldConfig {
            dimension: DIM,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: EMBEDDER_ID.to_string(),
            role: VectorRole::Text,
            embedder: Some(EMBEDDER_CONFIG_ID.into()),
            base_weight: 1.2,
        },
    );
    field_configs.insert(
        BODY_FIELD.to_string(),
        VectorFieldConfig {
            dimension: DIM,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: EMBEDDER_ID.to_string(),
            role: VectorRole::Text,
            embedder: Some(EMBEDDER_CONFIG_ID.into()),
            base_weight: 1.0,
        },
    );

    let embedders = HashMap::from([(
        EMBEDDER_CONFIG_ID.into(),
        VectorEmbedderConfig {
            provider: VectorEmbedderProvider::External,
            model: "demo".into(),
            options: HashMap::new(),
        },
    )]);

    let config = VectorEngineConfig {
        fields: field_configs,
        embedders,
        default_fields: vec![TITLE_FIELD.into(), BODY_FIELD.into()],
        metadata: HashMap::new(),
    };

    let engine = VectorEngine::new(config, storage, None)?;
    engine.register_embedder_instance(EMBEDDER_CONFIG_ID, Arc::new(DemoTextEmbedder::new(DIM)))?;

    println!("2) Upsert documents with raw payloads that will be embedded automatically\n");
    let mut doc1 = DocumentPayload::new(1);
    doc1.add_metadata("lang".to_string(), "en".to_string());
    doc1.add_metadata("category".to_string(), "programming".to_string());

    let mut doc1_title = FieldPayload::default();
    doc1_title.add_metadata("section".into(), "title".into());
    doc1_title.add_text_segment(RawTextSegment::new("Rust overview"));
    doc1.add_field(TITLE_FIELD, doc1_title);

    let mut doc1_body = FieldPayload::default();
    doc1_body.add_metadata("section".into(), "body".into());
    doc1_body.add_text_segment(RawTextSegment::new(
        "Rust balances performance with memory safety",
    ));
    doc1_body.add_metadata("chunk".into(), "rust-body".into());
    doc1.add_field(BODY_FIELD, doc1_body);

    let mut doc2 = DocumentPayload::new(2);
    doc2.add_metadata("lang".to_string(), "ja".to_string());
    doc2.add_metadata("category".to_string(), "ai".to_string());

    let mut doc2_title = FieldPayload::default();
    doc2_title.add_metadata("section".into(), "title".into());
    doc2_title.add_text_segment(RawTextSegment::new("LLM primer"));
    doc2.add_field(TITLE_FIELD, doc2_title);

    let mut doc2_body = FieldPayload::default();
    doc2_body.add_metadata("section".into(), "body".into());
    doc2_body.add_text_segment(RawTextSegment::new("LLM internals"));
    doc2_body.add_metadata("chunk".into(), "llm-body".into());
    doc2.add_field(BODY_FIELD, doc2_body);

    // doc3 demonstrates splitting a long body into multiple text segments (e.g., per page).
    let mut doc3 = DocumentPayload::new(3);
    doc3.add_metadata("lang".to_string(), "en".to_string());
    doc3.add_metadata("category".to_string(), "manual".to_string());

    let mut doc3_body = FieldPayload::default();
    doc3_body.add_metadata("section".into(), "body".into());
    doc3_body.add_metadata("source".into(), "user-guide".into());

    let mut page1 = RawTextSegment::new("Page 1: Installation steps for the runtime environment");
    page1.attributes.insert("page".into(), "1".into());
    doc3_body.add_text_segment(page1);

    let mut page2 = RawTextSegment::new("Page 2: Configuration references and tuning knobs");
    page2.attributes.insert("page".into(), "2".into());

    doc3_body.add_text_segment(page2);
    let mut page3 = RawTextSegment::new("Page 3: Troubleshooting common deployment issues");
    page3.attributes.insert("page".into(), "3".into());

    doc3_body.add_text_segment(page3);
    doc3.add_field(BODY_FIELD, doc3_body);

    engine.upsert_document_payload(doc1)?;
    engine.upsert_document_payload(doc2)?;
    engine.upsert_document_payload(doc3)?;
    println!("   -> Inserted {} docs\n", engine.stats()?.document_count);

    println!("3) Build a VectorEngineSearchRequest directly from query text\n");
    let mut doc_filter = VectorEngineFilter::default();
    doc_filter
        .document
        .equals
        .insert("lang".into(), "en".into());

    let mut query = VectorEngineSearchRequest::default();
    query.limit = 5;
    query.fields = Some(vec![
        FieldSelector::Exact(BODY_FIELD.into()),
        FieldSelector::Exact(TITLE_FIELD.into()),
    ]);
    query.filter = Some(doc_filter);
    query.score_mode = VectorScoreMode::WeightedSum;

    let mut title_query = FieldPayload::default();
    title_query.add_metadata("section".into(), "title".into());
    title_query.add_text_segment(RawTextSegment::new("systems programming"));
    query
        .query_vectors
        .extend(engine.embed_query_field_payload(TITLE_FIELD, title_query)?);

    let mut body_query = FieldPayload::default();
    body_query.add_metadata("section".into(), "body".into());
    body_query.add_text_segment(RawTextSegment::new("memory safety"));
    query
        .query_vectors
        .extend(engine.embed_query_field_payload(BODY_FIELD, body_query)?);

    println!("4) Execute the search and inspect doc-centric hits\n");
    let results = engine.search(&query)?;
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
#[derive(Debug)]
struct DemoTextEmbedder {
    dimension: usize,
}

impl DemoTextEmbedder {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl TextEmbedder for DemoTextEmbedder {
    async fn embed(&self, text: &str) -> Result<Vector> {
        let mut data = vec![0.0_f32; self.dimension];
        for (idx, byte) in text.bytes().enumerate() {
            let bucket = idx % self.dimension;
            data[bucket] += (byte as f32) / 255.0;
        }
        Ok(Vector::new(data))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "demo-text-embedder"
    }
}
