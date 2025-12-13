//! Step-by-step doc-centric vector search example using automatic embeddings.
//!
//! Run with `cargo run --example vector_search --features embeddings-candle`.

#[cfg(not(feature = "embeddings-candle"))]
fn main() {
    eprintln!("This example requires the 'embeddings-candle' feature.");
    eprintln!("Please rerun with: cargo run --example vector_search --features embeddings-candle");
    std::process::exit(1);
}

#[cfg(feature = "embeddings-candle")]
mod candle_vector_example {
    use std::{collections::HashMap, sync::Arc};

    use platypus::{
        embedding::{
            candle_text_embedder::CandleTextEmbedder, embedder::Embedder,
            per_field::PerFieldEmbedder,
        },
        error::Result,
        storage::{
            Storage,
            memory::{MemoryStorage, MemoryStorageConfig},
        },
        vector::{
            DistanceMetric,
            core::document::{DocumentPayload, FieldPayload, SegmentPayload, VectorType},
            engine::{
                FieldSelector, VectorEngine, VectorEngineFilter, VectorEngineSearchRequest,
                VectorFieldConfig, VectorIndexConfig, VectorIndexKind, VectorScoreMode,
            },
        },
    };

    const EMBEDDER_CONFIG_ID: &str = "candle_text_embedder";
    const TITLE_FIELD: &str = "title_embedding";
    const BODY_FIELD: &str = "body_embedding";
    const TITLE_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
    const BODY_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

    pub fn run() -> Result<()> {
        println!("1) Configure storage + VectorEngine fields with an embedder registry\n");
        println!("   Loading Candle embedders (downloads may occur on first run)...\n");

        let storage =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default())) as Arc<dyn Storage>;

        let title_embedder: Arc<dyn Embedder> = Arc::new(CandleTextEmbedder::new(TITLE_MODEL)?);
        let body_embedder: Arc<dyn Embedder> = Arc::new(CandleTextEmbedder::new(BODY_MODEL)?);
        let title_dim = title_embedder.dimension();
        let body_dim = body_embedder.dimension();

        // Configure PerFieldEmbedder so each vector field can transparently use Candle embedders.
        let mut per_field_embedder = PerFieldEmbedder::new(Arc::clone(&body_embedder));
        per_field_embedder.add_embedder(TITLE_FIELD, Arc::clone(&title_embedder));

        // Build config using the new embedder field API
        let config = VectorIndexConfig::builder()
            .field(
                TITLE_FIELD,
                VectorFieldConfig {
                    dimension: title_dim,
                    distance: DistanceMetric::Cosine,
                    index: VectorIndexKind::Flat,
                    embedder_id: EMBEDDER_CONFIG_ID.to_string(),
                    vector_type: VectorType::Text,
                    embedder: Some(EMBEDDER_CONFIG_ID.into()),
                    base_weight: 1.2,
                },
            )
            .field(
                BODY_FIELD,
                VectorFieldConfig {
                    dimension: body_dim,
                    distance: DistanceMetric::Cosine,
                    index: VectorIndexKind::Flat,
                    embedder_id: EMBEDDER_CONFIG_ID.to_string(),
                    vector_type: VectorType::Text,
                    embedder: Some(EMBEDDER_CONFIG_ID.into()),
                    base_weight: 1.0,
                },
            )
            .default_field(TITLE_FIELD)
            .default_field(BODY_FIELD)
            .embedder(per_field_embedder)
            .build()?;

        let engine = VectorEngine::new(storage, config)?;

        println!("2) Upsert documents with raw payloads that will be embedded automatically\n");
        let mut doc1 = DocumentPayload::new();
        doc1.add_metadata("lang".to_string(), "en".to_string());
        doc1.add_metadata("category".to_string(), "programming".to_string());

        let mut doc1_title = FieldPayload::default();
        doc1_title.add_metadata("section".into(), "title".into());
        doc1_title.add_text_segment("Rust overview");
        doc1.add_field(TITLE_FIELD, doc1_title);

        let mut doc1_body = FieldPayload::default();
        doc1_body.add_metadata("section".into(), "body".into());
        doc1_body.add_text_segment("Rust balances performance with memory safety");
        doc1_body.add_metadata("chunk".into(), "rust-body".into());
        doc1.add_field(BODY_FIELD, doc1_body);

        let mut doc2 = DocumentPayload::new();
        doc2.add_metadata("lang".to_string(), "ja".to_string());
        doc2.add_metadata("category".to_string(), "ai".to_string());

        let mut doc2_title = FieldPayload::default();
        doc2_title.add_metadata("section".into(), "title".into());
        doc2_title.add_text_segment("LLM primer");
        doc2.add_field(TITLE_FIELD, doc2_title);

        let mut doc2_body = FieldPayload::default();
        doc2_body.add_metadata("section".into(), "body".into());
        doc2_body.add_text_segment("LLM internals");
        doc2_body.add_metadata("chunk".into(), "llm-body".into());
        doc2.add_field(BODY_FIELD, doc2_body);

        // doc3 demonstrates splitting a long body into multiple text segments (e.g., per page).
        let mut doc3 = DocumentPayload::new();
        doc3.add_metadata("lang".to_string(), "en".to_string());
        doc3.add_metadata("category".to_string(), "manual".to_string());

        let mut doc3_body = FieldPayload::default();
        doc3_body.add_metadata("section".into(), "body".into());
        doc3_body.add_metadata("source".into(), "user-guide".into());

        doc3_body.add_segment(
            SegmentPayload::text("Page 1: Installation steps for the runtime environment")
                .with_metadata(HashMap::from([(String::from("page"), String::from("1"))])),
        );

        doc3_body.add_segment(
            SegmentPayload::text("Page 2: Configuration references and tuning knobs")
                .with_metadata(HashMap::from([(String::from("page"), String::from("2"))])),
        );

        doc3_body.add_segment(
            SegmentPayload::text("Page 3: Troubleshooting common deployment issues")
                .with_metadata(HashMap::from([(String::from("page"), String::from("3"))])),
        );
        doc3.add_field(BODY_FIELD, doc3_body);

        engine.upsert_document_payload(1, doc1)?;
        engine.upsert_document_payload(2, doc2)?;
        engine.upsert_document_payload(3, doc3)?;
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
        title_query.add_text_segment("systems programming");
        query
            .query_vectors
            .extend(engine.embed_query_field_payload(TITLE_FIELD, title_query)?);

        let mut body_query = FieldPayload::default();
        body_query.add_metadata("section".into(), "body".into());
        body_query.add_text_segment("memory safety");
        query
            .query_vectors
            .extend(engine.embed_query_field_payload(BODY_FIELD, body_query)?);

        println!("4) Execute the search and inspect doc-centric hits\n");
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
}

#[cfg(feature = "embeddings-candle")]
fn main() -> platypus::error::Result<()> {
    candle_vector_example::run()
}
