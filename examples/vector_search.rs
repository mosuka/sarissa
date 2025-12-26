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
    use std::sync::Arc;

    use sarissa::{
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
            core::document::{DocumentPayload, Payload, VectorType},
            engine::{
                FieldSelector, QueryPayload, VectorEngine, VectorFieldConfig, VectorFilter,
                VectorIndexConfig, VectorIndexKind, VectorScoreMode, VectorSearchRequest,
            },
        },
    };

    const TITLE_FIELD: &str = "title_embedding";
    const BODY_FIELD: &str = "body_embedding";
    // const TITLE_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
    // const BODY_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

    pub fn run() -> Result<()> {
        println!("1) Configure storage + VectorEngine fields with an embedder registry\n");
        println!("   Loading Candle embedders (downloads may occur on first run)...\n");

        let storage =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default())) as Arc<dyn Storage>;

        let candle_text_embedder: Arc<dyn Embedder> = Arc::new(CandleTextEmbedder::new(
            "sentence-transformers/all-MiniLM-L6-v2",
        )?);
        let dimension: usize = 384; // 明示指定（モデルの出力次元）

        // let title_embedder: Arc<dyn Embedder> = Arc::new(CandleTextEmbedder::new(TITLE_MODEL)?);
        // let body_embedder: Arc<dyn Embedder> = Arc::new(CandleTextEmbedder::new(BODY_MODEL)?);
        // let title_dim = 384; // 各モデルの出力次元を明示指定
        // let body_dim = 384;

        // Configure PerFieldEmbedder so each vector field can transparently use Candle embedders.
        let mut per_field_embedder = PerFieldEmbedder::new(Arc::clone(&candle_text_embedder));
        per_field_embedder.add_embedder(TITLE_FIELD, Arc::clone(&candle_text_embedder));

        // Build config (PerFieldEmbedderはフィールド名で解決するためembedder_keyは不要)
        let config = VectorIndexConfig::builder()
            .field(
                TITLE_FIELD,
                VectorFieldConfig {
                    // dimension: title_dim,
                    dimension, // embedderのdimensionを共通利用するからわざわざフィールドで持つ必要ある？
                    distance: DistanceMetric::Cosine,
                    index: VectorIndexKind::Flat,
                    source_tag: TITLE_FIELD.to_string(),
                    vector_type: VectorType::Text,
                    base_weight: 1.2,
                },
            )
            .field(
                BODY_FIELD,
                VectorFieldConfig {
                    // dimension: body_dim,
                    dimension,
                    distance: DistanceMetric::Cosine,
                    index: VectorIndexKind::Flat,
                    source_tag: BODY_FIELD.to_string(),
                    vector_type: VectorType::Text,
                    base_weight: 1.0,
                },
            )
            .default_field(TITLE_FIELD)
            .default_field(BODY_FIELD)
            .embedder(per_field_embedder)
            .build()?;

        let engine = VectorEngine::new(storage, config)?;

        println!("2) Upsert documents with raw payloads that will be embedded automatically\n");

        // Document 1: Rust programming article
        let mut doc1 = DocumentPayload::new();
        doc1.set_metadata("lang", "en");
        doc1.set_metadata("category", "programming");
        doc1.set_text(TITLE_FIELD, "Rust overview");
        doc1.set_text(BODY_FIELD, "Rust balances performance with memory safety");

        // Document 2: LLM article in Japanese
        let mut doc2 = DocumentPayload::new();
        doc2.set_metadata("lang", "ja");
        doc2.set_metadata("category", "ai");
        doc2.set_text(TITLE_FIELD, "LLM primer");
        doc2.set_text(BODY_FIELD, "LLM internals");

        // Documents 3-5: User guide pages (chunked as separate documents)
        // In the flattened model, long documents should be split into separate documents
        // with metadata linking them to the original document.
        let mut doc3 = DocumentPayload::new();
        doc3.set_metadata("lang", "en");
        doc3.set_metadata("category", "manual");
        doc3.set_metadata("parent_doc_id", "user-guide");
        doc3.set_metadata("chunk_index", "0");
        doc3.set_text(
            BODY_FIELD,
            "Page 1: Installation steps for the runtime environment",
        );

        let mut doc4 = DocumentPayload::new();
        doc4.set_metadata("lang", "en");
        doc4.set_metadata("category", "manual");
        doc4.set_metadata("parent_doc_id", "user-guide");
        doc4.set_metadata("chunk_index", "1");
        doc4.set_text(
            BODY_FIELD,
            "Page 2: Configuration references and tuning knobs",
        );

        let mut doc5 = DocumentPayload::new();
        doc5.set_metadata("lang", "en");
        doc5.set_metadata("category", "manual");
        doc5.set_metadata("parent_doc_id", "user-guide");
        doc5.set_metadata("chunk_index", "2");
        doc5.set_text(
            BODY_FIELD,
            "Page 3: Troubleshooting common deployment issues",
        );

        engine.upsert_payloads(1, doc1)?;
        engine.upsert_payloads(2, doc2)?;
        engine.upsert_payloads(3, doc3)?;
        engine.upsert_payloads(4, doc4)?;
        engine.upsert_payloads(5, doc5)?;
        println!("   -> Inserted {} docs\n", engine.stats()?.document_count);

        println!("3) Build a VectorSearchRequest directly from query text\n");

        let mut doc_filter = VectorFilter::default();
        doc_filter
            .document
            .equals
            .insert("lang".into(), "en".into());

        let mut query = VectorSearchRequest::default();
        query.limit = 5;
        query.fields = Some(vec![
            FieldSelector::Exact(BODY_FIELD.into()),
            FieldSelector::Exact(TITLE_FIELD.into()),
        ]);
        query.filter = Some(doc_filter);
        query.score_mode = VectorScoreMode::WeightedSum;

        query.query_payloads.push(QueryPayload::new(
            TITLE_FIELD,
            Payload::text("systems programming"),
        ));

        query.query_payloads.push(QueryPayload::new(
            BODY_FIELD,
            Payload::text("memory safety"),
        ));

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
fn main() -> sarissa::error::Result<()> {
    candle_vector_example::run()
}
