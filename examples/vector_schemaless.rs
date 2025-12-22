//! Schemaless-like vector search example using implicit schema generation.
//!
//! 未登録フィールドでもデフォルト embedder と暗黙スキーマ生成で自動登録する例。
//! `--features embeddings-candle` が必要。

#[cfg(not(feature = "embeddings-candle"))]
fn main() {
    eprintln!("This example requires the 'embeddings-candle' feature.");
    eprintln!(
        "Please rerun with: cargo run --example vector_schemaless --features embeddings-candle"
    );
    std::process::exit(1);
}

#[cfg(feature = "embeddings-candle")]
mod candle_schemaless_example {
    use std::sync::Arc;

    use platypus::{
        embedding::{candle_text_embedder::CandleTextEmbedder, embedder::Embedder},
        error::Result,
        storage::{Storage, memory::MemoryStorage},
        vector::{
            DistanceMetric,
            core::document::{DocumentPayload, Payload},
            engine::{
                FieldSelector, QueryPayload, VectorEngine, VectorIndexConfig, VectorScoreMode,
                VectorSearchRequest,
            },
        },
    };

    const TITLE: &str = "title";
    const BODY: &str = "body";

    pub fn run() -> Result<()> {
        println!("Schemaless-like vector search (implicit schema generation)\n");

        // 1) Storage & embedder
        let storage = Arc::new(MemoryStorage::default()) as Arc<dyn Storage>;
        let embedder: Arc<dyn Embedder> = Arc::new(CandleTextEmbedder::new(
            "sentence-transformers/all-MiniLM-L6-v2",
        )?);

        // 2) Config: implicit schema ON, no field definitions.
        let config = VectorIndexConfig::builder()
            .implicit_schema(true)
            .default_distance(DistanceMetric::Cosine)
            .embedder_arc(embedder)
            .build()?;

        let engine = VectorEngine::new(storage, config)?;

        // 3) Upsert payloads with unseen fields; fields will be auto-registered.
        let mut doc1 = DocumentPayload::new();
        doc1.set_text(TITLE, "Rust overview");
        doc1.set_text(BODY, "Rust balances performance with memory safety.");

        let mut doc2 = DocumentPayload::new();
        doc2.set_text(TITLE, "LLM primer");
        doc2.set_text(BODY, "Transformer models power modern LLMs.");

        engine.add_payloads(doc1)?;
        engine.add_payloads(doc2)?;

        // 4) Build query without predefined fields; use the same names as payloads.
        let mut query = VectorSearchRequest::default();
        query.limit = 5;
        query.fields = Some(vec![
            FieldSelector::Exact(BODY.into()),
            FieldSelector::Exact(TITLE.into()),
        ]);
        query.score_mode = VectorScoreMode::WeightedSum;

        query
            .query_payloads
            .push(QueryPayload::new(BODY, Payload::text("memory safety")));
        query
            .query_payloads
            .push(QueryPayload::new(TITLE, Payload::text("rust")));

        println!("Executing search...\n");
        let results = engine.search(query)?;
        for (rank, hit) in results.hits.iter().enumerate() {
            println!("{}. doc {} • score {:.3}", rank + 1, hit.doc_id, hit.score);
            for fh in &hit.field_hits {
                println!(
                    "   field {:<10} distance {:.3} score {:.3}",
                    fh.field, fh.distance, fh.score
                );
            }
        }

        Ok(())
    }
}

#[cfg(feature = "embeddings-candle")]
fn main() -> platypus::error::Result<()> {
    candle_schemaless_example::run()
}
