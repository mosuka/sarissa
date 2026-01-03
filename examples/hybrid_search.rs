use std::error::Error;
use std::sync::Arc;

use sarissa::analysis::analyzer::standard::StandardAnalyzer;
use sarissa::hybrid::core::document::HybridDocument;
use sarissa::hybrid::search::searcher::HybridSearchRequest;
use sarissa::hybrid::writer::HybridIndexWriter;
use sarissa::lexical::core::document::Document;
use sarissa::lexical::core::field::TextOption;
use sarissa::lexical::index::inverted::writer::{InvertedIndexWriter, InvertedIndexWriterConfig};
use sarissa::lexical::search::searcher::LexicalSearchRequest;
use sarissa::storage::memory::{MemoryStorage, MemoryStorageConfig};
use sarissa::vector::core::distance::DistanceMetric;
use sarissa::vector::core::document::StoredVector;
use sarissa::vector::engine::request::{QueryVector, VectorSearchRequest};
use sarissa::vector::index::config::HnswIndexConfig;
use sarissa::vector::index::hnsw::writer::HnswIndexWriter;
use sarissa::vector::writer::VectorIndexWriterConfig;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Hybrid Search Example");
    println!("=====================");

    // 1. Setup Storage
    // In this example, we use in-memory storage for simplicity.
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    // 2. Configure Writers
    // Lexical writer configuration (Inverted Index)
    let lexical_config = InvertedIndexWriterConfig {
        analyzer: Arc::new(StandardAnalyzer::new()?),
        ..Default::default()
    };

    // Vector writer configuration (HNSW Index)
    let vector_config = HnswIndexConfig {
        dimension: 3,
        distance_metric: DistanceMetric::Euclidean,
        ..Default::default()
    };

    let vector_writer_config = VectorIndexWriterConfig::default();

    // 3. Create Writers
    let lexical_writer = Box::new(InvertedIndexWriter::new(storage.clone(), lexical_config)?);
    let vector_writer = Box::new(HnswIndexWriter::with_storage(
        vector_config,
        vector_writer_config,
        "example_idx",
        storage.clone(),
    )?);

    // Create HybridIndexWriter
    // Note: We pass storage to manage shared metadata like document IDs
    let mut writer = HybridIndexWriter::new(lexical_writer, vector_writer, storage.clone())?;

    // 4. Add Documents
    println!("Adding specific documents...");

    // Doc 1: "apple banana", vector [1.0, 0.0, 0.0]
    let doc1 = Document::builder()
        .add_text("content", "apple banana", TextOption::default())
        .build();
    let mut payload1 = sarissa::vector::core::document::DocumentPayload::new();
    payload1.set_field(
        "vector",
        sarissa::vector::core::document::Payload::vector(std::sync::Arc::<[f32]>::from(vec![
            1.0, 0.0, 0.0,
        ])),
    );
    let hybrid_doc1 = HybridDocument::builder()
        .add_lexical_doc(doc1)
        .add_vector_payload(payload1)
        .build();
    let id1 = writer.add_document(hybrid_doc1)?;
    println!(
        "Added Document 1 (ID: {}) - 'apple banana', [1.0, 0.0, 0.0]",
        id1
    );

    // Doc 2: "banana orange", vector [0.0, 1.0, 0.0]
    let doc2 = Document::builder()
        .add_text("content", "banana orange", TextOption::default())
        .build();
    let mut payload2 = sarissa::vector::core::document::DocumentPayload::new();
    payload2.set_field(
        "vector",
        sarissa::vector::core::document::Payload::vector(std::sync::Arc::<[f32]>::from(vec![
            0.0, 1.0, 0.0,
        ])),
    );
    let hybrid_doc2 = HybridDocument::builder()
        .add_lexical_doc(doc2)
        .add_vector_payload(payload2)
        .build();
    let id2 = writer.add_document(hybrid_doc2)?;
    println!(
        "Added Document 2 (ID: {}) - 'banana orange', [0.0, 1.0, 0.0]",
        id2
    );

    // 5. Commit
    writer.commit()?;
    println!("Committed changes.");

    // 6. Build Reader (HybridIndex)
    let index = writer.build()?;
    println!("Built Hybrid Index.");

    // 7. Search / Verify
    println!("\n--- Hybrid Search ---");

    // Lexical Query part
    let lexical_req = LexicalSearchRequest::new("content:apple");

    // Vector Query part
    let query_vec_data = vec![0.95, 0.05, 0.0]; // Close to doc1
    let vector_req = VectorSearchRequest {
        query_vectors: vec![QueryVector {
            vector: StoredVector {
                data: std::sync::Arc::from(query_vec_data.as_slice()),
                weight: 1.0,
                attributes: Default::default(),
            },
            fields: Some(vec!["vector".to_string()]),
            weight: 1.0,
        }],
        ..Default::default()
    };

    // Hybrid Query part
    let search_request = HybridSearchRequest::new()
        .with_lexical_request(lexical_req)
        .with_vector_request(vector_req);

    // Execute Search (need async runtime)
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let results = rt.block_on(index.search(search_request))?;

    println!("Found {} results:", results.results.len());
    for (i, result) in results.results.iter().enumerate() {
        println!(
            "{}. Doc ID: {} (Score: {:.4})",
            i + 1,
            result.doc_id,
            result.hybrid_score
        );
        println!("   Keyword Score: {:?}", result.keyword_score);
        println!("   Vector Similarity: {:?}", result.vector_similarity);
    }

    Ok(())
}
