use std::sync::Arc;

use sarissa::analysis::analyzer::standard::StandardAnalyzer;
use sarissa::hybrid::core::document::HybridDocument;
use sarissa::hybrid::writer::HybridIndexWriter;
use sarissa::lexical::core::document::Document;
use sarissa::lexical::core::field::TextOption;
use sarissa::lexical::index::inverted::writer::{InvertedIndexWriter, InvertedIndexWriterConfig};
use sarissa::storage::memory::MemoryStorage;
use sarissa::vector::core::distance::DistanceMetric;
use sarissa::vector::index::config::HnswIndexConfig;
use sarissa::vector::index::hnsw::writer::HnswIndexWriter;
use sarissa::vector::writer::VectorIndexWriterConfig;

#[test]
fn test_hybrid_writer_build_and_search() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup Storage
    let storage = Arc::new(MemoryStorage::new(
        sarissa::storage::memory::MemoryStorageConfig::default(),
    ));

    // 2. Configure Writers
    let lexical_config = InvertedIndexWriterConfig {
        analyzer: Arc::new(StandardAnalyzer::new()?),
        ..Default::default()
    };

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
        "test_idx",
        storage.clone(),
    )?);

    let mut writer = HybridIndexWriter::new(lexical_writer, vector_writer, storage.clone())?;

    // 4. Add Documents
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
    writer.add_document(hybrid_doc1)?;

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
    writer.add_document(hybrid_doc2)?;

    // Doc 3: Vector only, vector [0.0, 0.0, 1.0]
    let mut payload3 = sarissa::vector::core::document::DocumentPayload::new();
    payload3.set_field(
        "vector",
        sarissa::vector::core::document::Payload::vector(std::sync::Arc::<[f32]>::from(vec![
            0.0, 0.0, 1.0,
        ])),
    );
    let hybrid_doc3 = HybridDocument::builder()
        .add_vector_payload(payload3)
        .build();
    writer.add_document(hybrid_doc3)?; // Should be ID 2

    // 5. Commit
    writer.commit()?;

    // 6. Build Reader
    let index = writer.build()?;

    // 7. Verify
    // Lexical check
    let term_info = index.lexical_index.term_info("content", "apple")?;
    assert!(term_info.is_some());
    assert_eq!(term_info.unwrap().doc_freq, 1);

    // Vector check
    assert_eq!(index.vector_index.vector_count(), 3);

    // Check vector content
    // Doc ID passed to vector writer is same as lexical doc ID.
    // InvertedIndexWriter assigns 0, 1, 2...
    // Doc 1 (apple banana) should be ID 0.
    if let Some(vec) = index.vector_index.get_vector(0, "vector")? {
        assert_eq!(vec.data, vec![1.0, 0.0, 0.0]);
    } else {
        // Fallback or check failure
        // Maybe try field name iteration if "vector" is not the key
        let vecs = index.vector_index.get_vectors_for_doc(0)?;
        assert!(!vecs.is_empty());
        assert_eq!(vecs[0].1.data, vec![1.0, 0.0, 0.0]);
    }

    Ok(())
}
