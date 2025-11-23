//! Step-by-step doc-centric vector search example.
//!
//! Run with `cargo run --example vector_search`.

use std::collections::HashMap;
use std::sync::Arc;

use platypus::error::Result;
use platypus::storage::Storage;
use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use platypus::vector::DistanceMetric;
use platypus::vector::core::document::{DocumentVectors, FieldVectors, StoredVector, VectorRole};
use platypus::vector::engine::{
    FieldSelector, QueryVector, VectorEngine, VectorEngineConfig, VectorEngineFilter,
    VectorEngineQuery, VectorFieldConfig, VectorIndexKind, VectorScoreMode,
};

const DIM: usize = 4;
const EMBEDDER_ID: &str = "demo-encoder";
const TITLE_FIELD: &str = "title_embedding";
const BODY_FIELD: &str = "body_embedding";

fn main() -> Result<()> {
    println!("1) Configure storage + VectorEngine fields\n");
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
            base_weight: 1.0,
        },
    );
    let config = VectorEngineConfig {
        fields: field_configs,
        default_fields: vec![TITLE_FIELD.into(), BODY_FIELD.into()],
        metadata: HashMap::new(),
    };
    let engine = VectorEngine::new(config, storage, None)?;

    println!("2) Upsert documents where each doc owns multiple vector fields\n");
    let mut doc1 = DocumentVectors::new(1);
    doc1.metadata.insert("lang".into(), "en".into());
    doc1.metadata
        .insert("category".into(), "programming".into());
    doc1.fields.insert(
        TITLE_FIELD.into(),
        FieldVectors {
            vectors: vec![StoredVector {
                data: Arc::from([0.95_f32, 0.05, 0.0, 0.0]),
                embedder_id: EMBEDDER_ID.into(),
                role: VectorRole::Text,
                weight: 1.0,
                attributes: HashMap::from([(String::from("text"), String::from("Rust overview"))]),
            }],
            weight: 1.0,
            metadata: HashMap::from([(String::from("section"), String::from("title"))]),
        },
    );
    doc1.fields.insert(
        BODY_FIELD.into(),
        FieldVectors {
            vectors: vec![StoredVector {
                data: Arc::from([0.2_f32, 0.1, 0.65, 0.05]),
                embedder_id: EMBEDDER_ID.into(),
                role: VectorRole::Text,
                weight: 1.0,
                attributes: HashMap::from([(String::from("chunk"), String::from("rust-body"))]),
            }],
            weight: 1.0,
            metadata: HashMap::from([(String::from("section"), String::from("body"))]),
        },
    );

    let mut doc2 = DocumentVectors::new(2);
    doc2.metadata.insert("lang".into(), "ja".into());
    doc2.metadata.insert("category".into(), "ai".into());
    doc2.fields.insert(
        TITLE_FIELD.into(),
        FieldVectors {
            vectors: vec![StoredVector {
                data: Arc::from([0.1_f32, 0.85, 0.05, 0.0]),
                embedder_id: EMBEDDER_ID.into(),
                role: VectorRole::Text,
                weight: 1.0,
                attributes: HashMap::from([(String::from("text"), String::from("LLM primer"))]),
            }],
            weight: 0.9,
            metadata: HashMap::from([(String::from("section"), String::from("title"))]),
        },
    );
    doc2.fields.insert(
        BODY_FIELD.into(),
        FieldVectors {
            vectors: vec![StoredVector {
                data: Arc::from([0.1_f32, 0.15, 0.7, 0.05]),
                embedder_id: EMBEDDER_ID.into(),
                role: VectorRole::Text,
                weight: 1.0,
                attributes: HashMap::from([(String::from("chunk"), String::from("llm-body"))]),
            }],
            weight: 1.1,
            metadata: HashMap::from([(String::from("section"), String::from("body"))]),
        },
    );

    engine.upsert_document(doc1)?;
    engine.upsert_document(doc2)?;
    println!("   -> Inserted {} docs\n", engine.stats()?.document_count);

    println!("3) Build a VectorEngineQuery (pick fields, filters, limit)\n");
    let mut doc_filter = VectorEngineFilter::default();
    doc_filter
        .document
        .equals
        .insert("lang".into(), "en".into());
    let mut query = VectorEngineQuery::default();
    query.limit = 5;
    query.fields = Some(vec![
        FieldSelector::Exact(BODY_FIELD.into()),
        FieldSelector::Exact(TITLE_FIELD.into()),
    ]);
    query.filter = Some(doc_filter);
    query.score_mode = VectorScoreMode::WeightedSum;
    query.query_vectors.push(QueryVector {
        vector: StoredVector {
            data: Arc::from([0.9_f32, 0.05, 0.05, 0.0]),
            embedder_id: EMBEDDER_ID.into(),
            role: VectorRole::Text,
            weight: 1.0,
            attributes: HashMap::from([(
                String::from("text"),
                String::from("systems programming"),
            )]),
        },
        weight: 1.0,
    });

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
