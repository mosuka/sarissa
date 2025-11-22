//! Minimal VectorCollection query example without legacy adapters.
//!
//! This sample builds an in-memory `VectorCollection`, ingests a handful of
//! demo documents, and executes `VectorCollectionQuery` instances directly. It
//! illustrates field-scoped queries, metadata-driven previews, and score-mode
//! defaults without relying on `VectorSearchRequest` or other legacy APIs.

use std::collections::HashMap;
use std::sync::Arc;

use platypus::error::Result;
use platypus::storage::Storage;
use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use platypus::vector::DistanceMetric;
use platypus::vector::collection::{
    FieldSelector, MetadataFilter, QueryVector, VectorCollection, VectorCollectionConfig,
    VectorCollectionFilter, VectorCollectionQuery, VectorCollectionSearchResults,
    VectorFieldConfig, VectorIndexKind, VectorScoreMode,
};
use platypus::vector::core::document::{DocumentVectors, StoredVector, VectorRole};
use platypus::vector::core::vector::ORIGINAL_TEXT_METADATA_KEY;

const DIMENSION: usize = 4;
const EMBEDDER_ID: &str = "text-encoder-v1";
const TITLE_FIELD: &str = "title_embedding";
const CONTENT_FIELD: &str = "body_embedding";
const SAMPLE_VECTORS_JSON: &str = include_str!("../resources/vector_collection_sample.json");

fn main() -> Result<()> {
    println!("=== VectorCollection query demo ===\n");

    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default())) as Arc<dyn Storage>;
    let config = build_collection_config();
    let collection = VectorCollection::new(config, storage, None)?;

    ingest_documents(&collection)?;
    demo_queries(&collection)?;

    Ok(())
}

fn build_collection_config() -> VectorCollectionConfig {
    let mut fields = HashMap::new();
    fields.insert(
        TITLE_FIELD.to_string(),
        VectorFieldConfig {
            dimension: DIMENSION,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: EMBEDDER_ID.to_string(),
            role: VectorRole::Text,
            base_weight: 1.2,
        },
    );
    fields.insert(
        CONTENT_FIELD.to_string(),
        VectorFieldConfig {
            dimension: DIMENSION,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: EMBEDDER_ID.to_string(),
            role: VectorRole::Text,
            base_weight: 1.0,
        },
    );

    VectorCollectionConfig {
        fields,
        default_fields: vec![TITLE_FIELD.into(), CONTENT_FIELD.into()],
        metadata: HashMap::new(),
    }
}

fn ingest_documents(collection: &VectorCollection) -> Result<()> {
    let documents: Vec<DocumentVectors> = serde_json::from_str(SAMPLE_VECTORS_JSON)?;
    let total = documents.len();
    for document in documents {
        collection.upsert_document(document)?;
    }

    println!("Inserted demo documents ({} total).\n", total);
    Ok(())
}

fn demo_queries(collection: &VectorCollection) -> Result<()> {
    println!("Running VectorCollection queries...\n");

    run_query(
        collection,
        make_query(
            [0.92, 0.08, 0.0, 0.0],
            None,
            None,
            VectorScoreMode::WeightedSum,
        ),
        "All fields • programming focus",
    )?;

    let doc_filter = VectorCollectionFilter {
        document: metadata_filter(&[("lang", "ja")]),
        field: MetadataFilter::default(),
    };

    run_query(
        collection,
        make_query(
            [0.2, 0.1, 0.9, 0.05],
            None,
            Some(doc_filter),
            VectorScoreMode::WeightedSum,
        ),
        "Document metadata filter • lang=ja",
    )?;

    let field_filter = VectorCollectionFilter {
        document: MetadataFilter::default(),
        field: metadata_filter(&[("section", "body")]),
    };

    run_query(
        collection,
        make_query(
            [0.1, 0.1, 0.85, 0.1],
            Some(vec![FieldSelector::Exact(CONTENT_FIELD.into())]),
            Some(field_filter),
            VectorScoreMode::MaxSim,
        ),
        "Content-only • section=body filter • MaxSim",
    )?;

    Ok(())
}

fn make_query(
    data: [f32; DIMENSION],
    fields: Option<Vec<FieldSelector>>,
    filter: Option<VectorCollectionFilter>,
    score_mode: VectorScoreMode,
) -> VectorCollectionQuery {
    let mut query = VectorCollectionQuery::default();
    query.limit = 3;
    query.fields = fields;
    query.filter = filter;
    query.score_mode = score_mode;
    query.query_vectors.push(query_vector(data, "demo query"));
    query
}

fn run_query(
    collection: &VectorCollection,
    query: VectorCollectionQuery,
    label: &str,
) -> Result<()> {
    println!("{}", "-".repeat(72));
    println!("{label}");
    let results = collection.search(&query)?;
    display_results(&results, label);
    Ok(())
}

fn display_results(results: &VectorCollectionSearchResults, context: &str) {
    if results.hits.is_empty() {
        println!("  No hits for {context}.\n");
        return;
    }

    println!("  {} aggregated hits", results.hits.len());
    for (rank, hit) in results.hits.iter().enumerate() {
        println!(
            "  {}. doc #{:02} • score {:.3}",
            rank + 1,
            hit.doc_id,
            hit.score
        );
        for field_hit in &hit.field_hits {
            println!(
                "     field {:<7} • score {:.3} • distance {:.3}",
                field_hit.field, field_hit.score, field_hit.distance
            );
        }
    }
    println!();
}

fn query_vector(data: [f32; DIMENSION], text: &str) -> QueryVector {
    let mut vector = StoredVector::new(Arc::from(data), EMBEDDER_ID.to_string(), VectorRole::Text);
    vector
        .attributes
        .insert(ORIGINAL_TEXT_METADATA_KEY.to_string(), text.to_string());
    QueryVector {
        vector,
        weight: 1.0,
    }
}

fn metadata_filter(pairs: &[(&str, &str)]) -> MetadataFilter {
    let mut filter = MetadataFilter::default();
    for (key, value) in pairs {
        filter
            .equals
            .insert((*key).to_string(), (*value).to_string());
    }
    filter
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_search_demo_runs() {
        let result = main();
        assert!(result.is_ok());
    }
}
