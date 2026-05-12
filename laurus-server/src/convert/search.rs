//! Conversion between search-related laurus domain types and protobuf types.
//!
//! [`from_proto`] builds a [`laurus::SearchRequest`] from the incoming proto
//! message, mapping the `query` field to [`SearchQuery::Dsl`] so the engine
//! can parse unified query DSL (including vector clauses) internally.
//! [`result_to_proto`] converts engine results back to proto.

use laurus::vector::Vector;
use laurus::{
    FusionAlgorithm, LexicalSearchQuery, QueryVector, SearchRequestBuilder, SearchResult,
    SortField, SortOrder, VectorScoreMode, VectorSearchQuery,
};

use crate::convert::document;
use crate::proto::laurus::v1;

/// Build a laurus SearchRequest from a proto SearchRequest.
///
/// The proto `query` field is mapped to [`SearchQuery::Dsl`] so the engine
/// handles unified query DSL parsing (including vector clauses) internally.
///
/// When `lexical_params` or `field_boosts` are provided, the query is
/// wrapped as [`LexicalSearchQuery::Dsl`] and lexical options are set
/// directly on the builder.
#[allow(clippy::result_large_err)]
pub fn from_proto(proto: &v1::SearchRequest) -> Result<laurus::SearchRequest, tonic::Status> {
    let mut builder = SearchRequestBuilder::new();

    let has_lexical_overrides = proto.lexical_params.is_some() || !proto.field_boosts.is_empty();

    if !proto.query.is_empty() {
        if has_lexical_overrides {
            // Wrap query as LexicalSearchQuery::Dsl so that lexical options
            // are preserved via builder methods.
            builder = builder.lexical_query(LexicalSearchQuery::Dsl(proto.query.clone()));

            // Apply field boosts
            for (field, boost) in &proto.field_boosts {
                builder = builder.add_field_boost(field.clone(), *boost);
            }

            // Apply lexical params
            if let Some(p) = &proto.lexical_params {
                builder = builder.lexical_min_score(p.min_score);
                if let Some(timeout_ms) = p.timeout_ms
                    && timeout_ms > 0
                {
                    builder = builder.lexical_timeout_ms(timeout_ms);
                }
                if p.parallel {
                    builder = builder.lexical_parallel(true);
                }
                if let Some(spec) = &p.sort_by
                    && !spec.field.is_empty()
                {
                    let order = match v1::SortOrder::try_from(spec.order) {
                        Ok(v1::SortOrder::Desc) => SortOrder::Desc,
                        _ => SortOrder::Asc,
                    };
                    builder = builder.sort_by(SortField::Field {
                        name: spec.field.clone(),
                        order,
                    });
                }
            }
        } else {
            // Use the DSL variant — engine will parse with UnifiedQueryParser
            builder = builder.query_dsl(proto.query.clone());
        }
    }

    // Explicit pre-embedded vectors
    if !proto.query_vectors.is_empty() {
        let query_vectors: Vec<QueryVector> = proto
            .query_vectors
            .iter()
            .map(|qv| QueryVector {
                vector: Vector::new(qv.vector.clone()),
                weight: if qv.weight == 0.0 { 1.0 } else { qv.weight },
                fields: if qv.fields.is_empty() {
                    None
                } else {
                    Some(qv.fields.clone())
                },
            })
            .collect();

        builder = builder.vector_query(VectorSearchQuery::Vectors(query_vectors));

        // Apply vector params
        if let Some(vp) = &proto.vector_params {
            let score_mode = match v1::VectorScoreMode::try_from(vp.score_mode) {
                Ok(v1::VectorScoreMode::MaxSim) => VectorScoreMode::MaxSim,
                Ok(v1::VectorScoreMode::LateInteraction) => VectorScoreMode::LateInteraction,
                _ => VectorScoreMode::WeightedSum,
            };
            builder = builder.vector_score_mode(score_mode);
            if vp.min_score > 0.0 {
                builder = builder.vector_min_score(vp.min_score);
            }
            // Issue #481 Stage 2: forward rerank_factor to the engine
            // (which forwards it to the HNSW searcher). Per-field
            // capability checks happen later: HNSW fields with
            // rerank_storage configured honor the value; everything
            // else silently ignores it. A zero value disables rerank
            // (defensive: matches `None` semantics).
            if let Some(factor) = vp.rerank_factor
                && factor > 0
            {
                builder = builder.vector_rerank_factor(factor as usize);
            }
        }
    }

    // Limit and offset
    if proto.limit > 0 {
        builder = builder.limit(proto.limit as usize);
    }
    builder = builder.offset(proto.offset as usize);

    // Fusion
    if let Some(fusion) = &proto.fusion
        && let Some(alg) = &fusion.algorithm
    {
        let fusion_alg = match alg {
            v1::fusion_algorithm::Algorithm::Rrf(rrf) => FusionAlgorithm::RRF { k: rrf.k },
            v1::fusion_algorithm::Algorithm::WeightedSum(ws) => FusionAlgorithm::WeightedSum {
                lexical_weight: ws.lexical_weight,
                vector_weight: ws.vector_weight,
            },
        };
        builder = builder.fusion_algorithm(fusion_alg);
    }

    Ok(builder.build())
}

/// Convert a laurus SearchResult into a proto SearchResult.
pub fn result_to_proto(result: &SearchResult) -> v1::SearchResult {
    v1::SearchResult {
        id: result.id.clone(),
        score: result.score,
        document: result.document.as_ref().map(document::to_proto),
    }
}
