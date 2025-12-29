//! Builder for VectorSearchRequest.
//!
//! This module provides a fluent API for constructing vector search requests.

use std::sync::Arc;

use crate::vector::core::document::{Payload, StoredVector, VectorType};
use crate::vector::engine::filter::VectorFilter;
use crate::vector::engine::request::{
    FieldSelector, QueryPayload, QueryVector, VectorScoreMode, VectorSearchRequest,
};

/// Builder for constructing VectorSearchRequest.
///
/// # Example
///
/// ```
/// use sarissa::vector::engine::VectorSearchRequestBuilder;
///
/// let request = VectorSearchRequestBuilder::new()
///     .add_vector(vec![0.1, 0.2, 0.3])
///     .limit(5)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct VectorSearchRequestBuilder {
    request: VectorSearchRequest,
}

impl VectorSearchRequestBuilder {
    /// Create a new VectorSearchRequestBuilder.
    pub fn new() -> Self {
        Self {
            request: VectorSearchRequest::default(),
        }
    }

    /// Add a raw query vector.
    ///
    /// The vector type defaults to `VectorType::Text`.
    pub fn add_vector(mut self, vector: Vec<f32>) -> Self {
        self.request.query_vectors.push(QueryVector {
            vector: StoredVector::new(Arc::<[f32]>::from(vector.as_slice()), VectorType::Text),
            weight: 1.0,
        });
        self
    }

    /// Add a raw query vector with explicit type and weight.
    pub fn add_vector_with_options(
        mut self,
        vector: Vec<f32>,
        vector_type: VectorType,
        weight: f32,
    ) -> Self {
        self.request.query_vectors.push(QueryVector {
            vector: StoredVector::new(Arc::<[f32]>::from(vector.as_slice()), vector_type),
            weight,
        });
        self
    }

    /// Add a text query to be embedded.
    ///
    /// This requires an embedder to be configured in the VectorEngine.
    pub fn add_text(mut self, field: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        // Use the helper method which handles PayloadSource details
        let payload = Payload::text(text);

        self.request
            .query_payloads
            .push(QueryPayload::new(field, payload));
        self
    }

    /// Set the fields to search in.
    pub fn fields(mut self, fields: Vec<String>) -> Self {
        self.request.fields = Some(fields.into_iter().map(FieldSelector::Exact).collect());
        self
    }

    /// Set the search limit.
    pub fn limit(mut self, limit: usize) -> Self {
        self.request.limit = limit;
        self
    }

    /// Set the score mode.
    pub fn score_mode(mut self, mode: VectorScoreMode) -> Self {
        self.request.score_mode = mode;
        self
    }

    /// Set the overfetch factor.
    pub fn overfetch(mut self, overfetch: f32) -> Self {
        self.request.overfetch = overfetch;
        self
    }

    /// Set a filter for the query.
    pub fn filter(mut self, filter: VectorFilter) -> Self {
        self.request.filter = Some(filter);
        self
    }

    /// Set the minimum score threshold.
    pub fn min_score(mut self, min_score: f32) -> Self {
        self.request.min_score = min_score;
        self
    }

    /// Build the VectorSearchRequest.
    pub fn build(self) -> VectorSearchRequest {
        self.request
    }
}
