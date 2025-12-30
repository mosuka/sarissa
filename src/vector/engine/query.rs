//! Builder for VectorSearchRequest.
//!
//! This module provides a fluent API for constructing vector search requests.

use std::sync::Arc;

use crate::vector::core::document::{Payload, PayloadSource, StoredVector};
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
///     .add_vector("content", vec![0.1, 0.2, 0.3])
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

    /// Add a raw query vector for a specific field.
    pub fn add_vector(mut self, field: impl Into<String>, vector: Vec<f32>) -> Self {
        self.request.query_vectors.push(QueryVector {
            vector: StoredVector::new(Arc::<[f32]>::from(vector.as_slice())),
            weight: 1.0,
            fields: Some(vec![field.into()]),
        });
        self
    }

    /// Add a raw query vector with explicit weight for a specific field.
    pub fn add_vector_with_weight(
        mut self,
        field: impl Into<String>,
        vector: Vec<f32>,
        weight: f32,
    ) -> Self {
        self.request.query_vectors.push(QueryVector {
            vector: StoredVector::new(Arc::<[f32]>::from(vector.as_slice())).with_weight(weight),
            weight,
            fields: Some(vec![field.into()]),
        });
        self
    }

    /// Add a payload to be embedded.
    ///
    /// This is the unified method for all modalities (text, image, video, etc.).
    /// The bytes will be processed by the configured embedder.
    ///
    /// # Arguments
    ///
    /// * `field` - The target field name
    /// * `bytes` - Raw bytes of the content (text as UTF-8, image bytes, etc.)
    ///
    /// # Example
    ///
    /// ```
    /// use sarissa::vector::engine::VectorSearchRequestBuilder;
    ///
    /// // Text search
    /// let request = VectorSearchRequestBuilder::new()
    ///     .add_payload("title", "search query".as_bytes())
    ///     .build();
    ///
    /// // Image search (with image bytes)
    /// let image_bytes: Vec<u8> = vec![0x89, 0x50, 0x4E, 0x47]; // PNG header
    /// let request = VectorSearchRequestBuilder::new()
    ///     .add_payload("image", image_bytes)
    ///     .build();
    /// ```
    pub fn add_payload(mut self, field: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Self {
        let payload = Payload::new(PayloadSource::bytes(bytes.into(), None));

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

    /// Add a field to search in.
    ///
    /// This is a convenience method to add a single field.
    pub fn field(mut self, field: impl Into<String>) -> Self {
        let field = field.into();
        if let Some(fields) = &mut self.request.fields {
            fields.push(FieldSelector::Exact(field));
        } else {
            self.request.fields = Some(vec![FieldSelector::Exact(field)]);
        }
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
