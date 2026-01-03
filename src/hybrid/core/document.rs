use crate::lexical::core::document::Document as LexicalDocument;
use crate::vector::core::document::DocumentPayload as VectorPayload;
use serde::{Deserialize, Serialize};

/// A document that contains both lexical and vector data.
///
/// This structure encapsulates all data needed to index a document in the hybrid engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridDocument {
    /// Document ID (if known/assigned externally, though usually assigned by engine).
    /// Note: HybridEngine assigns its own ID, so this might be for reference.
    /// However, for consistency with other parts, we might not strictly need it here
    /// if add_document returns ID. But strict mapping requires us to handle it.
    /// In this implementation, we just hold the content.

    /// The lexical component of the document (optional).
    pub lexical_doc: Option<LexicalDocument>,

    /// The vector component of the document (optional).
    /// If provided, this payload will be embedded and indexed in the vector engine.
    pub vector_payload: Option<VectorPayload>,
}

impl HybridDocument {
    /// Create a new hybrid document from a lexical document and optional vector payload.
    pub fn new(
        lexical_doc: Option<LexicalDocument>,
        vector_payload: Option<VectorPayload>,
    ) -> Self {
        Self {
            lexical_doc,
            vector_payload,
        }
    }

    /// Create a new hybrid document with only lexical content.
    pub fn from_lexical(lexical_doc: LexicalDocument) -> Self {
        Self {
            lexical_doc: Some(lexical_doc),
            vector_payload: None,
        }
    }

    /// Create a new hybrid document with only vector payload.
    pub fn from_vector_payload(vector_payload: VectorPayload) -> Self {
        Self {
            lexical_doc: None,
            vector_payload: Some(vector_payload),
        }
    }

    /// Returns a new builder for `HybridDocument`.
    pub fn builder() -> HybridDocumentBuilder {
        HybridDocumentBuilder::new()
    }
}

/// Builder for `HybridDocument`.
#[derive(Default)]
pub struct HybridDocumentBuilder {
    lexical_doc: Option<LexicalDocument>,
    vector_payload: Option<VectorPayload>,
}

impl HybridDocumentBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a lexical document.
    pub fn add_lexical_doc(mut self, doc: LexicalDocument) -> Self {
        self.lexical_doc = Some(doc);
        self
    }

    /// Add a vector payload.
    pub fn add_vector_payload(mut self, payload: VectorPayload) -> Self {
        self.vector_payload = Some(payload);
        self
    }

    /// Build the `HybridDocument`.
    pub fn build(self) -> HybridDocument {
        HybridDocument {
            lexical_doc: self.lexical_doc,
            vector_payload: self.vector_payload,
        }
    }
}
