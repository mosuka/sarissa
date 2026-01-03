//! Hybrid index writer.
//!
//! This module provides the writer interface for building hybrid indexes
//! that combine lexical and vector data.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SarissaError};
use crate::hybrid::core::document::HybridDocument;
use crate::hybrid::index::HybridIndex;
use crate::lexical::writer::LexicalIndexWriter;
use crate::storage::Storage;
use crate::vector::core::vector::Vector;
use crate::vector::writer::VectorIndexWriter;

/// Writer for building hybrid indexes.
///
/// This writer coordinates writes to both lexical and vector indexes,
/// ensuring consistency between the two.
pub struct HybridIndexWriter {
    /// Writer for the lexical index
    pub lexical_writer: Box<dyn LexicalIndexWriter>,
    /// Writer for the vector index
    pub vector_writer: Box<dyn VectorIndexWriter>,
    /// Storage for managing hybrid metadata
    storage: Arc<dyn Storage>,
    /// Document ID counter
    next_doc_id: u64,
}

/// Metadata for the hybrid index.
#[derive(Serialize, Deserialize)]
struct HybridIndexMetadata {
    next_doc_id: u64,
}

impl HybridIndexWriter {
    /// Create a new hybrid index writer.
    ///
    /// # Arguments
    ///
    /// * `lexical_writer` - Writer for the lexical index
    /// * `vector_writer` - Writer for the vector index
    /// * `storage` - Storage backend
    ///
    /// # Returns
    ///
    /// A new `HybridIndexWriter` instance
    pub fn new(
        lexical_writer: Box<dyn LexicalIndexWriter>,
        vector_writer: Box<dyn VectorIndexWriter>,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        let mut writer = Self {
            lexical_writer,
            vector_writer,
            storage,
            next_doc_id: 0,
        };
        writer.load_metadata()?;
        Ok(writer)
    }

    fn load_metadata(&mut self) -> Result<()> {
        if self.storage.file_exists("hybrid_meta.json") {
            let reader = self.storage.open_input("hybrid_meta.json")?;
            let mut content = String::new();
            std::io::Read::read_to_string(&mut std::io::BufReader::new(reader), &mut content)?;
            let metadata: HybridIndexMetadata = serde_json::from_str(&content).map_err(|e| {
                SarissaError::index(format!("Failed to parse hybrid metadata: {e}"))
            })?;
            self.next_doc_id = metadata.next_doc_id;
        }
        Ok(())
    }

    fn save_metadata(&mut self) -> Result<()> {
        let metadata = HybridIndexMetadata {
            next_doc_id: self.next_doc_id,
        };
        let content = serde_json::to_string(&metadata).map_err(|e| {
            SarissaError::index(format!("Failed to serialize hybrid metadata: {e}"))
        })?;
        let mut writer = self.storage.create_output("hybrid_meta.json")?;
        std::io::Write::write_all(&mut writer, content.as_bytes())?;
        writer.flush()?;
        Ok(())
    }

    /// Add a document to the hybrid index.
    ///
    /// This adds the lexical part to the lexical index and the vector part (if any)
    /// to the vector index.
    ///
    /// # Arguments
    ///
    /// * `doc` - The hybrid document to add
    ///
    /// # Returns
    ///
    /// The assigned document ID (from the lexical writer)
    pub fn add_document(&mut self, doc: HybridDocument) -> Result<u64> {
        // Use locally managed document ID
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        // Add to lexical index
        // We use upsert to enforce the specific ID.
        if let Some(lexical_doc) = doc.lexical_doc {
            self.lexical_writer.upsert_document(doc_id, lexical_doc)?;
        }

        // Add to vector index if payload is present
        if let Some(payload) = doc.vector_payload {
            let mut vectors = Vec::new();
            for (field_name, field_payload) in payload.fields {
                match field_payload.source {
                    crate::vector::core::document::PayloadSource::Vector { data } => {
                        let vector = Vector::new(data.to_vec());
                        vectors.push((doc_id, field_name, vector));
                    }
                    crate::vector::core::document::PayloadSource::Text { .. }
                    | crate::vector::core::document::PayloadSource::Bytes { .. } => {
                        // For now we only support pre-embedded vectors in the writer.
                        return Err(SarissaError::InvalidOperation(format!(
                            "Unembedded payload in field '{}'. HybridIndexWriter expects vectors.",
                            field_name
                        )));
                    }
                }
            }
            if !vectors.is_empty() {
                self.vector_writer.add_vectors(vectors)?;
            }
        }

        Ok(doc_id)
    }

    /// Commit changes to both indexes.
    pub fn commit(&mut self) -> Result<()> {
        self.lexical_writer.commit()?;
        self.vector_writer.commit()?;
        self.save_metadata()?;
        Ok(())
    }

    /// Finalize both indexes.
    ///
    /// This method ensures both the lexical and vector indexes are properly
    /// finalized and ready for search operations.
    ///
    /// # Returns
    ///
    /// `Ok(())` if finalization succeeds
    ///
    /// # Errors
    ///
    /// Returns an error if vector index finalization fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use sarissa::hybrid::writer::HybridIndexWriter;
    /// # use sarissa::lexical::writer::LexicalIndexWriter;
    /// # use sarissa::vector::writer::VectorIndexWriter;
    /// # fn example(lexical: Box<dyn LexicalIndexWriter>, vector: Box<dyn VectorIndexWriter>) -> sarissa::error::Result<()> {
    /// let mut writer = HybridIndexWriter::new(lexical, vector);
    /// // ... write documents ...
    /// writer.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn finalize(&mut self) -> Result<()> {
        self.vector_writer.finalize()?;
        // Lexical writer finalization is handled separately
        Ok(())
    }

    /// Build a hybrid index from the written data.
    ///
    /// # Returns
    ///
    /// A fully constructed hybrid index ready for search operations
    ///
    /// # Note
    ///
    /// This is currently a placeholder. Full implementation will construct
    /// the indexes from the writers.
    pub fn build(self) -> Result<HybridIndex> {
        let lexical_index = self.lexical_writer.build_reader()?;
        let vector_index = self.vector_writer.build_reader()?;

        Ok(HybridIndex::new(lexical_index, vector_index))
    }
}
