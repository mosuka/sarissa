//! Flat vector index builder for exact search.

use crate::error::{Result, SarissaError};
use crate::vector::Vector;
use crate::vector_index::{VectorIndexBuildConfig, VectorIndexBuilder};
use rayon::prelude::*;

/// Builder for flat vector indexes (exact search).
pub struct FlatVectorIndexBuilder {
    config: VectorIndexBuildConfig,
    vectors: Vec<(u64, Vector)>,
    is_finalized: bool,
    total_vectors_to_add: Option<usize>,
}

impl FlatVectorIndexBuilder {
    /// Create a new flat vector index builder.
    pub fn new(config: VectorIndexBuildConfig) -> Result<Self> {
        Ok(Self {
            config,
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
        })
    }

    /// Set the expected total number of vectors (for progress tracking).
    pub fn set_expected_vector_count(&mut self, count: usize) {
        self.total_vectors_to_add = Some(count);
    }

    /// Get the stored vectors (for testing/debugging).
    pub fn vectors(&self) -> &[(u64, Vector)] {
        &self.vectors
    }

    /// Validate vectors before adding them.
    fn validate_vectors(&self, vectors: &[(u64, Vector)]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        // Check dimensions
        for (doc_id, vector) in vectors {
            if vector.dimension() != self.config.dimension {
                return Err(SarissaError::InvalidOperation(format!(
                    "Vector {} has dimension {}, expected {}",
                    doc_id,
                    vector.dimension(),
                    self.config.dimension
                )));
            }

            if !vector.is_valid() {
                return Err(SarissaError::InvalidOperation(format!(
                    "Vector {doc_id} contains invalid values (NaN or infinity)"
                )));
            }
        }

        Ok(())
    }

    /// Normalize vectors if configured to do so.
    fn normalize_vectors(&self, vectors: &mut [(u64, Vector)]) {
        if !self.config.normalize_vectors {
            return;
        }

        if self.config.parallel_build && vectors.len() > 100 {
            vectors.par_iter_mut().for_each(|(_, vector)| {
                vector.normalize();
            });
        } else {
            for (_, vector) in vectors {
                vector.normalize();
            }
        }
    }

    /// Check for memory limits.
    fn check_memory_limit(&self) -> Result<()> {
        if let Some(limit) = self.config.memory_limit {
            let current_usage = self.estimated_memory_usage();
            if current_usage > limit {
                return Err(SarissaError::ResourceExhausted(format!(
                    "Memory usage {current_usage} bytes exceeds limit {limit} bytes"
                )));
            }
        }
        Ok(())
    }

    /// Sort vectors by document ID for better cache locality.
    fn sort_vectors(&mut self) {
        if self.config.parallel_build && self.vectors.len() > 10000 {
            self.vectors.par_sort_by_key(|(doc_id, _)| *doc_id);
        } else {
            self.vectors.sort_by_key(|(doc_id, _)| *doc_id);
        }
    }

    /// Remove duplicate vectors (keeping the last one).
    fn deduplicate_vectors(&mut self) {
        if self.vectors.is_empty() {
            return;
        }

        // Sort first to group duplicates
        self.sort_vectors();

        // Remove duplicates, keeping the last occurrence
        let mut unique_vectors = Vec::new();
        let mut last_doc_id: Option<u64> = None;

        for (doc_id, vector) in std::mem::take(&mut self.vectors) {
            if last_doc_id != Some(doc_id) {
                unique_vectors.push((doc_id, vector));
                last_doc_id = Some(doc_id);
            } else {
                // Replace with newer vector
                if let Some((_, last_vector)) = unique_vectors.last_mut() {
                    *last_vector = vector;
                }
            }
        }

        self.vectors = unique_vectors;
    }
}

impl VectorIndexBuilder for FlatVectorIndexBuilder {
    fn build(&mut self, mut vectors: Vec<(u64, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Cannot build on finalized index".to_string(),
            ));
        }

        self.validate_vectors(&vectors)?;
        self.normalize_vectors(&mut vectors);

        self.vectors = vectors;
        self.total_vectors_to_add = Some(self.vectors.len());

        self.check_memory_limit()?;
        Ok(())
    }

    fn add_vectors(&mut self, mut vectors: Vec<(u64, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Cannot add vectors to finalized index".to_string(),
            ));
        }

        self.validate_vectors(&vectors)?;
        self.normalize_vectors(&mut vectors);

        self.vectors.extend(vectors);
        self.check_memory_limit()?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        if self.is_finalized {
            return Ok(());
        }

        // Remove duplicates and sort
        self.deduplicate_vectors();
        self.sort_vectors();

        self.is_finalized = true;
        Ok(())
    }

    fn progress(&self) -> f32 {
        if let Some(total) = self.total_vectors_to_add {
            if total == 0 {
                if self.is_finalized { 1.0 } else { 0.0 }
            } else {
                let current = self.vectors.len() as f32;
                let progress = current / total as f32;
                if self.is_finalized {
                    1.0
                } else {
                    progress.min(0.99) // Never report 100% until finalized
                }
            }
        } else if self.is_finalized {
            1.0
        } else {
            0.0
        }
    }

    fn estimated_memory_usage(&self) -> usize {
        let vector_memory = self.vectors.len()
            * (
                8 + // doc_id
            self.config.dimension * 4 + // f32 values
            std::mem::size_of::<Vector>()
                // Vector struct overhead
            );

        let metadata_memory = self.vectors.len() * 64; // Rough estimate for metadata

        vector_memory + metadata_memory
    }

    fn optimize(&mut self) -> Result<()> {
        if !self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Index must be finalized before optimization".to_string(),
            ));
        }

        // For flat index, optimization just means ensuring everything is sorted
        // and we've removed duplicates (already done in finalize)

        // Could also include memory compaction here
        self.vectors.shrink_to_fit();

        Ok(())
    }
}
