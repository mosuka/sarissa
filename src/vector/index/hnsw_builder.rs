//! HNSW (Hierarchical Navigable Small World) index builder for approximate search.

use crate::error::{Result, SageError};
use crate::vector::Vector;
use crate::vector::index::{VectorIndexBuildConfig, VectorIndexBuilder};

/// Builder for HNSW vector indexes (approximate search).
pub struct HnswIndexBuilder {
    config: VectorIndexBuildConfig,
    m: usize,               // Maximum number of connections per layer
    ef_construction: usize, // Size of the dynamic candidate list during construction
    _ml: f64,               // Level normalization factor
    vectors: Vec<(u64, Vector)>,
    is_finalized: bool,
    total_vectors_to_add: Option<usize>,
}

impl HnswIndexBuilder {
    /// Create a new HNSW index builder.
    pub fn new(config: VectorIndexBuildConfig) -> Result<Self> {
        Ok(Self {
            config,
            m: 16,                     // Default M parameter
            ef_construction: 200,      // Default efConstruction parameter
            _ml: 1.0 / (2.0_f64).ln(), // 1/ln(2)
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
        })
    }

    /// Set HNSW-specific parameters.
    pub fn with_hnsw_params(mut self, m: usize, ef_construction: usize) -> Self {
        self.m = m;
        self.ef_construction = ef_construction;
        self
    }

    /// Set the expected total number of vectors (for progress tracking).
    pub fn set_expected_vector_count(&mut self, count: usize) {
        self.total_vectors_to_add = Some(count);
    }

    /// Calculate the layer for a new vector.
    #[allow(dead_code)]
    fn select_layer(&self) -> usize {
        let mut layer = 0;
        let mut rng = rand::rng();

        while rand::Rng::random::<f64>(&mut rng) < 0.5 && layer < 16 {
            layer += 1;
        }

        layer
    }

    /// Validate vectors before adding them.
    fn validate_vectors(&self, vectors: &[(u64, Vector)]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        for (doc_id, vector) in vectors {
            if vector.dimension() != self.config.dimension {
                return Err(SageError::InvalidOperation(format!(
                    "Vector {} has dimension {}, expected {}",
                    doc_id,
                    vector.dimension(),
                    self.config.dimension
                )));
            }

            if !vector.is_valid() {
                return Err(SageError::InvalidOperation(format!(
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

        use rayon::prelude::*;

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

    /// Build the HNSW graph structure (placeholder implementation).
    fn build_hnsw_graph(&mut self) -> Result<()> {
        // This is a placeholder for the actual HNSW graph construction
        // Real implementation would:
        // 1. Create layered graph structure
        // 2. For each vector, determine its layer
        // 3. Insert vector and create connections using greedy search
        // 4. Maintain M connections per layer with pruning

        println!("Building HNSW graph with {} vectors", self.vectors.len());
        println!(
            "Parameters: M={}, efConstruction={}",
            self.m, self.ef_construction
        );

        // Placeholder: just sort vectors by ID for now
        self.vectors.sort_by_key(|(doc_id, _)| *doc_id);

        Ok(())
    }

    /// Check for memory limits.
    fn check_memory_limit(&self) -> Result<()> {
        if let Some(limit) = self.config.memory_limit {
            let current_usage = self.estimated_memory_usage();
            if current_usage > limit {
                return Err(SageError::ResourceExhausted(format!(
                    "Memory usage {current_usage} bytes exceeds limit {limit} bytes"
                )));
            }
        }
        Ok(())
    }

    /// Get the stored vectors (for testing/debugging).
    pub fn vectors(&self) -> &[(u64, Vector)] {
        &self.vectors
    }

    /// Get HNSW parameters.
    pub fn hnsw_params(&self) -> (usize, usize) {
        (self.m, self.ef_construction)
    }
}

impl VectorIndexBuilder for HnswIndexBuilder {
    fn build(&mut self, mut vectors: Vec<(u64, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(SageError::InvalidOperation(
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
            return Err(SageError::InvalidOperation(
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

        // Build the actual HNSW graph structure
        self.build_hnsw_graph()?;

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

        // HNSW graph overhead (rough estimate)
        // Each vector can have up to M connections per layer
        // Average layers per vector is approximately 1/(1-p) where p=0.5
        let avg_layers = 2.0;
        let graph_memory = self.vectors.len() * (self.m as f32 * avg_layers * 8.0) as usize;

        let metadata_memory = self.vectors.len() * 128; // Increased for graph structure

        vector_memory + graph_memory + metadata_memory
    }

    fn optimize(&mut self) -> Result<()> {
        if !self.is_finalized {
            return Err(SageError::InvalidOperation(
                "Index must be finalized before optimization".to_string(),
            ));
        }

        // HNSW optimization could include:
        // 1. Graph pruning to remove low-quality connections
        // 2. Memory compaction
        // 3. Connection rebalancing

        println!("Optimizing HNSW index...");

        // For now, just compact memory
        self.vectors.shrink_to_fit();

        Ok(())
    }

    fn vectors(&self) -> &[(u64, Vector)] {
        &self.vectors
    }
}
