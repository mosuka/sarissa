//! IVF (Inverted File) index builder for memory-efficient search.

use rayon::prelude::*;

use crate::error::{Result, SarissaError};
use crate::vector::Vector;
use crate::vector_index::{VectorIndexBuildConfig, VectorIndexBuilder};

/// Builder for IVF vector indexes (memory-efficient search).
pub struct IvfIndexBuilder {
    config: VectorIndexBuildConfig,
    n_clusters: usize,                       // Number of clusters (cells)
    n_probe: usize,                          // Number of clusters to search
    centroids: Vec<Vector>,                  // Cluster centroids
    inverted_lists: Vec<Vec<(u64, Vector)>>, // Inverted lists for each cluster
    vectors: Vec<(u64, Vector)>,             // All vectors (used during construction)
    is_finalized: bool,
    total_vectors_to_add: Option<usize>,
}

impl IvfIndexBuilder {
    /// Create a new IVF index builder.
    pub fn new(config: VectorIndexBuildConfig) -> Result<Self> {
        let n_clusters = Self::compute_default_clusters(1000); // Default for small datasets

        Ok(Self {
            config,
            n_clusters,
            n_probe: 1, // Default to searching 1 cluster
            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
        })
    }

    /// Set IVF-specific parameters.
    pub fn with_ivf_params(mut self, n_clusters: usize, n_probe: usize) -> Self {
        self.n_clusters = n_clusters;
        self.n_probe = n_probe;
        self
    }

    /// Set the expected total number of vectors (for progress tracking).
    pub fn set_expected_vector_count(&mut self, count: usize) {
        self.total_vectors_to_add = Some(count);
        // Adjust number of clusters based on dataset size
        self.n_clusters = Self::compute_default_clusters(count);
    }

    /// Compute default number of clusters based on dataset size.
    fn compute_default_clusters(n_vectors: usize) -> usize {
        // Rule of thumb: sqrt(n_vectors), with reasonable min/max bounds
        let clusters = (n_vectors as f64).sqrt() as usize;
        clusters.clamp(10, 10000)
    }

    /// Validate vectors before adding them.
    fn validate_vectors(&self, vectors: &[(u64, Vector)]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

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

    /// Train centroids using k-means clustering.
    fn train_centroids(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Err(SarissaError::InvalidOperation(
                "Cannot train centroids on empty vector set".to_string(),
            ));
        }

        if self.vectors.len() < self.n_clusters {
            return Err(SarissaError::InvalidOperation(format!(
                "Cannot create {} clusters from {} vectors",
                self.n_clusters,
                self.vectors.len()
            )));
        }

        println!("Training {} centroids using k-means...", self.n_clusters);

        // Initialize centroids with k-means++
        self.init_centroids_kmeans_plus_plus()?;

        // Run k-means iterations
        let max_iterations = 100;
        let convergence_threshold = 1e-6;

        for iteration in 0..max_iterations {
            let old_centroids = self.centroids.clone();

            // Assign vectors to clusters
            let assignments = self.assign_vectors_to_clusters();

            // Update centroids
            self.update_centroids(&assignments)?;

            // Check for convergence
            let convergence = self.compute_convergence(&old_centroids);
            if convergence < convergence_threshold {
                println!("K-means converged after {} iterations", iteration + 1);
                break;
            }
        }

        Ok(())
    }

    /// Initialize centroids using k-means++ algorithm.
    fn init_centroids_kmeans_plus_plus(&mut self) -> Result<()> {
        use rand::prelude::*;
        let mut rng = rand::rng();

        self.centroids.clear();

        // Choose first centroid randomly
        let first_idx = rng.random_range(0..self.vectors.len());
        self.centroids.push(self.vectors[first_idx].1.clone());

        // Choose remaining centroids with probability proportional to squared distance
        for _ in 1..self.n_clusters {
            let mut distances = Vec::with_capacity(self.vectors.len());
            let mut total_weight = 0.0;

            for (_, vector) in &self.vectors {
                let min_dist = self
                    .centroids
                    .iter()
                    .map(|centroid| {
                        self.config
                            .distance_metric
                            .distance(&vector.data, &centroid.data)
                            .unwrap_or(f32::INFINITY)
                    })
                    .fold(f32::INFINITY, f32::min);

                let weight = min_dist * min_dist;
                distances.push(weight);
                total_weight += weight;
            }

            if total_weight == 0.0 {
                // Fallback to random selection
                let idx = rng.random_range(0..self.vectors.len());
                self.centroids.push(self.vectors[idx].1.clone());
                continue;
            }

            // Select next centroid based on weighted probability
            let target = rng.random::<f32>() * total_weight;
            let mut cumsum = 0.0;

            for (i, &weight) in distances.iter().enumerate() {
                cumsum += weight;
                if cumsum >= target {
                    self.centroids.push(self.vectors[i].1.clone());
                    break;
                }
            }
        }

        Ok(())
    }

    /// Assign each vector to its nearest cluster.
    fn assign_vectors_to_clusters(&self) -> Vec<usize> {
        if self.config.parallel_build && self.vectors.len() > 1000 {
            self.vectors
                .par_iter()
                .map(|(_, vector)| self.find_nearest_centroid(vector))
                .collect()
        } else {
            self.vectors
                .iter()
                .map(|(_, vector)| self.find_nearest_centroid(vector))
                .collect()
        }
    }

    /// Find the index of the nearest centroid for a vector.
    fn find_nearest_centroid(&self, vector: &Vector) -> usize {
        let mut best_cluster = 0;
        let mut best_distance = f32::INFINITY;

        for (i, centroid) in self.centroids.iter().enumerate() {
            if let Ok(distance) = self
                .config
                .distance_metric
                .distance(&vector.data, &centroid.data)
            {
                if distance < best_distance {
                    best_distance = distance;
                    best_cluster = i;
                }
            }
        }

        best_cluster
    }

    /// Update centroids based on cluster assignments.
    fn update_centroids(&mut self, assignments: &[usize]) -> Result<()> {
        let mut cluster_sums = vec![vec![0.0; self.config.dimension]; self.n_clusters];
        let mut cluster_counts = vec![0; self.n_clusters];

        // Sum vectors in each cluster
        for (i, (_, vector)) in self.vectors.iter().enumerate() {
            let cluster = assignments[i];
            cluster_counts[cluster] += 1;

            for (j, &value) in vector.data.iter().enumerate() {
                cluster_sums[cluster][j] += value;
            }
        }

        // Compute new centroids as averages
        for (i, (sum, count)) in cluster_sums.iter().zip(cluster_counts.iter()).enumerate() {
            if *count == 0 {
                // Keep the old centroid if no vectors assigned
                continue;
            }

            let centroid_data: Vec<f32> = sum.iter().map(|&s| s / *count as f32).collect();

            self.centroids[i] = Vector::new(centroid_data);
        }

        Ok(())
    }

    /// Compute convergence metric between old and new centroids.
    fn compute_convergence(&self, old_centroids: &[Vector]) -> f32 {
        if old_centroids.len() != self.centroids.len() {
            return f32::INFINITY;
        }

        let mut total_movement = 0.0;

        for (old, new) in old_centroids.iter().zip(self.centroids.iter()) {
            if let Ok(distance) = self.config.distance_metric.distance(&old.data, &new.data) {
                total_movement += distance;
            }
        }

        total_movement / self.centroids.len() as f32
    }

    /// Build inverted lists by assigning vectors to clusters.
    fn build_inverted_lists(&mut self) -> Result<()> {
        self.inverted_lists = vec![Vec::new(); self.n_clusters];

        for (doc_id, vector) in &self.vectors {
            let cluster = self.find_nearest_centroid(vector);
            self.inverted_lists[cluster].push((*doc_id, vector.clone()));
        }

        // Sort each inverted list by document ID
        if self.config.parallel_build {
            self.inverted_lists.par_iter_mut().for_each(|list| {
                list.sort_by_key(|(doc_id, _)| *doc_id);
            });
        } else {
            for list in &mut self.inverted_lists {
                list.sort_by_key(|(doc_id, _)| *doc_id);
            }
        }

        Ok(())
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

    /// Get the stored vectors (for testing/debugging).
    pub fn vectors(&self) -> &[(u64, Vector)] {
        &self.vectors
    }

    /// Get IVF parameters.
    pub fn ivf_params(&self) -> (usize, usize) {
        (self.n_clusters, self.n_probe)
    }

    /// Get centroids.
    pub fn centroids(&self) -> &[Vector] {
        &self.centroids
    }

    /// Get inverted lists.
    pub fn inverted_lists(&self) -> &[Vec<(u64, Vector)>] {
        &self.inverted_lists
    }
}

impl VectorIndexBuilder for IvfIndexBuilder {
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

        // Adjust cluster count if needed
        if self.vectors.len() < self.n_clusters {
            self.n_clusters = self.vectors.len().max(1);
        }

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

        if self.vectors.is_empty() {
            return Err(SarissaError::InvalidOperation(
                "Cannot finalize empty index".to_string(),
            ));
        }

        // Train centroids using k-means
        self.train_centroids()?;

        // Build inverted lists
        self.build_inverted_lists()?;

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

        // Centroid memory
        let centroid_memory =
            self.centroids.len() * (self.config.dimension * 4 + std::mem::size_of::<Vector>());

        // Inverted list overhead (pointers and metadata)
        let inverted_list_memory =
            self.inverted_lists.len() * (std::mem::size_of::<Vec<(u64, Vector)>>() + 64); // Rough estimate

        let metadata_memory = self.vectors.len() * 64;

        vector_memory + centroid_memory + inverted_list_memory + metadata_memory
    }

    fn optimize(&mut self) -> Result<()> {
        if !self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Index must be finalized before optimization".to_string(),
            ));
        }

        // IVF optimization could include:
        // 1. Rebalancing clusters if they're too uneven
        // 2. Memory compaction
        // 3. Quantization of vectors within clusters

        println!("Optimizing IVF index...");

        // For now, just compact memory
        self.vectors.shrink_to_fit();
        self.centroids.shrink_to_fit();
        for list in &mut self.inverted_lists {
            list.shrink_to_fit();
        }

        Ok(())
    }
}
