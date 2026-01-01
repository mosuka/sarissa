//! IVF (Inverted File) index builder for memory-efficient search.

use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{Result, SarissaError};
use crate::storage::Storage;
use crate::vector::core::vector::Vector;
use crate::vector::index::IvfIndexConfig;
use crate::vector::index::field::LegacyVectorFieldWriter;
use crate::vector::index::io::{read_metadata, write_metadata};
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
/// Builder for IVF vector indexes (memory-efficient search).
pub struct IvfIndexWriter {
    index_config: IvfIndexConfig,
    writer_config: VectorIndexWriterConfig,
    storage: Option<Arc<dyn Storage>>,
    centroids: Vec<Vector>,                          // Cluster centroids
    inverted_lists: Vec<Vec<(u64, String, Vector)>>, // Inverted lists for each cluster
    vectors: Vec<(u64, String, Vector)>,             // All vectors (used during construction)
    is_finalized: bool,
    total_vectors_to_add: Option<usize>,
    next_vec_id: u64,
}

impl IvfIndexWriter {
    /// Create a new IVF index builder.
    ///
    /// # Arguments
    ///
    /// * `config` - Vector index configuration
    /// * `n_clusters` - Number of clusters (cells) to create (typical: sqrt(n_vectors))
    /// * `n_probe` - Number of clusters to search (typical: 1-10, higher = more accurate but slower)
    pub fn new(
        index_config: IvfIndexConfig,
        writer_config: VectorIndexWriterConfig,
    ) -> Result<Self> {
        Ok(Self {
            index_config,
            writer_config,
            storage: None,

            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
        })
    }

    /// Create a new IVF index builder with storage.
    pub fn with_storage(
        index_config: IvfIndexConfig,
        writer_config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
        })
    }

    /// Convert this writer into a doc-centric field writer adapter.
    pub fn into_field_writer(self, field_name: impl Into<String>) -> LegacyVectorFieldWriter<Self> {
        LegacyVectorFieldWriter::new(field_name, self)
    }

    /// Load an existing IVF index from storage.
    pub fn load(
        index_config: IvfIndexConfig,
        writer_config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
        path: &str,
    ) -> Result<Self> {
        use std::io::Read;

        // Open the index file
        let file_name = format!("{}.ivf", path);
        let mut input = storage.open_input(&file_name)?;

        // Read metadata
        let mut num_vectors_buf = [0u8; 4];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u32::from_le_bytes(num_vectors_buf) as usize;

        let mut dimension_buf = [0u8; 4];
        input.read_exact(&mut dimension_buf)?;
        let dimension = u32::from_le_bytes(dimension_buf) as usize;

        let mut n_clusters_buf = [0u8; 4];
        input.read_exact(&mut n_clusters_buf)?;
        let n_clusters = u32::from_le_bytes(n_clusters_buf) as usize;

        let mut n_probe_buf = [0u8; 4];
        input.read_exact(&mut n_probe_buf)?;
        let _n_probe = u32::from_le_bytes(n_probe_buf) as usize;

        if dimension != index_config.dimension {
            return Err(SarissaError::InvalidOperation(format!(
                "Dimension mismatch: expected {}, found {}",
                index_config.dimension, dimension
            )));
        }

        // Read centroids
        let mut centroids = Vec::with_capacity(n_clusters);
        for _ in 0..n_clusters {
            let mut values = vec![0.0f32; dimension];
            for value in &mut values {
                let mut value_buf = [0u8; 4];
                input.read_exact(&mut value_buf)?;
                *value = f32::from_le_bytes(value_buf);
            }
            centroids.push(Vector::new(values));
        }

        // Read inverted lists
        let mut inverted_lists = vec![Vec::new(); n_clusters];
        for list in &mut inverted_lists {
            let mut list_size_buf = [0u8; 4];
            input.read_exact(&mut list_size_buf)?;
            let list_size = u32::from_le_bytes(list_size_buf) as usize;

            for _ in 0..list_size {
                let mut doc_id_buf = [0u8; 8];
                input.read_exact(&mut doc_id_buf)?;
                let doc_id = u64::from_le_bytes(doc_id_buf);

                // Read field name
                let mut field_name_len_buf = [0u8; 4];
                input.read_exact(&mut field_name_len_buf)?;
                let field_name_len = u32::from_le_bytes(field_name_len_buf) as usize;

                let mut field_name_buf = vec![0u8; field_name_len];
                input.read_exact(&mut field_name_buf)?;
                let field_name = String::from_utf8(field_name_buf).map_err(|e| {
                    SarissaError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
                })?;

                // Read metadata and vector data
                let metadata = read_metadata(&mut input)?;
                let mut values = vec![0.0f32; dimension];
                for value in &mut values {
                    let mut value_buf = [0u8; 4];
                    input.read_exact(&mut value_buf)?;
                    *value = f32::from_le_bytes(value_buf);
                }

                list.push((doc_id, field_name, Vector::with_metadata(values, metadata)));
            }
        }

        // Reconstruct vectors from inverted lists
        let mut vectors = Vec::with_capacity(num_vectors);
        for list in &inverted_lists {
            vectors.extend(list.iter().cloned());
        }

        // Calculate next_vec_id from loaded vectors
        let max_id = vectors.iter().map(|(id, _, _)| *id).max().unwrap_or(0);
        let next_vec_id = if num_vectors > 0 { max_id + 1 } else { 0 };

        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),

            centroids,
            inverted_lists,
            vectors,
            is_finalized: true,
            total_vectors_to_add: Some(num_vectors),
            next_vec_id,
        })
    }

    /// Set IVF-specific parameters.
    pub fn with_ivf_params(mut self, n_clusters: usize, n_probe: usize) -> Self {
        self.index_config.n_clusters = n_clusters;
        self.index_config.n_probe = n_probe;
        self
    }

    /// Set the expected total number of vectors (for progress tracking).
    pub fn set_expected_vector_count(&mut self, count: usize) {
        self.total_vectors_to_add = Some(count);
        // Adjust number of clusters based on dataset size
        self.index_config.n_clusters = Self::compute_default_clusters(count);
    }

    /// Compute default number of clusters based on dataset size.
    fn compute_default_clusters(n_vectors: usize) -> usize {
        // Rule of thumb: sqrt(n_vectors), with reasonable min/max bounds
        let clusters = (n_vectors as f64).sqrt() as usize;
        clusters.clamp(10, 10000)
    }

    /// Validate vectors before adding them.
    fn validate_vectors(&self, vectors: &[(u64, String, Vector)]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        for (doc_id, _field_name, vector) in vectors {
            if vector.dimension() != self.index_config.dimension {
                return Err(SarissaError::InvalidOperation(format!(
                    "Vector {} has dimension {}, expected {}",
                    doc_id,
                    vector.dimension(),
                    self.index_config.dimension
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
    fn normalize_vectors(&self, vectors: &mut [(u64, String, Vector)]) {
        if !self.index_config.normalize_vectors {
            return;
        }

        if self.writer_config.parallel_build && vectors.len() > 100 {
            vectors.par_iter_mut().for_each(|(_, _, vector)| {
                vector.normalize();
            });
        } else {
            for (_, _, vector) in vectors {
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

        if self.vectors.len() < self.index_config.n_clusters {
            return Err(SarissaError::InvalidOperation(format!(
                "Cannot create {} clusters from {} vectors",
                self.index_config.n_clusters,
                self.vectors.len() as u64
            )));
        }

        println!(
            "Training {} centroids using k-means...",
            self.index_config.n_clusters
        );

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
        self.centroids.push(self.vectors[first_idx].2.clone());

        // Choose remaining centroids with probability proportional to squared distance
        for _ in 1..self.index_config.n_clusters {
            let mut distances = Vec::with_capacity(self.vectors.len());
            let mut total_weight = 0.0;

            for (_, _, vector) in &self.vectors {
                let min_dist = self
                    .centroids
                    .iter()
                    .map(|centroid| {
                        self.index_config
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
                self.centroids.push(self.vectors[idx].2.clone());
                continue;
            }

            // Select next centroid based on weighted probability
            let target = rng.random::<f32>() * total_weight;
            let mut cumsum = 0.0;

            for (i, &weight) in distances.iter().enumerate() {
                cumsum += weight;
                if cumsum >= target {
                    self.centroids.push(self.vectors[i].2.clone());
                    break;
                }
            }
        }

        Ok(())
    }

    /// Assign each vector to its nearest cluster.
    fn assign_vectors_to_clusters(&self) -> Vec<usize> {
        if self.writer_config.parallel_build && self.vectors.len() as u64 > 1000 {
            self.vectors
                .par_iter()
                .map(|(_, _, vector)| self.find_nearest_centroid(vector))
                .collect()
        } else {
            self.vectors
                .iter()
                .map(|(_, _, vector)| self.find_nearest_centroid(vector))
                .collect()
        }
    }

    /// Find the index of the nearest centroid for a vector.
    fn find_nearest_centroid(&self, vector: &Vector) -> usize {
        let mut best_cluster = 0;
        let mut best_distance = f32::INFINITY;

        for (i, centroid) in self.centroids.iter().enumerate() {
            if let Ok(distance) = self
                .index_config
                .distance_metric
                .distance(&vector.data, &centroid.data)
                && distance < best_distance
            {
                best_distance = distance;
                best_cluster = i;
            }
        }

        best_cluster
    }

    /// Update centroids based on cluster assignments.
    fn update_centroids(&mut self, assignments: &[usize]) -> Result<()> {
        let mut cluster_sums =
            vec![vec![0.0; self.index_config.dimension]; self.index_config.n_clusters];
        let mut cluster_counts = vec![0; self.index_config.n_clusters];

        // Sum vectors in each cluster
        for (i, (_, _, vector)) in self.vectors.iter().enumerate() {
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
            if let Ok(distance) = self
                .index_config
                .distance_metric
                .distance(&old.data, &new.data)
            {
                total_movement += distance;
            }
        }

        total_movement / self.centroids.len() as f32
    }

    /// Build inverted lists by assigning vectors to clusters.
    fn build_inverted_lists(&mut self) -> Result<()> {
        self.inverted_lists = vec![Vec::new(); self.index_config.n_clusters];

        for (doc_id, field_name, vector) in &self.vectors {
            let cluster = self.find_nearest_centroid(vector);
            self.inverted_lists[cluster].push((*doc_id, field_name.clone(), vector.clone()));
        }

        // Sort each inverted list by document ID
        if self.writer_config.parallel_build {
            self.inverted_lists.par_iter_mut().for_each(|list| {
                list.sort_by_key(|(doc_id, _, _)| *doc_id);
            });
        } else {
            for list in &mut self.inverted_lists {
                list.sort_by_key(|(doc_id, _, _)| *doc_id);
            }
        }

        Ok(())
    }

    /// Check for memory limits.
    fn check_memory_limit(&self) -> Result<()> {
        if let Some(limit) = self.writer_config.memory_limit {
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
    pub fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    /// Get IVF parameters.
    pub fn ivf_params(&self) -> (usize, usize) {
        (self.index_config.n_clusters, self.index_config.n_probe)
    }

    /// Get centroids.
    pub fn centroids(&self) -> &[Vector] {
        &self.centroids
    }

    /// Get inverted lists.
    pub fn inverted_lists(&self) -> &[Vec<(u64, String, Vector)>] {
        &self.inverted_lists
    }
}

/// Statistics for a single IVF cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    pub cluster_id: usize,
    pub count: usize,
    pub avg_distance: f32,
}

impl IvfIndexWriter {
    /// Get statistics for each cluster in the index.
    pub fn get_cluster_stats(&self) -> Vec<ClusterStats> {
        if !self.is_finalized || self.centroids.is_empty() {
            return Vec::new();
        }

        let mut stats = Vec::with_capacity(self.centroids.len());

        for (i, (centroid, list)) in self
            .centroids
            .iter()
            .zip(self.inverted_lists.iter())
            .enumerate()
        {
            let count = list.len();
            let total_dist: f32 = list
                .iter()
                .map(|(_, _, vec)| {
                    self.index_config
                        .distance_metric
                        .distance(&vec.data, &centroid.data)
                        .unwrap_or(0.0)
                })
                .sum();

            stats.push(ClusterStats {
                cluster_id: i,
                count,
                avg_distance: if count > 0 {
                    total_dist / count as f32
                } else {
                    0.0
                },
            });
        }

        stats
    }

    /// Merge sparse clusters into their nearest neighbors.
    pub fn merge_sparse_clusters(&mut self, threshold: usize) -> Result<usize> {
        if !self.is_finalized || self.centroids.is_empty() {
            return Ok(0);
        }

        let stats = self.get_cluster_stats();
        let sparse_indices: Vec<usize> = stats
            .iter()
            .filter(|s| s.count < threshold)
            .map(|s| s.cluster_id)
            .collect();

        if sparse_indices.is_empty() || sparse_indices.len() == self.centroids.len() {
            return Ok(0);
        }

        let non_sparse_indices: Vec<usize> = stats
            .iter()
            .filter(|s| s.count >= threshold)
            .map(|s| s.cluster_id)
            .collect();

        let merged_count = sparse_indices.len();
        let mut moves = Vec::new();

        for &sparse_idx in &sparse_indices {
            let sparse_centroid = &self.centroids[sparse_idx];
            let mut best_target = non_sparse_indices[0];
            let mut best_dist = f32::INFINITY;

            for &target_idx in &non_sparse_indices {
                if let Ok(dist) = self
                    .index_config
                    .distance_metric
                    .distance(&sparse_centroid.data, &self.centroids[target_idx].data)
                {
                    if dist < best_dist {
                        best_dist = dist;
                        best_target = target_idx;
                    }
                }
            }
            moves.push((sparse_idx, best_target));
        }

        // Apply moves
        for (sparse_idx, target_idx) in moves {
            let mut vectors_to_move = std::mem::take(&mut self.inverted_lists[sparse_idx]);
            self.inverted_lists[target_idx].append(&mut vectors_to_move);
        }

        // Rebuild centroids and inverted lists
        let mut new_centroids = Vec::new();
        let mut new_inverted_lists = Vec::new();

        for i in 0..self.centroids.len() {
            if !sparse_indices.contains(&i) {
                new_centroids.push(self.centroids[i].clone());
                new_inverted_lists.push(std::mem::take(&mut self.inverted_lists[i]));
            }
        }

        self.centroids = new_centroids;
        self.inverted_lists = new_inverted_lists;
        self.index_config.n_clusters = self.centroids.len();

        // Update centroids for the merged clusters (re-average)
        for (i, list) in self.inverted_lists.iter().enumerate() {
            if !list.is_empty() {
                let dim = self.index_config.dimension;
                let mut sum = vec![0.0; dim];
                for (_, _, vec) in list {
                    for (j, &val) in vec.data.iter().enumerate() {
                        sum[j] += val;
                    }
                }
                let new_data: Vec<f32> = sum.iter().map(|&s| s / list.len() as f32).collect();
                self.centroids[i] = Vector::new(new_data);
            }
        }

        Ok(merged_count)
    }

    /// Split dense clusters into multiple clusters using K-means (k=2).
    pub fn split_dense_clusters(&mut self, threshold: usize) -> Result<usize> {
        if !self.is_finalized || self.centroids.is_empty() {
            return Ok(0);
        }

        let stats = self.get_cluster_stats();
        let dense_indices: Vec<usize> = stats
            .iter()
            .filter(|s| s.count > threshold)
            .map(|s| s.cluster_id)
            .collect();

        if dense_indices.is_empty() {
            return Ok(0);
        }

        let mut additional_clusters = 0;
        let mut new_centroids = Vec::new();
        let mut new_inverted_lists = Vec::new();

        for i in 0..self.centroids.len() {
            if dense_indices.contains(&i) {
                let list = std::mem::take(&mut self.inverted_lists[i]);
                if list.len() < 2 {
                    // Cannot split
                    new_centroids.push(self.centroids[i].clone());
                    new_inverted_lists.push(list);
                    continue;
                }

                // Perform k=2 split
                let (c1, l1, c2, l2) = self.split_cluster_kmeans_k2(list)?;
                new_centroids.push(c1);
                new_inverted_lists.push(l1);
                new_centroids.push(c2);
                new_inverted_lists.push(l2);
                additional_clusters += 1;
            } else {
                new_centroids.push(self.centroids[i].clone());
                new_inverted_lists.push(std::mem::take(&mut self.inverted_lists[i]));
            }
        }

        self.centroids = new_centroids;
        self.inverted_lists = new_inverted_lists;
        self.index_config.n_clusters = self.centroids.len();

        Ok(additional_clusters)
    }

    /// Split a cluster into two using K-means.
    fn split_cluster_kmeans_k2(
        &self,
        vectors: Vec<(u64, String, Vector)>,
    ) -> Result<(
        Vector,
        Vec<(u64, String, Vector)>,
        Vector,
        Vec<(u64, String, Vector)>,
    )> {
        use rand::prelude::*;
        let mut rng = rand::rng();

        // Pick two initial centroids
        let idx1 = rng.random_range(0..vectors.len());
        let mut idx2 = rng.random_range(0..vectors.len());
        while idx1 == idx2 && vectors.len() > 1 {
            idx2 = rng.random_range(0..vectors.len());
        }

        let mut c1 = vectors[idx1].2.clone();
        let mut c2 = vectors[idx2].2.clone();

        let mut l1 = Vec::new();
        let mut l2 = Vec::new();

        // Simple 10 iterations of K-means
        for _ in 0..10 {
            l1.clear();
            l2.clear();

            for (_, _, vec) in &vectors {
                let d1 = self
                    .index_config
                    .distance_metric
                    .distance(&vec.data, &c1.data)
                    .unwrap_or(f32::INFINITY);
                let d2 = self
                    .index_config
                    .distance_metric
                    .distance(&vec.data, &c2.data)
                    .unwrap_or(f32::INFINITY);

                if d1 < d2 {
                    l1.push((0, String::new(), vec.clone())); // We'll restore the actual IDs later
                } else {
                    l2.push((0, String::new(), vec.clone()));
                }
            }

            // Update centroids
            if !l1.is_empty() {
                c1 = self.calculate_mean_vector(&l1);
            }
            if !l2.is_empty() {
                c2 = self.calculate_mean_vector(&l2);
            }
        }

        // Final assignment with original vectors to preserve IDs
        l1.clear();
        l2.clear();
        for item in vectors {
            let d1 = self
                .index_config
                .distance_metric
                .distance(&item.2.data, &c1.data)
                .unwrap_or(f32::INFINITY);
            let d2 = self
                .index_config
                .distance_metric
                .distance(&item.2.data, &c2.data)
                .unwrap_or(f32::INFINITY);

            if d1 < d2 {
                l1.push(item);
            } else {
                l2.push(item);
            }
        }

        Ok((c1, l1, c2, l2))
    }

    /// Calculate the mean vector for a list of vectors.
    fn calculate_mean_vector(&self, list: &[(u64, String, Vector)]) -> Vector {
        let dim = self.index_config.dimension;
        let mut sum = vec![0.0; dim];
        for (_, _, vec) in list {
            for (j, &val) in vec.data.iter().enumerate() {
                sum[j] += val;
            }
        }
        let data: Vec<f32> = sum.iter().map(|&s| s / list.len() as f32).collect();
        Vector::new(data)
    }
}

impl VectorIndexWriter for IvfIndexWriter {
    fn next_vector_id(&self) -> u64 {
        self.next_vec_id
    }

    fn build(&mut self, mut vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Cannot build on finalized index".to_string(),
            ));
        }

        self.validate_vectors(&vectors)?;
        self.normalize_vectors(&mut vectors);

        // Update next_vec_id
        if let Some(max_id) = vectors.iter().map(|(id, _, _)| *id).max()
            && max_id >= self.next_vec_id
        {
            self.next_vec_id = max_id + 1;
        }

        self.vectors = vectors;
        self.total_vectors_to_add = Some(self.vectors.len());

        // Adjust cluster count if needed
        if self.vectors.len() < self.index_config.n_clusters {
            self.index_config.n_clusters = self.vectors.len().max(1);
        }

        self.check_memory_limit()?;
        Ok(())
    }

    fn add_vectors(&mut self, mut vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Cannot add vectors to finalized index".to_string(),
            ));
        }

        self.validate_vectors(&vectors)?;
        self.normalize_vectors(&mut vectors);

        // Update next_vec_id
        if let Some(max_id) = vectors.iter().map(|(id, _, _)| *id).max()
            && max_id >= self.next_vec_id
        {
            self.next_vec_id = max_id + 1;
        }

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
                let current = self.vectors.len() as u64 as f32;
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
            self.index_config.dimension * 4 + // f32 values
            std::mem::size_of::<Vector>()
                // Vector struct overhead
            );

        // Centroid memory
        let centroid_memory = self.centroids.len()
            * (self.index_config.dimension * 4 + std::mem::size_of::<Vector>());

        // Inverted list overhead (pointers and metadata)
        let inverted_list_memory =
            self.inverted_lists.len() * (std::mem::size_of::<Vec<(u64, String, Vector)>>() + 64); // Rough estimate

        let metadata_memory = self.vectors.len() * 64;

        vector_memory + centroid_memory + inverted_list_memory + metadata_memory
    }

    fn optimize(&mut self) -> Result<()> {
        if !self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Index must be finalized before optimization".to_string(),
            ));
        }

        println!("Optimizing IVF index...");

        // Rebalance clusters if they're too uneven
        let total_vectors = self.vectors.len();
        let avg_vectors_per_cluster = total_vectors / self.index_config.n_clusters.max(1);
        let sparse_threshold = avg_vectors_per_cluster / 4;
        let dense_threshold = avg_vectors_per_cluster * 4;

        let merged = self.merge_sparse_clusters(sparse_threshold.max(2))?;
        if merged > 0 {
            println!("Merged {} sparse clusters", merged);
        }

        let split = self.split_dense_clusters(dense_threshold)?;
        if split > 0 {
            println!("Split {} dense clusters", split);
        }

        // For now, just compact memory
        self.vectors.shrink_to_fit();
        self.centroids.shrink_to_fit();
        for list in &mut self.inverted_lists {
            list.shrink_to_fit();
        }

        Ok(())
    }

    fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    fn write(&self, path: &str) -> Result<()> {
        use std::io::Write;

        if !self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Index must be finalized before writing".to_string(),
            ));
        }

        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| SarissaError::InvalidOperation("No storage configured".to_string()))?;

        // Create the index file
        let file_name = format!("{}.ivf", path);
        let mut output = storage.create_output(&file_name)?;

        // Write metadata
        output.write_all(&(self.vectors.len() as u64 as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.dimension as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.n_clusters as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.n_probe as u32).to_le_bytes())?;

        // Write centroids
        for centroid in &self.centroids {
            for value in &centroid.data {
                output.write_all(&value.to_le_bytes())?;
            }
        }

        // Write inverted lists
        for list in &self.inverted_lists {
            output.write_all(&(list.len() as u32).to_le_bytes())?;
            for (doc_id, field_name, vector) in list {
                output.write_all(&doc_id.to_le_bytes())?;

                // Write field name length and field name
                let field_name_bytes = field_name.as_bytes();
                output.write_all(&(field_name_bytes.len() as u32).to_le_bytes())?;
                output.write_all(field_name_bytes)?;

                write_metadata(&mut output, &vector.metadata)?;

                // Write vector data
                for value in &vector.data {
                    output.write_all(&value.to_le_bytes())?;
                }
            }
        }

        output.flush()?;
        Ok(())
    }

    fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    fn delete_documents(&mut self, field: &str, value: &str) -> Result<u64> {
        // Simplified implementation - returns 0
        // TODO: Implement proper deletion with metadata storage
        let _field = field;
        let _value = value;
        Ok(0)
    }

    fn rollback(&mut self) -> Result<()> {
        self.vectors.clear();
        self.is_finalized = false;
        self.next_vec_id = 0;
        Ok(())
    }

    fn pending_docs(&self) -> u64 {
        if self.is_finalized {
            0
        } else {
            self.vectors.len() as u64
        }
    }

    fn close(&mut self) -> Result<()> {
        self.vectors.clear();
        self.is_finalized = true;
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.is_finalized && self.vectors.is_empty()
    }
}
