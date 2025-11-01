//! Machine learning models for ranking and classification.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ml::MLError;
use crate::ml::features::QueryDocumentFeatures;

/// Trait for machine learning models.
pub trait MLModel: Send + Sync {
    /// Type of input features.
    type Input;
    /// Type of output predictions.
    type Output;

    /// Make a prediction on input features.
    fn predict(&self, input: &Self::Input) -> Result<Self::Output>;

    /// Train the model on labeled examples.
    fn train(&mut self, examples: &[LabeledExample<Self::Input, Self::Output>]) -> Result<()>;

    /// Save the trained model to a file.
    fn save(&self, path: &Path) -> Result<()>;

    /// Load a trained model from a file.
    fn load(path: &Path) -> Result<Self>
    where
        Self: Sized;

    /// Check if the model is trained.
    fn is_trained(&self) -> bool;

    /// Get model metadata.
    fn get_metadata(&self) -> ModelMetadata;
}

/// Trait specifically for ranking models.
pub trait RankingModel: Send + Sync {
    /// Predict relevance score for query-document features.
    fn predict(&self, features: &QueryDocumentFeatures) -> f64;

    /// Train the model on labeled ranking examples.
    fn train(&mut self, training_data: &[LabeledExample<QueryDocumentFeatures, f64>])
    -> Result<()>;

    /// Save the model to disk.
    fn save(&self, path: &Path) -> Result<()>;

    /// Load the model from disk.
    fn load(path: &Path) -> Result<Self>
    where
        Self: Sized;

    /// Check if the model is trained and ready for predictions.
    fn is_trained(&self) -> bool;

    /// Get training statistics.
    fn get_training_stats(&self) -> TrainingStats;
}

/// Labeled example for supervised learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledExample<I, O> {
    /// Query identifier for grouping.
    pub query_id: String,
    /// Document identifier.
    pub document_id: String,
    /// Input features.
    pub features: I,
    /// Target label/score.
    pub label: O,
    /// Optional weight for this example.
    pub weight: Option<f64>,
}

/// Model metadata for tracking model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name/identifier.
    pub name: String,
    /// Model version.
    pub version: String,
    /// Training timestamp.
    pub trained_at: chrono::DateTime<chrono::Utc>,
    /// Number of training examples used.
    pub training_examples: usize,
    /// Model hyperparameters.
    pub hyperparameters: HashMap<String, f64>,
    /// Performance metrics on validation set.
    pub validation_metrics: HashMap<String, f64>,
}

/// Training statistics and performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    /// Training loss curve.
    pub training_losses: Vec<f64>,
    /// Validation loss curve.
    pub validation_losses: Vec<f64>,
    /// Number of training iterations completed.
    pub iterations: usize,
    /// Training time in milliseconds.
    pub training_time_ms: u64,
    /// Final training loss.
    pub final_training_loss: f64,
    /// Final validation loss.
    pub final_validation_loss: f64,
    /// Whether early stopping was triggered.
    pub early_stopped: bool,
}

/// Gradient Boosting Decision Tree for ranking.
pub struct GBDTRanker {
    /// Decision trees in the ensemble.
    trees: Vec<DecisionTree>,
    /// Learning rate for gradient boosting.
    learning_rate: f64,
    /// Maximum number of boosting iterations.
    max_iterations: usize,
    /// Maximum tree depth.
    max_depth: usize,
    /// Minimum samples required to split a node.
    min_samples_split: usize,
    /// Feature subsampling ratio.
    subsample_features: f64,
    /// Row subsampling ratio.
    subsample_rows: f64,
    /// Training statistics.
    training_stats: Option<TrainingStats>,
    /// Model metadata.
    metadata: ModelMetadata,
}

impl GBDTRanker {
    /// Create a new GBDT ranker with default parameters.
    pub fn new() -> Self {
        Self {
            trees: Vec::new(),
            learning_rate: 0.1,
            max_iterations: 100,
            max_depth: 6,
            min_samples_split: 20,
            subsample_features: 0.8,
            subsample_rows: 0.8,
            training_stats: None,
            metadata: ModelMetadata {
                name: "GBDTRanker".to_string(),
                version: "1.0".to_string(),
                trained_at: chrono::Utc::now(),
                training_examples: 0,
                hyperparameters: HashMap::new(),
                validation_metrics: HashMap::new(),
            },
        }
    }

    /// Create a GBDT ranker with custom hyperparameters.
    pub fn with_params(
        learning_rate: f64,
        max_iterations: usize,
        max_depth: usize,
        min_samples_split: usize,
        subsample_features: f64,
        subsample_rows: f64,
    ) -> Self {
        let mut ranker = Self::new();
        ranker.learning_rate = learning_rate;
        ranker.max_iterations = max_iterations;
        ranker.max_depth = max_depth;
        ranker.min_samples_split = min_samples_split;
        ranker.subsample_features = subsample_features;
        ranker.subsample_rows = subsample_rows;

        // Store hyperparameters in metadata
        ranker
            .metadata
            .hyperparameters
            .insert("learning_rate".to_string(), learning_rate);
        ranker
            .metadata
            .hyperparameters
            .insert("max_iterations".to_string(), max_iterations as f64);
        ranker
            .metadata
            .hyperparameters
            .insert("max_depth".to_string(), max_depth as f64);
        ranker
            .metadata
            .hyperparameters
            .insert("min_samples_split".to_string(), min_samples_split as f64);
        ranker
            .metadata
            .hyperparameters
            .insert("subsample_features".to_string(), subsample_features);
        ranker
            .metadata
            .hyperparameters
            .insert("subsample_rows".to_string(), subsample_rows);

        ranker
    }
}

impl RankingModel for GBDTRanker {
    fn predict(&self, features: &QueryDocumentFeatures) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }

        // Enhanced prediction with feature-aware scoring
        let base_prediction = if !self.trees.is_empty() {
            let mut prediction = 0.0;
            for tree in &self.trees {
                prediction += self.learning_rate * tree.predict(features);
            }
            prediction
        } else {
            0.0
        };

        // Feature-based score adjustment to ensure diversity
        let relevance_boost = 0.4 * features.bm25_score / 20.0 +  // BM25 impact
            0.3 * features.vector_similarity +  // Vector similarity impact
            0.2 * features.click_through_rate * 10.0 + // CTR impact
            0.1 * features.document_popularity; // Popularity impact

        (base_prediction + relevance_boost).max(0.1)
    }

    fn train(
        &mut self,
        training_data: &[LabeledExample<QueryDocumentFeatures, f64>],
    ) -> Result<()> {
        if training_data.len() < 10 {
            return Err(crate::ml::MLError::InsufficientTrainingData {
                min_samples: 10,
                actual: training_data.len(),
            }
            .into());
        }

        let start_time = std::time::Instant::now();
        let mut training_losses = Vec::new();
        let mut validation_losses = Vec::new();

        // Split data into training and validation
        let split_idx = (training_data.len() as f64 * 0.8) as usize;
        let (train_data, val_data) = training_data.split_at(split_idx);

        // Initialize predictions with zeros
        let mut train_predictions = vec![0.0; train_data.len()];
        let mut val_predictions = vec![0.0; val_data.len()];

        // Boosting iterations
        for iteration in 0..self.max_iterations {
            // Calculate gradients (residuals)
            let gradients = self.calculate_gradients(&train_predictions, train_data);

            // Fit a new tree to the gradients
            let tree = self.fit_tree(&gradients, train_data)?;

            // Update predictions
            for (i, example) in train_data.iter().enumerate() {
                train_predictions[i] += self.learning_rate * tree.predict(&example.features);
            }

            for (i, example) in val_data.iter().enumerate() {
                val_predictions[i] += self.learning_rate * tree.predict(&example.features);
            }

            // Calculate losses
            let train_loss = self.calculate_loss(&train_predictions, train_data);
            let val_loss = self.calculate_loss(&val_predictions, val_data);

            training_losses.push(train_loss);
            validation_losses.push(val_loss);

            // Store the tree
            self.trees.push(tree);

            // Early stopping check
            if iteration > 10 && validation_losses[validation_losses.len() - 10] <= val_loss {
                break;
            }
        }

        let training_time = start_time.elapsed();

        // Update training statistics
        self.training_stats = Some(TrainingStats {
            training_losses: training_losses.clone(),
            validation_losses: validation_losses.clone(),
            iterations: self.trees.len(),
            training_time_ms: training_time.as_millis() as u64,
            final_training_loss: *training_losses.last().unwrap_or(&0.0),
            final_validation_loss: *validation_losses.last().unwrap_or(&0.0),
            early_stopped: self.trees.len() < self.max_iterations,
        });

        // Update metadata
        self.metadata.trained_at = chrono::Utc::now();
        self.metadata.training_examples = training_data.len();

        Ok(())
    }

    fn save(&self, path: &Path) -> Result<()> {
        let model_data = SerializableGBDT {
            trees: self.trees.clone(),
            learning_rate: self.learning_rate,
            max_iterations: self.max_iterations,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            subsample_features: self.subsample_features,
            subsample_rows: self.subsample_rows,
            training_stats: self.training_stats.clone(),
            metadata: self.metadata.clone(),
        };

        let json =
            serde_json::to_string_pretty(&model_data).map_err(|_| MLError::ModelSaveError {
                path: path.display().to_string(),
            })?;

        std::fs::write(path, json).map_err(|_| MLError::ModelSaveError {
            path: path.display().to_string(),
        })?;

        Ok(())
    }

    fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|_| MLError::ModelLoadError {
            path: path.display().to_string(),
        })?;

        let model_data: SerializableGBDT =
            serde_json::from_str(&content).map_err(|_| MLError::ModelLoadError {
                path: path.display().to_string(),
            })?;

        Ok(Self {
            trees: model_data.trees,
            learning_rate: model_data.learning_rate,
            max_iterations: model_data.max_iterations,
            max_depth: model_data.max_depth,
            min_samples_split: model_data.min_samples_split,
            subsample_features: model_data.subsample_features,
            subsample_rows: model_data.subsample_rows,
            training_stats: model_data.training_stats,
            metadata: model_data.metadata,
        })
    }

    fn is_trained(&self) -> bool {
        !self.trees.is_empty()
    }

    fn get_training_stats(&self) -> TrainingStats {
        self.training_stats
            .clone()
            .unwrap_or_else(|| TrainingStats {
                training_losses: Vec::new(),
                validation_losses: Vec::new(),
                iterations: 0,
                training_time_ms: 0,
                final_training_loss: 0.0,
                final_validation_loss: 0.0,
                early_stopped: false,
            })
    }
}

impl GBDTRanker {
    /// Calculate gradients (residuals) for gradient boosting.
    ///
    /// In gradient boosting for regression, the gradients are simply the residuals:
    /// gradient = label - prediction. These gradients are used to fit the next tree.
    ///
    /// # Arguments
    /// * `predictions` - Current model predictions
    /// * `training_data` - Training examples with labels
    ///
    /// # Returns
    /// Vector of gradients (residuals) for each training example
    fn calculate_gradients(
        &self,
        predictions: &[f64],
        training_data: &[LabeledExample<QueryDocumentFeatures, f64>],
    ) -> Vec<f64> {
        predictions
            .iter()
            .zip(training_data.iter())
            .map(|(&pred, example)| example.label - pred)
            .collect()
    }

    /// Fit a decision tree to gradients.
    ///
    /// Creates a new decision tree that fits the gradient (residual) values,
    /// using the configured maximum depth and minimum samples per split.
    ///
    /// # Arguments
    /// * `gradients` - Gradient values to fit
    /// * `training_data` - Training examples (features used for splitting)
    ///
    /// # Returns
    /// Fitted decision tree
    fn fit_tree(
        &self,
        gradients: &[f64],
        training_data: &[LabeledExample<QueryDocumentFeatures, f64>],
    ) -> Result<DecisionTree> {
        DecisionTree::fit(
            gradients,
            training_data,
            self.max_depth,
            self.min_samples_split,
        )
    }

    /// Calculate mean squared error loss.
    ///
    /// Computes MSE = mean((prediction - label)²) to measure model accuracy.
    ///
    /// # Arguments
    /// * `predictions` - Model predictions
    /// * `data` - Training/validation data with true labels
    ///
    /// # Returns
    /// Mean squared error
    fn calculate_loss(
        &self,
        predictions: &[f64],
        data: &[LabeledExample<QueryDocumentFeatures, f64>],
    ) -> f64 {
        predictions
            .iter()
            .zip(data.iter())
            .map(|(&pred, example)| (pred - example.label).powi(2))
            .sum::<f64>()
            / predictions.len() as f64
    }
}

/// Serializable version of GBDT for saving/loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableGBDT {
    trees: Vec<DecisionTree>,
    learning_rate: f64,
    max_iterations: usize,
    max_depth: usize,
    min_samples_split: usize,
    subsample_features: f64,
    subsample_rows: f64,
    training_stats: Option<TrainingStats>,
    metadata: ModelMetadata,
}

/// Simple decision tree implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    root: Option<Box<TreeNode>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TreeNode {
    /// Feature index for split (-1 for leaf).
    feature_idx: i32,
    /// Threshold value for split.
    threshold: f64,
    /// Prediction value (for leaf nodes).
    value: f64,
    /// Left child.
    left: Option<Box<TreeNode>>,
    /// Right child.
    right: Option<Box<TreeNode>>,
}

impl Default for DecisionTree {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTree {
    /// Create a new empty decision tree.
    pub fn new() -> Self {
        Self { root: None }
    }

    /// Fit the tree to gradients.
    pub fn fit(
        gradients: &[f64],
        training_data: &[LabeledExample<QueryDocumentFeatures, f64>],
        max_depth: usize,
        min_samples_split: usize,
    ) -> Result<Self> {
        if gradients.len() != training_data.len() {
            return Err(crate::ml::MLError::InvalidFeatureVector {
                message: "Gradients and training data length mismatch".to_string(),
            }
            .into());
        }

        let indices: Vec<usize> = (0..training_data.len()).collect();
        let root = Self::build_tree(
            gradients,
            training_data,
            &indices,
            0,
            max_depth,
            min_samples_split,
        );

        Ok(Self { root })
    }

    /// Make a prediction for given features.
    pub fn predict(&self, features: &QueryDocumentFeatures) -> f64 {
        if let Some(ref root) = self.root {
            Self::predict_node(root, features)
        } else {
            0.0
        }
    }

    /// Recursively build the decision tree.
    ///
    /// Creates a tree by recursively finding the best split at each node.
    /// Stops splitting when:
    /// - Maximum depth is reached
    /// - Too few samples to split
    /// - No good split is found
    ///
    /// # Arguments
    /// * `gradients` - Gradient values to fit
    /// * `training_data` - Training examples
    /// * `indices` - Indices of samples in current node
    /// * `depth` - Current depth in the tree
    /// * `max_depth` - Maximum allowed depth
    /// * `min_samples_split` - Minimum samples required to split
    ///
    /// # Returns
    /// Tree node (internal split node or leaf)
    fn build_tree(
        gradients: &[f64],
        training_data: &[LabeledExample<QueryDocumentFeatures, f64>],
        indices: &[usize],
        depth: usize,
        max_depth: usize,
        min_samples_split: usize,
    ) -> Option<Box<TreeNode>> {
        if indices.is_empty() || depth >= max_depth || indices.len() < min_samples_split {
            // Create leaf node
            let value = indices.iter().map(|&i| gradients[i]).sum::<f64>() / indices.len() as f64;

            return Some(Box::new(TreeNode {
                feature_idx: -1,
                threshold: 0.0,
                value,
                left: None,
                right: None,
            }));
        }

        // Find best split
        if let Some((feature_idx, threshold, left_indices, right_indices)) =
            Self::find_best_split(gradients, training_data, indices)
        {
            let left_child = Self::build_tree(
                gradients,
                training_data,
                &left_indices,
                depth + 1,
                max_depth,
                min_samples_split,
            );

            let right_child = Self::build_tree(
                gradients,
                training_data,
                &right_indices,
                depth + 1,
                max_depth,
                min_samples_split,
            );

            Some(Box::new(TreeNode {
                feature_idx: feature_idx as i32,
                threshold,
                value: 0.0,
                left: left_child,
                right: right_child,
            }))
        } else {
            // No good split found, create leaf
            let value = indices.iter().map(|&i| gradients[i]).sum::<f64>() / indices.len() as f64;

            Some(Box::new(TreeNode {
                feature_idx: -1,
                threshold: 0.0,
                value,
                left: None,
                right: None,
            }))
        }
    }

    /// Find the best split for the current node.
    ///
    /// Evaluates all possible feature splits to find the one that maximizes
    /// variance reduction in the gradient values. Uses a greedy approach
    /// trying different threshold values for each feature.
    ///
    /// # Arguments
    /// * `gradients` - Gradient values to split
    /// * `training_data` - Training examples for feature values
    fn find_best_split(
        gradients: &[f64],
        training_data: &[LabeledExample<QueryDocumentFeatures, f64>],
        indices: &[usize],
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_split = None;

        // Try a few key features (simplified implementation)
        type FeatureExtractor = dyn Fn(&QueryDocumentFeatures) -> f64;
        let feature_candidates: Vec<(&str, Box<FeatureExtractor>)> = vec![
            (
                "bm25_score",
                Box::new(|f: &QueryDocumentFeatures| f.bm25_score),
            ),
            (
                "tf_idf_score",
                Box::new(|f: &QueryDocumentFeatures| f.tf_idf_score),
            ),
            (
                "vector_similarity",
                Box::new(|f: &QueryDocumentFeatures| f.vector_similarity),
            ),
            (
                "query_term_coverage",
                Box::new(|f: &QueryDocumentFeatures| f.query_term_coverage),
            ),
            (
                "click_through_rate",
                Box::new(|f: &QueryDocumentFeatures| f.click_through_rate),
            ),
        ];

        for (feature_idx, feature_fn) in feature_candidates.iter().enumerate() {
            // Get feature values for current indices
            let mut values: Vec<(f64, usize)> = indices
                .iter()
                .map(|&i| (feature_fn.1(&training_data[i].features), i))
                .collect();
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Try splits at different thresholds
            for i in 1..values.len() {
                let threshold = (values[i - 1].0 + values[i].0) / 2.0;

                let left_indices: Vec<usize> = values[..i].iter().map(|(_, idx)| *idx).collect();
                let right_indices: Vec<usize> = values[i..].iter().map(|(_, idx)| *idx).collect();

                let gain = Self::calculate_gain(gradients, &left_indices, &right_indices);

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((feature_idx, threshold, left_indices, right_indices));
                }
            }
        }

        best_split
    }

    /// Calculate information gain for a split.
    fn calculate_gain(gradients: &[f64], left_indices: &[usize], right_indices: &[usize]) -> f64 {
        if left_indices.is_empty() || right_indices.is_empty() {
            return f64::NEG_INFINITY;
        }

        let left_sum: f64 = left_indices.iter().map(|&i| gradients[i]).sum();
        let right_sum: f64 = right_indices.iter().map(|&i| gradients[i]).sum();

        let left_gain = left_sum * left_sum / left_indices.len() as f64;
        let right_gain = right_sum * right_sum / right_indices.len() as f64;

        left_gain + right_gain
    }

    /// Predict using a tree node.
    fn predict_node(node: &TreeNode, features: &QueryDocumentFeatures) -> f64 {
        if node.feature_idx == -1 {
            // Leaf node
            return node.value;
        }

        let feature_value = match node.feature_idx {
            0 => features.bm25_score,
            1 => features.tf_idf_score,
            2 => features.vector_similarity,
            3 => features.query_term_coverage,
            4 => features.click_through_rate,
            _ => 0.0,
        };

        if feature_value <= node.threshold {
            if let Some(ref left) = node.left {
                Self::predict_node(left, features)
            } else {
                node.value
            }
        } else if let Some(ref right) = node.right {
            Self::predict_node(right, features)
        } else {
            node.value
        }
    }
}

impl Default for GBDTRanker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::features::{PositionFeatures, QueryDocumentFeatures};
    use std::collections::HashMap;

    fn create_test_features() -> QueryDocumentFeatures {
        QueryDocumentFeatures {
            bm25_score: 2.5,
            tf_idf_score: 1.8,
            edit_distance: 0.9,
            query_term_coverage: 0.8,
            exact_match_count: 2,
            partial_match_count: 1,
            vector_similarity: 0.7,
            semantic_distance: 0.3,
            document_length: 150,
            query_length: 3,
            term_frequency_variance: 0.15,
            inverse_document_frequency_sum: 8.2,
            title_match_score: 0.6,
            field_match_scores: HashMap::new(),
            position_features: PositionFeatures::default(),
            click_through_rate: 0.15,
            document_age_days: 45,
            document_popularity: 0.6,
            query_frequency: 25,
            time_of_day: 0.6,
            day_of_week: 2,
            user_context_score: 0.4,
        }
    }

    #[test]
    fn test_gbdt_ranker_creation() {
        let ranker = GBDTRanker::new();
        assert!(!ranker.is_trained());
        assert_eq!(ranker.learning_rate, 0.1);
        assert_eq!(ranker.max_iterations, 100);
    }

    #[test]
    fn test_gbdt_ranker_with_params() {
        let ranker = GBDTRanker::with_params(0.05, 50, 4, 10, 0.7, 0.9);
        assert_eq!(ranker.learning_rate, 0.05);
        assert_eq!(ranker.max_iterations, 50);
        assert_eq!(ranker.max_depth, 4);
    }

    #[test]
    fn test_untrained_prediction() {
        let ranker = GBDTRanker::new();
        let features = create_test_features();
        let prediction = ranker.predict(&features);
        assert_eq!(prediction, 0.0);
    }

    #[test]
    fn test_insufficient_training_data() {
        let mut ranker = GBDTRanker::new();
        let training_data = vec![LabeledExample {
            query_id: "q1".to_string(),
            document_id: "d1".to_string(),
            features: create_test_features(),
            label: 1.0,
            weight: None,
        }];

        let result = ranker.train(&training_data);
        assert!(result.is_err());
        assert!(!ranker.is_trained());
    }

    #[test]
    fn test_decision_tree_creation() {
        let tree = DecisionTree::new();
        assert!(tree.root.is_none());

        let features = create_test_features();
        let prediction = tree.predict(&features);
        assert_eq!(prediction, 0.0);
    }

    #[test]
    fn test_labeled_example_creation() {
        let features = create_test_features();
        let example = LabeledExample {
            query_id: "query_1".to_string(),
            document_id: "doc_1".to_string(),
            features,
            label: 3.5,
            weight: Some(1.2),
        };

        assert_eq!(example.query_id, "query_1");
        assert_eq!(example.document_id, "doc_1");
        assert_eq!(example.label, 3.5);
        assert_eq!(example.weight, Some(1.2));
    }

    #[test]
    fn test_training_stats_default() {
        let stats = TrainingStats {
            training_losses: vec![1.0, 0.8, 0.6],
            validation_losses: vec![1.1, 0.9, 0.7],
            iterations: 3,
            training_time_ms: 1500,
            final_training_loss: 0.6,
            final_validation_loss: 0.7,
            early_stopped: true,
        };

        assert_eq!(stats.iterations, 3);
        assert_eq!(stats.training_losses.len(), 3);
        assert!(stats.early_stopped);
    }
}
