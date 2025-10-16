//! Recommendation system for improving search personalization.
//!
//! This module provides recommendation capabilities using:
//! - Collaborative filtering
//! - Content-based filtering  
//! - Hybrid recommendation approaches
//! - User preference learning

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::document::document::Document;
use crate::error::Result;
use crate::ml::{FeedbackSignal, MLContext, SearchHistoryItem, UserSession};

/// Configuration for recommendation system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationConfig {
    /// Enable recommendation system.
    pub enabled: bool,
    /// Maximum number of recommendations to generate.
    pub max_recommendations: usize,
    /// Minimum similarity threshold for recommendations.
    pub similarity_threshold: f64,
    /// Enable collaborative filtering.
    pub enable_collaborative: bool,
    /// Enable content-based filtering.
    pub enable_content_based: bool,
    /// Weight for collaborative filtering (0.0 - 1.0).
    pub collaborative_weight: f64,
    /// Weight for content-based filtering (0.0 - 1.0).
    pub content_based_weight: f64,
    /// Minimum user interactions required for collaborative filtering.
    pub min_user_interactions: usize,
    /// Decay factor for time-based weighting.
    pub time_decay_factor: f64,
    /// Popular items boost factor.
    pub popularity_boost: f64,
}

impl Default for RecommendationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_recommendations: 20,
            similarity_threshold: 0.3,
            enable_collaborative: true,
            enable_content_based: true,
            collaborative_weight: 0.6,
            content_based_weight: 0.4,
            min_user_interactions: 5,
            time_decay_factor: 0.95,
            popularity_boost: 0.1,
        }
    }
}

/// Recommendation system.
pub struct RecommendationSystem {
    /// Configuration.
    config: RecommendationConfig,
    /// User-item interaction matrix.
    interaction_matrix: UserItemMatrix,
    /// Item feature vectors for content-based filtering.
    item_features: ItemFeatureMatrix,
    /// User preference profiles.
    user_profiles: HashMap<String, UserProfile>,
    /// Item popularity statistics.
    item_popularity: HashMap<String, PopularityStats>,
    /// Similarity cache.
    similarity_cache: SimilarityCache,
}

impl RecommendationSystem {
    /// Create a new recommendation system.
    pub fn new(config: RecommendationConfig) -> Self {
        Self {
            config,
            interaction_matrix: UserItemMatrix::new(),
            item_features: ItemFeatureMatrix::new(),
            user_profiles: HashMap::new(),
            item_popularity: HashMap::new(),
            similarity_cache: SimilarityCache::new(),
        }
    }

    /// Generate recommendations for a user query.
    pub fn recommend(
        &self,
        query: &str,
        user_context: Option<&UserSession>,
        search_context: &MLContext,
        candidate_documents: &[String],
    ) -> Result<Vec<Recommendation>> {
        if !self.config.enabled || candidate_documents.is_empty() {
            return Ok(Vec::new());
        }

        let mut recommendations = Vec::new();

        // Get user ID for personalization
        let user_id = user_context
            .and_then(|session| session.user_id.as_ref())
            .cloned();

        // Generate collaborative filtering recommendations
        if self.config.enable_collaborative
            && let Some(collab_recs) =
                self.collaborative_recommendations(&user_id, candidate_documents, search_context)?
        {
            recommendations.extend(collab_recs);
        }

        // Generate content-based recommendations
        if self.config.enable_content_based {
            let content_recs = self.content_based_recommendations(
                query,
                &user_id,
                candidate_documents,
                search_context,
            )?;
            recommendations.extend(content_recs);
        }

        // Merge and rank recommendations
        let merged_recommendations = self.merge_recommendations(recommendations)?;

        // Apply final ranking and filtering
        let mut final_recs = merged_recommendations;
        final_recs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        final_recs.truncate(self.config.max_recommendations);

        Ok(final_recs)
    }

    /// Update the recommendation system with user feedback.
    pub fn update_with_feedback(&mut self, feedback: &FeedbackSignal) -> Result<()> {
        // Update interaction matrix
        self.interaction_matrix.add_interaction(
            &feedback.query, // Use query as user context
            &feedback.document_id,
            feedback.relevance_score,
            feedback.timestamp,
        )?;

        // Update item popularity
        let popularity = self
            .item_popularity
            .entry(feedback.document_id.clone())
            .or_insert_with(PopularityStats::new);
        popularity.add_interaction(feedback.relevance_score);

        // Invalidate similarity cache for affected items
        self.similarity_cache
            .invalidate_for_item(&feedback.document_id);

        Ok(())
    }

    /// Update user profile with search history.
    pub fn update_user_profile(
        &mut self,
        user_id: &str,
        search_history: &[SearchHistoryItem],
    ) -> Result<()> {
        let profile = self
            .user_profiles
            .entry(user_id.to_string())
            .or_insert_with(UserProfile::new);

        profile.update_from_history(search_history)?;

        Ok(())
    }

    /// Add document features for content-based filtering.
    pub fn add_document_features(&mut self, document_id: &str, document: &Document) -> Result<()> {
        let features = self.extract_document_features(document)?;
        self.item_features.add_item(document_id, features);
        Ok(())
    }

    // Private recommendation methods

    fn collaborative_recommendations(
        &self,
        user_id: &Option<String>,
        candidates: &[String],
        context: &MLContext,
    ) -> Result<Option<Vec<Recommendation>>> {
        if let Some(uid) = user_id
            && self.interaction_matrix.get_user_interaction_count(uid)
                >= self.config.min_user_interactions
        {
            return Ok(Some(self.generate_collaborative_recommendations(
                uid, candidates, context,
            )?));
        }

        // Fall back to popularity-based recommendations for new users
        Ok(Some(
            self.generate_popularity_based_recommendations(candidates)?,
        ))
    }

    fn generate_collaborative_recommendations(
        &self,
        user_id: &str,
        candidates: &[String],
        context: &MLContext,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // Find similar users
        let similar_users = self.find_similar_users(user_id, 10)?;

        // Recommend items liked by similar users
        for candidate in candidates {
            if !self
                .interaction_matrix
                .has_user_interacted(user_id, candidate)
            {
                let mut score = 0.0;
                let mut similarity_sum = 0.0;

                for (similar_user, similarity) in &similar_users {
                    if let Some(interaction_score) = self
                        .interaction_matrix
                        .get_interaction(similar_user, candidate)
                    {
                        score += similarity * interaction_score;
                        similarity_sum += similarity.abs();
                    }
                }

                if similarity_sum > 0.0 {
                    score /= similarity_sum;

                    // Apply time decay
                    let time_weight = self.calculate_time_weight(context.timestamp);
                    score *= time_weight;

                    if score >= self.config.similarity_threshold {
                        recommendations.push(Recommendation {
                            document_id: candidate.clone(),
                            score: score * self.config.collaborative_weight,
                            recommendation_type: RecommendationType::Collaborative,
                            explanation: Some(
                                "Users with similar interests also liked this".to_string(),
                            ),
                            confidence: similarity_sum / similar_users.len() as f64,
                        });
                    }
                }
            }
        }

        Ok(recommendations)
    }

    fn generate_popularity_based_recommendations(
        &self,
        candidates: &[String],
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        for candidate in candidates {
            if let Some(popularity) = self.item_popularity.get(candidate) {
                let score = popularity.average_score * self.config.popularity_boost;

                if score >= self.config.similarity_threshold {
                    recommendations.push(Recommendation {
                        document_id: candidate.clone(),
                        score,
                        recommendation_type: RecommendationType::Popular,
                        explanation: Some("Popular among other users".to_string()),
                        confidence: 0.5, // Medium confidence for popularity-based
                    });
                }
            }
        }

        Ok(recommendations)
    }

    fn content_based_recommendations(
        &self,
        query: &str,
        user_id: &Option<String>,
        candidates: &[String],
        _context: &MLContext,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // Create query feature vector
        let query_features = self.extract_query_features(query)?;

        // Get user preferences if available
        let user_preferences = user_id
            .as_ref()
            .and_then(|id| self.user_profiles.get(id))
            .map(|profile| &profile.preferences)
            .cloned()
            .unwrap_or_default();

        for candidate in candidates {
            if let Some(item_features) = self.item_features.get_features(candidate) {
                // Calculate content similarity
                let content_similarity =
                    self.calculate_feature_similarity(&query_features, item_features)?;

                // Calculate user preference similarity
                let user_similarity = if !user_preferences.is_empty() {
                    self.calculate_feature_similarity(&user_preferences, item_features)?
                } else {
                    0.0
                };

                // Combine similarities
                let combined_score = (0.7 * content_similarity + 0.3 * user_similarity)
                    * self.config.content_based_weight;

                if combined_score >= self.config.similarity_threshold {
                    recommendations.push(Recommendation {
                        document_id: candidate.clone(),
                        score: combined_score,
                        recommendation_type: RecommendationType::ContentBased,
                        explanation: Some("Similar to your query and preferences".to_string()),
                        confidence: content_similarity.max(user_similarity),
                    });
                }
            }
        }

        Ok(recommendations)
    }

    fn merge_recommendations(
        &self,
        recommendations: Vec<Recommendation>,
    ) -> Result<Vec<Recommendation>> {
        let mut merged: HashMap<String, Recommendation> = HashMap::new();

        for rec in recommendations {
            match merged.get_mut(&rec.document_id) {
                Some(existing) => {
                    // Combine scores and update metadata
                    existing.score = (existing.score + rec.score).max(existing.score);
                    existing.confidence = existing.confidence.max(rec.confidence);

                    // Combine explanations
                    if let (Some(existing_exp), Some(rec_exp)) =
                        (&mut existing.explanation, &rec.explanation)
                    {
                        if !existing_exp.contains(rec_exp) {
                            existing_exp.push_str(&format!("; {rec_exp}"));
                        }
                    } else if existing.explanation.is_none() {
                        existing.explanation = rec.explanation;
                    }
                }
                None => {
                    merged.insert(rec.document_id.clone(), rec);
                }
            }
        }

        Ok(merged.into_values().collect())
    }

    // Helper methods

    fn find_similar_users(&self, user_id: &str, count: usize) -> Result<Vec<(String, f64)>> {
        let user_vector = self.interaction_matrix.get_user_vector(user_id);
        let mut similarities = Vec::new();

        for other_user in self.interaction_matrix.get_all_users() {
            if other_user != user_id {
                let other_vector = self.interaction_matrix.get_user_vector(&other_user);
                let similarity = self.cosine_similarity(&user_vector, &other_vector);

                if similarity > 0.0 {
                    similarities.push((other_user, similarity));
                }
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(count);

        Ok(similarities)
    }

    fn calculate_feature_similarity(&self, features1: &[f64], features2: &[f64]) -> Result<f64> {
        if features1.len() != features2.len() {
            return Ok(0.0);
        }

        Ok(self.cosine_similarity(features1, features2))
    }

    fn cosine_similarity(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
        if vec1.len() != vec2.len() || vec1.is_empty() {
            return 0.0;
        }

        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    fn calculate_time_weight(&self, current_time: chrono::DateTime<chrono::Utc>) -> f64 {
        // Simple exponential decay based on time
        let hours_ago = (current_time - chrono::Utc::now()).num_hours().abs() as f64;
        self.config.time_decay_factor.powf(hours_ago / 24.0) // Daily decay
    }

    fn extract_query_features(&self, query: &str) -> Result<Vec<f64>> {
        // Simple TF-IDF-like features for query
        let terms: Vec<&str> = query.split_whitespace().collect();
        let mut features = vec![0.0; 100]; // Fixed size feature vector

        // Hash terms to feature indices
        for term in terms {
            let hash = self.simple_hash(term);
            let index = hash % features.len();
            features[index] += 1.0;
        }

        // Normalize
        let norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for feature in &mut features {
                *feature /= norm;
            }
        }

        Ok(features)
    }

    fn extract_document_features(&self, document: &Document) -> Result<Vec<f64>> {
        // Extract features from document content
        let mut features = vec![0.0; 100]; // Fixed size feature vector

        // Extract text from all fields
        let mut all_text = String::new();
        for field_name in document.field_names() {
            if let Some(field_value) = document.get_field(field_name)
                && let Some(text) = field_value.as_text()
            {
                all_text.push_str(text);
                all_text.push(' ');
            }
        }

        let terms: Vec<&str> = all_text.split_whitespace().collect();

        // Create feature vector
        for term in terms {
            let hash = self.simple_hash(term);
            let index = hash % features.len();
            features[index] += 1.0;
        }

        // Normalize
        let norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for feature in &mut features {
                *feature /= norm;
            }
        }

        Ok(features)
    }

    fn simple_hash(&self, text: &str) -> usize {
        let mut hash = 0usize;
        for byte in text.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as usize);
        }
        hash
    }
}

/// A single recommendation result.
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// Document ID being recommended.
    pub document_id: String,
    /// Recommendation score (0.0 - 1.0).
    pub score: f64,
    /// Type of recommendation.
    pub recommendation_type: RecommendationType,
    /// Human-readable explanation.
    pub explanation: Option<String>,
    /// Confidence in the recommendation.
    pub confidence: f64,
}

/// Types of recommendation algorithms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Collaborative filtering based.
    Collaborative,
    /// Content-based filtering.
    ContentBased,
    /// Hybrid approach.
    Hybrid,
    /// Popular items.
    Popular,
    /// Similar items.
    Similar,
}

/// User-item interaction matrix for collaborative filtering.
#[derive(Debug)]
struct UserItemMatrix {
    interactions: HashMap<String, HashMap<String, f64>>,
    user_interaction_counts: HashMap<String, usize>,
}

impl UserItemMatrix {
    fn new() -> Self {
        Self {
            interactions: HashMap::new(),
            user_interaction_counts: HashMap::new(),
        }
    }

    fn add_interaction(
        &mut self,
        user_id: &str,
        item_id: &str,
        score: f64,
        _timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        self.interactions
            .entry(user_id.to_string())
            .or_default()
            .insert(item_id.to_string(), score);

        *self
            .user_interaction_counts
            .entry(user_id.to_string())
            .or_insert(0) += 1;

        Ok(())
    }

    fn get_interaction(&self, user_id: &str, item_id: &str) -> Option<f64> {
        self.interactions.get(user_id)?.get(item_id).copied()
    }

    fn has_user_interacted(&self, user_id: &str, item_id: &str) -> bool {
        self.interactions
            .get(user_id)
            .map(|items| items.contains_key(item_id))
            .unwrap_or(false)
    }

    fn get_user_interaction_count(&self, user_id: &str) -> usize {
        self.user_interaction_counts
            .get(user_id)
            .copied()
            .unwrap_or(0)
    }

    fn get_user_vector(&self, user_id: &str) -> Vec<f64> {
        self.interactions
            .get(user_id)
            .map(|items| items.values().copied().collect())
            .unwrap_or_default()
    }

    fn get_all_users(&self) -> Vec<String> {
        self.interactions.keys().cloned().collect()
    }
}

/// Item feature matrix for content-based filtering.
#[derive(Debug)]
struct ItemFeatureMatrix {
    features: HashMap<String, Vec<f64>>,
}

impl ItemFeatureMatrix {
    fn new() -> Self {
        Self {
            features: HashMap::new(),
        }
    }

    fn add_item(&mut self, item_id: &str, features: Vec<f64>) {
        self.features.insert(item_id.to_string(), features);
    }

    fn get_features(&self, item_id: &str) -> Option<&Vec<f64>> {
        self.features.get(item_id)
    }
}

/// User preference profile.
#[derive(Debug)]
struct UserProfile {
    preferences: Vec<f64>,
    last_updated: chrono::DateTime<chrono::Utc>,
}

impl UserProfile {
    fn new() -> Self {
        Self {
            preferences: vec![0.0; 100], // Fixed size preference vector
            last_updated: chrono::Utc::now(),
        }
    }

    fn update_from_history(&mut self, history: &[SearchHistoryItem]) -> Result<()> {
        // Simple preference learning from search history
        for _item in history {
            // Update preferences based on clicked documents
            // This is a simplified implementation
        }
        self.last_updated = chrono::Utc::now();
        Ok(())
    }
}

/// Item popularity statistics.
#[derive(Debug)]
struct PopularityStats {
    total_interactions: u64,
    total_score: f64,
    average_score: f64,
}

impl PopularityStats {
    fn new() -> Self {
        Self {
            total_interactions: 0,
            total_score: 0.0,
            average_score: 0.0,
        }
    }

    fn add_interaction(&mut self, score: f64) {
        self.total_interactions += 1;
        self.total_score += score;
        self.average_score = self.total_score / self.total_interactions as f64;
    }
}

/// Similarity computation cache.
#[derive(Debug)]
struct SimilarityCache {
    cache: HashMap<String, HashMap<String, f64>>,
}

impl SimilarityCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    fn invalidate_for_item(&mut self, item_id: &str) {
        self.cache.remove(item_id);
        // Also remove from other items' caches
        for (_, similarities) in self.cache.iter_mut() {
            similarities.remove(item_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::FeedbackType;

    #[test]
    fn test_recommendation_config_default() {
        let config = RecommendationConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_recommendations, 20);
        assert_eq!(config.similarity_threshold, 0.3);
    }

    #[test]
    fn test_recommendation_system_creation() {
        let config = RecommendationConfig::default();
        let system = RecommendationSystem::new(config);
        assert!(system.config.enabled);
    }

    #[test]
    fn test_user_item_matrix() {
        let mut matrix = UserItemMatrix::new();

        matrix
            .add_interaction("user1", "item1", 0.8, chrono::Utc::now())
            .unwrap();
        matrix
            .add_interaction("user1", "item2", 0.6, chrono::Utc::now())
            .unwrap();

        assert_eq!(matrix.get_interaction("user1", "item1"), Some(0.8));
        assert!(matrix.has_user_interacted("user1", "item1"));
        assert!(!matrix.has_user_interacted("user1", "item3"));
        assert_eq!(matrix.get_user_interaction_count("user1"), 2);
    }

    #[test]
    fn test_recommendation_feedback_update() {
        let mut system = RecommendationSystem::new(RecommendationConfig::default());

        let feedback = FeedbackSignal {
            query: "rust programming".to_string(),
            document_id: "doc1".to_string(),
            feedback_type: FeedbackType::Click,
            relevance_score: 0.9,
            timestamp: chrono::Utc::now(),
        };

        let result = system.update_with_feedback(&feedback);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cosine_similarity() {
        let system = RecommendationSystem::new(RecommendationConfig::default());

        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let similarity = system.cosine_similarity(&vec1, &vec2);

        // Should be 1.0 for identical vectors
        assert!((similarity - 1.0).abs() < 0.0001);

        let vec3 = vec![0.0, 0.0, 0.0];
        let similarity2 = system.cosine_similarity(&vec1, &vec3);

        // Should be 0.0 for zero vector
        assert_eq!(similarity2, 0.0);
    }

    #[test]
    fn test_recommendation_types() {
        let rec = Recommendation {
            document_id: "doc1".to_string(),
            score: 0.8,
            recommendation_type: RecommendationType::Collaborative,
            explanation: Some("Similar users liked this".to_string()),
            confidence: 0.7,
        };

        assert_eq!(rec.document_id, "doc1");
        assert_eq!(rec.score, 0.8);
        assert_eq!(rec.recommendation_type, RecommendationType::Collaborative);
    }
}
