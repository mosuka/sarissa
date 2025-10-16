//! Auto-optimization system for search parameter tuning.
//!
//! This module provides automatic optimization capabilities for:
//! - Search parameter optimization
//! - Algorithm selection and tuning
//! - A/B testing framework
//! - Performance optimization

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ml::anomaly::AnomalyEvent;
use crate::ml::{FeedbackSignal, MLContext};
use crate::query::SearchResults;

/// Configuration for auto-optimization system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoOptimizationConfig {
    /// Enable auto-optimization.
    pub enabled: bool,
    /// Optimization interval in hours.
    pub optimization_interval_hours: u64,
    /// Minimum samples required for optimization.
    pub min_samples: usize,
    /// Enable parameter optimization.
    pub enable_parameter_optimization: bool,
    /// Enable algorithm selection.
    pub enable_algorithm_selection: bool,
    /// Enable A/B testing.
    pub enable_ab_testing: bool,
    /// Statistical significance threshold for A/B tests.
    pub significance_threshold: f64,
    /// Maximum number of A/B test variants.
    pub max_ab_variants: usize,
    /// Performance improvement threshold to accept changes.
    pub improvement_threshold: f64,
}

impl Default for AutoOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_interval_hours: 24,
            min_samples: 100,
            enable_parameter_optimization: true,
            enable_algorithm_selection: true,
            enable_ab_testing: true,
            significance_threshold: 0.05,
            max_ab_variants: 4,
            improvement_threshold: 0.05, // 5% improvement
        }
    }
}

/// Auto-optimization system.
pub struct AutoOptimization {
    /// Configuration.
    config: AutoOptimizationConfig,
    /// Parameter optimizer.
    parameter_optimizer: ParameterOptimizer,
    /// Algorithm selector.
    algorithm_selector: AlgorithmSelector,
    /// A/B testing framework.
    ab_tester: ABTester,
    /// Performance tracker.
    performance_tracker: PerformanceTracker,
    /// Last optimization timestamp.
    last_optimization: chrono::DateTime<chrono::Utc>,
}

impl AutoOptimization {
    /// Create a new auto-optimization system.
    pub fn new(config: AutoOptimizationConfig) -> Self {
        Self {
            config: config.clone(),
            parameter_optimizer: ParameterOptimizer::new(&config),
            algorithm_selector: AlgorithmSelector::new(&config),
            ab_tester: ABTester::new(&config),
            performance_tracker: PerformanceTracker::new(),
            last_optimization: chrono::Utc::now(),
        }
    }

    /// Get current optimization recommendations.
    pub fn get_optimization_recommendations(&mut self) -> Result<Vec<OptimizationRecommendation>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let mut recommendations = Vec::new();

        // Check if it's time for optimization
        let hours_since_last = (chrono::Utc::now() - self.last_optimization).num_hours();
        if hours_since_last < self.config.optimization_interval_hours as i64 {
            return Ok(recommendations);
        }

        // Parameter optimization recommendations
        if self.config.enable_parameter_optimization
            && let Some(param_rec) = self.parameter_optimizer.get_recommendations()?
        {
            recommendations.push(param_rec);
        }

        // Algorithm selection recommendations
        if self.config.enable_algorithm_selection
            && let Some(algo_rec) = self.algorithm_selector.get_recommendations()?
        {
            recommendations.push(algo_rec);
        }

        // A/B test recommendations
        if self.config.enable_ab_testing {
            recommendations.extend(self.ab_tester.get_recommendations()?);
        }

        self.last_optimization = chrono::Utc::now();
        Ok(recommendations)
    }

    /// Apply optimization recommendation.
    pub fn apply_optimization(
        &mut self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<()> {
        match &recommendation.optimization_type {
            OptimizationType::ParameterTuning => {
                self.parameter_optimizer
                    .apply_optimization(recommendation)?;
            }
            OptimizationType::AlgorithmSelection => {
                self.algorithm_selector.apply_optimization(recommendation)?;
            }
            OptimizationType::ABTest => {
                self.ab_tester.apply_optimization(recommendation)?;
            }
        }

        Ok(())
    }

    /// Record search performance data.
    pub fn record_search_performance(
        &mut self,
        query: &str,
        results: &SearchResults,
        feedback_signals: &[FeedbackSignal],
        response_time_ms: u64,
        _context: &MLContext,
    ) -> Result<()> {
        let performance_data = SearchPerformanceData {
            query: query.to_string(),
            result_count: results.hits.len(),
            response_time_ms,
            relevance_scores: feedback_signals.iter().map(|f| f.relevance_score).collect(),
            click_through_rate: self.calculate_ctr(feedback_signals),
            timestamp: chrono::Utc::now(),
        };

        self.performance_tracker.add_data(performance_data)?;
        self.parameter_optimizer
            .add_performance_data(query, results, feedback_signals)?;
        self.algorithm_selector
            .add_performance_data(query, results, feedback_signals)?;
        self.ab_tester
            .add_performance_data(query, results, feedback_signals)?;

        Ok(())
    }

    /// Handle anomaly events for optimization.
    pub fn handle_anomaly(
        &mut self,
        anomaly: &AnomalyEvent,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        match &anomaly.anomaly_type {
            crate::ml::anomaly::AnomalyType::Performance => {
                // Recommend parameter adjustments for performance issues
                if let Some(rec) = self
                    .parameter_optimizer
                    .handle_performance_anomaly(anomaly)?
                {
                    recommendations.push(rec);
                }
            }
            crate::ml::anomaly::AnomalyType::ResultQuality => {
                // Recommend algorithm adjustments for quality issues
                if let Some(rec) = self.algorithm_selector.handle_quality_anomaly(anomaly)? {
                    recommendations.push(rec);
                }
            }
            _ => {}
        }

        Ok(recommendations)
    }

    /// Get optimization statistics.
    pub fn get_optimization_stats(&self) -> OptimizationStats {
        OptimizationStats {
            total_optimizations: self.parameter_optimizer.optimization_count()
                + self.algorithm_selector.optimization_count()
                + self.ab_tester.optimization_count(),
            parameter_optimizations: self.parameter_optimizer.optimization_count(),
            algorithm_optimizations: self.algorithm_selector.optimization_count(),
            ab_tests_completed: self.ab_tester.completed_tests(),
            average_improvement: self.performance_tracker.average_improvement(),
            last_optimization: self.last_optimization,
        }
    }

    // Helper methods

    fn calculate_ctr(&self, feedback_signals: &[FeedbackSignal]) -> f64 {
        if feedback_signals.is_empty() {
            return 0.0;
        }

        let click_count = feedback_signals
            .iter()
            .filter(|f| matches!(f.feedback_type, crate::ml::FeedbackType::Click))
            .count();

        click_count as f64 / feedback_signals.len() as f64
    }
}

/// Parameter optimizer for search parameters.
#[derive(Debug)]
struct ParameterOptimizer {
    current_params: SearchParameters,
    performance_history: VecDeque<ParameterPerformance>,
    optimization_count: usize,
    config: AutoOptimizationConfig,
}

impl ParameterOptimizer {
    fn new(config: &AutoOptimizationConfig) -> Self {
        Self {
            current_params: SearchParameters::default(),
            performance_history: VecDeque::new(),
            optimization_count: 0,
            config: config.clone(),
        }
    }

    fn get_recommendations(&mut self) -> Result<Option<OptimizationRecommendation>> {
        if self.performance_history.len() < self.config.min_samples {
            return Ok(None);
        }

        // Analyze current performance
        let current_performance = self.calculate_current_performance();

        // Generate parameter suggestions using grid search
        if let Some(better_params) = self.find_better_parameters(current_performance)? {
            return Ok(Some(OptimizationRecommendation {
                optimization_type: OptimizationType::ParameterTuning,
                recommendation: format!("Adjust search parameters: {:?}", better_params.changes),
                expected_improvement: better_params.expected_improvement,
                confidence: better_params.confidence,
                parameters: Some(better_params.parameters),
                timestamp: chrono::Utc::now(),
            }));
        }

        Ok(None)
    }

    fn apply_optimization(&mut self, recommendation: &OptimizationRecommendation) -> Result<()> {
        if let Some(new_params) = &recommendation.parameters {
            self.current_params.update_from_map(new_params)?;
            self.optimization_count += 1;
        }
        Ok(())
    }

    fn add_performance_data(
        &mut self,
        _query: &str,
        results: &SearchResults,
        feedback: &[FeedbackSignal],
    ) -> Result<()> {
        let performance = ParameterPerformance {
            parameters: self.current_params.clone(),
            avg_relevance: feedback.iter().map(|f| f.relevance_score).sum::<f64>()
                / feedback.len().max(1) as f64,
            response_time_score: 1.0 / (results.hits.len() as f64 + 1.0), // Simplified metric
            result_count: results.hits.len(),
            timestamp: chrono::Utc::now(),
        };

        self.performance_history.push_back(performance);

        // Limit history size
        while self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        Ok(())
    }

    fn handle_performance_anomaly(
        &mut self,
        _anomaly: &AnomalyEvent,
    ) -> Result<Option<OptimizationRecommendation>> {
        // Generate immediate optimization for performance issues
        Ok(Some(OptimizationRecommendation {
            optimization_type: OptimizationType::ParameterTuning,
            recommendation: "Adjust parameters to address performance anomaly".to_string(),
            expected_improvement: 0.1,
            confidence: 0.7,
            parameters: Some(HashMap::from([
                ("timeout_ms".to_string(), "5000".to_string()),
                ("max_results".to_string(), "50".to_string()),
            ])),
            timestamp: chrono::Utc::now(),
        }))
    }

    fn optimization_count(&self) -> usize {
        self.optimization_count
    }

    // Private helper methods

    fn calculate_current_performance(&self) -> f64 {
        if self.performance_history.is_empty() {
            return 0.0;
        }

        let recent_performances: Vec<_> = self.performance_history.iter().rev().take(50).collect();

        let avg_relevance: f64 = recent_performances
            .iter()
            .map(|p| p.avg_relevance)
            .sum::<f64>()
            / recent_performances.len() as f64;

        let avg_response_score: f64 = recent_performances
            .iter()
            .map(|p| p.response_time_score)
            .sum::<f64>()
            / recent_performances.len() as f64;

        // Combined score
        0.7 * avg_relevance + 0.3 * avg_response_score
    }

    fn find_better_parameters(
        &self,
        _current_performance: f64,
    ) -> Result<Option<ParameterSuggestion>> {
        // Simplified parameter optimization - in practice, this would use more sophisticated methods
        let mut best_suggestion = None;
        let mut best_improvement = 0.0;

        // Try different parameter combinations
        let param_variants = vec![
            HashMap::from([("bm25_k1".to_string(), "1.5".to_string())]),
            HashMap::from([("bm25_b".to_string(), "0.5".to_string())]),
            HashMap::from([("max_results".to_string(), "100".to_string())]),
        ];

        for variant in param_variants {
            let expected_improvement = 0.05; // Simplified estimation
            if expected_improvement > best_improvement
                && expected_improvement > self.config.improvement_threshold
            {
                best_improvement = expected_improvement;
                best_suggestion = Some(ParameterSuggestion {
                    parameters: variant.clone(),
                    changes: format!("Parameter adjustment: {variant:?}"),
                    expected_improvement,
                    confidence: 0.6,
                });
            }
        }

        Ok(best_suggestion)
    }
}

/// Algorithm selector for choosing optimal algorithms.
#[derive(Debug)]
struct AlgorithmSelector {
    current_algorithms: AlgorithmConfiguration,
    algorithm_performance: HashMap<String, Vec<f64>>,
    optimization_count: usize,
    #[allow(dead_code)]
    config: AutoOptimizationConfig,
}

impl AlgorithmSelector {
    fn new(config: &AutoOptimizationConfig) -> Self {
        Self {
            current_algorithms: AlgorithmConfiguration::default(),
            algorithm_performance: HashMap::new(),
            optimization_count: 0,
            config: config.clone(),
        }
    }

    fn get_recommendations(&mut self) -> Result<Option<OptimizationRecommendation>> {
        // Analyze algorithm performance and suggest better alternatives
        if let Some(best_algo) = self.find_best_algorithm()?
            && best_algo != self.current_algorithms.primary_algorithm
        {
            return Ok(Some(OptimizationRecommendation {
                optimization_type: OptimizationType::AlgorithmSelection,
                recommendation: format!("Switch to {best_algo} algorithm"),
                expected_improvement: 0.08,
                confidence: 0.8,
                parameters: Some(HashMap::from([("algorithm".to_string(), best_algo)])),
                timestamp: chrono::Utc::now(),
            }));
        }

        Ok(None)
    }

    fn apply_optimization(&mut self, recommendation: &OptimizationRecommendation) -> Result<()> {
        if let Some(params) = &recommendation.parameters
            && let Some(algorithm) = params.get("algorithm")
        {
            self.current_algorithms.primary_algorithm = algorithm.clone();
            self.optimization_count += 1;
        }
        Ok(())
    }

    fn add_performance_data(
        &mut self,
        _query: &str,
        _results: &SearchResults,
        feedback: &[FeedbackSignal],
    ) -> Result<()> {
        let performance =
            feedback.iter().map(|f| f.relevance_score).sum::<f64>() / feedback.len().max(1) as f64;

        self.algorithm_performance
            .entry(self.current_algorithms.primary_algorithm.clone())
            .or_default()
            .push(performance);

        Ok(())
    }

    fn handle_quality_anomaly(
        &mut self,
        _anomaly: &AnomalyEvent,
    ) -> Result<Option<OptimizationRecommendation>> {
        Ok(Some(OptimizationRecommendation {
            optimization_type: OptimizationType::AlgorithmSelection,
            recommendation: "Switch algorithm to address quality issues".to_string(),
            expected_improvement: 0.12,
            confidence: 0.6,
            parameters: Some(HashMap::from([(
                "algorithm".to_string(),
                "BM25Plus".to_string(),
            )])),
            timestamp: chrono::Utc::now(),
        }))
    }

    fn optimization_count(&self) -> usize {
        self.optimization_count
    }

    fn find_best_algorithm(&self) -> Result<Option<String>> {
        let mut best_algorithm = None;
        let mut best_performance = 0.0;

        for (algorithm, performances) in &self.algorithm_performance {
            if performances.len() >= 10 {
                let avg_performance = performances.iter().sum::<f64>() / performances.len() as f64;
                if avg_performance > best_performance {
                    best_performance = avg_performance;
                    best_algorithm = Some(algorithm.clone());
                }
            }
        }

        Ok(best_algorithm)
    }
}

/// A/B testing framework.
#[derive(Debug)]
struct ABTester {
    active_tests: HashMap<String, ABTest>,
    completed_tests: usize,
    config: AutoOptimizationConfig,
}

impl ABTester {
    fn new(config: &AutoOptimizationConfig) -> Self {
        Self {
            active_tests: HashMap::new(),
            completed_tests: 0,
            config: config.clone(),
        }
    }

    fn get_recommendations(&mut self) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        // Check for completed A/B tests
        let mut completed_test_ids = Vec::new();

        for (test_id, test) in &self.active_tests {
            if test.is_completed(&self.config)
                && let Some(winner) = test.get_winner(&self.config)?
            {
                recommendations.push(OptimizationRecommendation {
                    optimization_type: OptimizationType::ABTest,
                    recommendation: format!(
                        "A/B test '{}' completed. Winner: {}",
                        test_id, winner.variant_name
                    ),
                    expected_improvement: winner.improvement,
                    confidence: winner.confidence,
                    parameters: winner.parameters.clone(),
                    timestamp: chrono::Utc::now(),
                });

                completed_test_ids.push(test_id.clone());
            }
        }

        // Remove completed tests
        for test_id in completed_test_ids {
            self.active_tests.remove(&test_id);
            self.completed_tests += 1;
        }

        // Suggest new A/B tests if we have capacity
        if self.active_tests.len() < 2 && recommendations.is_empty() {
            recommendations.push(self.suggest_new_ab_test()?);
        }

        Ok(recommendations)
    }

    fn apply_optimization(&mut self, recommendation: &OptimizationRecommendation) -> Result<()> {
        // Start new A/B test or apply winning variant
        if recommendation.recommendation.contains("A/B test")
            && recommendation.recommendation.contains("completed")
        {
            // Apply winning variant - implementation would update system configuration
        } else {
            // Start new A/B test
            let test_id = format!("test_{}", self.active_tests.len() + 1);
            let test = ABTest::new(
                test_id.clone(),
                recommendation.recommendation.clone(),
                recommendation.parameters.clone(),
            );
            self.active_tests.insert(test_id, test);
        }

        Ok(())
    }

    fn add_performance_data(
        &mut self,
        query: &str,
        results: &SearchResults,
        feedback: &[FeedbackSignal],
    ) -> Result<()> {
        // Add data to all active A/B tests
        for test in self.active_tests.values_mut() {
            test.add_performance_data(query, results, feedback)?;
        }

        Ok(())
    }

    fn completed_tests(&self) -> usize {
        self.completed_tests
    }

    fn optimization_count(&self) -> usize {
        self.active_tests.len() + self.completed_tests
    }

    fn suggest_new_ab_test(&self) -> Result<OptimizationRecommendation> {
        // Suggest a new A/B test based on current needs
        Ok(OptimizationRecommendation {
            optimization_type: OptimizationType::ABTest,
            recommendation: "Start A/B test for BM25 parameter optimization".to_string(),
            expected_improvement: 0.06,
            confidence: 0.5,
            parameters: Some(HashMap::from([
                ("test_type".to_string(), "parameter_tuning".to_string()),
                ("parameter".to_string(), "bm25_k1".to_string()),
            ])),
            timestamp: chrono::Utc::now(),
        })
    }
}

// Supporting data structures

#[derive(Debug, Clone)]
struct SearchParameters {
    bm25_k1: f64,
    #[allow(dead_code)]
    bm25_b: f64,
    #[allow(dead_code)]
    max_results: usize,
    #[allow(dead_code)]
    timeout_ms: u64,
}

impl Default for SearchParameters {
    fn default() -> Self {
        Self {
            bm25_k1: 1.2,
            bm25_b: 0.75,
            max_results: 100,
            timeout_ms: 10000,
        }
    }
}

impl SearchParameters {
    fn update_from_map(&mut self, params: &HashMap<String, String>) -> Result<()> {
        if let Some(k1) = params.get("bm25_k1") {
            self.bm25_k1 = k1
                .parse()
                .map_err(|_| crate::ml::MLError::InvalidFeatureVector {
                    message: "Invalid bm25_k1 value".to_string(),
                })?;
        }
        // Similar for other parameters...
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct AlgorithmConfiguration {
    primary_algorithm: String,
    #[allow(dead_code)]
    fallback_algorithm: String,
}

impl Default for AlgorithmConfiguration {
    fn default() -> Self {
        Self {
            primary_algorithm: "BM25".to_string(),
            fallback_algorithm: "TFIDF".to_string(),
        }
    }
}

#[derive(Debug)]
struct ParameterPerformance {
    #[allow(dead_code)]
    parameters: SearchParameters,
    avg_relevance: f64,
    response_time_score: f64,
    #[allow(dead_code)]
    result_count: usize,
    #[allow(dead_code)]
    timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug)]
struct ParameterSuggestion {
    parameters: HashMap<String, String>,
    changes: String,
    expected_improvement: f64,
    confidence: f64,
}

#[derive(Debug)]
struct ABTest {
    #[allow(dead_code)]
    test_id: String,
    #[allow(dead_code)]
    description: String,
    variants: Vec<ABTestVariant>,
    start_time: chrono::DateTime<chrono::Utc>,
}

impl ABTest {
    fn new(test_id: String, description: String, params: Option<HashMap<String, String>>) -> Self {
        let mut variants = vec![ABTestVariant::new("control".to_string(), None)];

        if let Some(test_params) = params {
            variants.push(ABTestVariant::new(
                "variant_a".to_string(),
                Some(test_params),
            ));
        }

        Self {
            test_id,
            description,
            variants,
            start_time: chrono::Utc::now(),
        }
    }

    fn add_performance_data(
        &mut self,
        _query: &str,
        _results: &SearchResults,
        feedback: &[FeedbackSignal],
    ) -> Result<()> {
        // Randomly assign to variant and record performance
        let variant_index = rand::random::<u8>() as usize % self.variants.len();
        let performance =
            feedback.iter().map(|f| f.relevance_score).sum::<f64>() / feedback.len().max(1) as f64;

        self.variants[variant_index].add_performance(performance);
        Ok(())
    }

    fn is_completed(&self, config: &AutoOptimizationConfig) -> bool {
        // Check if test has enough samples and statistical significance
        self.variants
            .iter()
            .all(|v| v.sample_count >= config.min_samples)
            && (chrono::Utc::now() - self.start_time).num_hours() >= 24
    }

    fn get_winner(&self, config: &AutoOptimizationConfig) -> Result<Option<ABTestWinner>> {
        if !self.is_completed(config) {
            return Ok(None);
        }

        // Find best performing variant
        let mut best_variant = None;
        let mut best_performance = 0.0;

        for variant in &self.variants {
            let performance = variant.average_performance();
            if performance > best_performance {
                best_performance = performance;
                best_variant = Some(variant);
            }
        }

        if let Some(winner) = best_variant {
            // Calculate statistical significance (simplified)
            let control_performance = self.variants[0].average_performance();
            let improvement =
                (best_performance - control_performance) / control_performance.max(0.001);

            if improvement > 0.02 {
                // 2% improvement threshold
                return Ok(Some(ABTestWinner {
                    variant_name: winner.variant_name.clone(),
                    improvement,
                    confidence: 0.8, // Simplified confidence calculation
                    parameters: winner.parameters.clone(),
                }));
            }
        }

        Ok(None)
    }
}

#[derive(Debug)]
struct ABTestVariant {
    variant_name: String,
    parameters: Option<HashMap<String, String>>,
    performance_data: Vec<f64>,
    sample_count: usize,
}

impl ABTestVariant {
    fn new(variant_name: String, parameters: Option<HashMap<String, String>>) -> Self {
        Self {
            variant_name,
            parameters,
            performance_data: Vec::new(),
            sample_count: 0,
        }
    }

    fn add_performance(&mut self, performance: f64) {
        self.performance_data.push(performance);
        self.sample_count += 1;
    }

    fn average_performance(&self) -> f64 {
        if self.performance_data.is_empty() {
            0.0
        } else {
            self.performance_data.iter().sum::<f64>() / self.performance_data.len() as f64
        }
    }
}

#[derive(Debug)]
struct ABTestWinner {
    variant_name: String,
    improvement: f64,
    confidence: f64,
    parameters: Option<HashMap<String, String>>,
}

#[derive(Debug)]
struct PerformanceTracker {
    performance_history: VecDeque<f64>,
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            performance_history: VecDeque::new(),
        }
    }

    fn add_data(&mut self, data: SearchPerformanceData) -> Result<()> {
        let performance_score =
            data.relevance_scores.iter().sum::<f64>() / data.relevance_scores.len().max(1) as f64;
        self.performance_history.push_back(performance_score);

        // Limit history
        while self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        Ok(())
    }

    fn average_improvement(&self) -> f64 {
        if self.performance_history.len() < 2 {
            return 0.0;
        }

        let recent: f64 = self.performance_history.iter().rev().take(50).sum::<f64>() / 50.0;
        let older: f64 = self
            .performance_history
            .iter()
            .rev()
            .skip(50)
            .take(50)
            .sum::<f64>()
            / 50.0;

        if older > 0.0 {
            (recent - older) / older
        } else {
            0.0
        }
    }
}

/// Public API structures
/// Optimization recommendation from the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Type of optimization.
    pub optimization_type: OptimizationType,
    /// Human-readable recommendation.
    pub recommendation: String,
    /// Expected performance improvement.
    pub expected_improvement: f64,
    /// Confidence in the recommendation.
    pub confidence: f64,
    /// Parameters to apply.
    pub parameters: Option<HashMap<String, String>>,
    /// When the recommendation was generated.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of optimizations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Parameter tuning optimization.
    ParameterTuning,
    /// Algorithm selection optimization.
    AlgorithmSelection,
    /// A/B test result.
    ABTest,
}

/// Search performance data for optimization.
#[derive(Debug, Clone)]
struct SearchPerformanceData {
    #[allow(dead_code)]
    query: String,
    #[allow(dead_code)]
    result_count: usize,
    #[allow(dead_code)]
    response_time_ms: u64,
    relevance_scores: Vec<f64>,
    #[allow(dead_code)]
    click_through_rate: f64,
    #[allow(dead_code)]
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Optimization system statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Total number of optimizations performed.
    pub total_optimizations: usize,
    /// Number of parameter optimizations.
    pub parameter_optimizations: usize,
    /// Number of algorithm optimizations.
    pub algorithm_optimizations: usize,
    /// Number of completed A/B tests.
    pub ab_tests_completed: usize,
    /// Average performance improvement.
    pub average_improvement: f64,
    /// Last optimization timestamp.
    pub last_optimization: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_optimization_config_default() {
        let config = AutoOptimizationConfig::default();
        assert!(config.enabled);
        assert_eq!(config.optimization_interval_hours, 24);
        assert_eq!(config.min_samples, 100);
    }

    #[test]
    fn test_auto_optimization_creation() {
        let config = AutoOptimizationConfig::default();
        let optimizer = AutoOptimization::new(config);
        assert!(optimizer.config.enabled);
    }

    #[test]
    fn test_search_parameters_default() {
        let params = SearchParameters::default();
        assert_eq!(params.bm25_k1, 1.2);
        assert_eq!(params.bm25_b, 0.75);
        assert_eq!(params.max_results, 100);
    }

    #[test]
    fn test_optimization_recommendation() {
        let recommendation = OptimizationRecommendation {
            optimization_type: OptimizationType::ParameterTuning,
            recommendation: "Adjust BM25 parameters".to_string(),
            expected_improvement: 0.1,
            confidence: 0.8,
            parameters: Some(HashMap::from([("bm25_k1".to_string(), "1.5".to_string())])),
            timestamp: chrono::Utc::now(),
        };

        assert_eq!(
            recommendation.optimization_type,
            OptimizationType::ParameterTuning
        );
        assert_eq!(recommendation.expected_improvement, 0.1);
    }

    #[test]
    fn test_performance_tracking() {
        let mut tracker = PerformanceTracker::new();

        let data = SearchPerformanceData {
            query: "test".to_string(),
            result_count: 10,
            response_time_ms: 100,
            relevance_scores: vec![0.8, 0.9, 0.7],
            click_through_rate: 0.3,
            timestamp: chrono::Utc::now(),
        };

        let result = tracker.add_data(data);
        assert!(result.is_ok());
        assert_eq!(tracker.performance_history.len(), 1);
    }

    #[test]
    fn test_ab_test_creation() {
        let test = ABTest::new(
            "test_1".to_string(),
            "Test BM25 parameters".to_string(),
            Some(HashMap::from([("bm25_k1".to_string(), "1.5".to_string())])),
        );

        assert_eq!(test.test_id, "test_1");
        assert_eq!(test.variants.len(), 2); // control + variant
    }

    #[test]
    fn test_optimization_stats() {
        let stats = OptimizationStats {
            total_optimizations: 10,
            parameter_optimizations: 5,
            algorithm_optimizations: 3,
            ab_tests_completed: 2,
            average_improvement: 0.08,
            last_optimization: chrono::Utc::now(),
        };

        assert_eq!(stats.total_optimizations, 10);
        assert_eq!(stats.average_improvement, 0.08);
    }
}
