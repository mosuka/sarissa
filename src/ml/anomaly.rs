//! Anomaly detection system for search quality monitoring.
//!
//! This module provides anomaly detection capabilities for:
//! - Search pattern anomalies
//! - Result quality degradation
//! - Performance anomalies
//! - User behavior anomalies

use crate::error::Result;
use crate::ml::{MLError, MLContext, SearchHistoryItem, FeedbackSignal};
use crate::query::SearchResults;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Configuration for anomaly detection system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Enable anomaly detection.
    pub enabled: bool,
    /// Window size for moving statistics (in minutes).
    pub window_size_minutes: u64,
    /// Number of standard deviations for anomaly threshold.
    pub anomaly_threshold_std: f64,
    /// Minimum samples required for anomaly detection.
    pub min_samples: usize,
    /// Enable search pattern anomaly detection.
    pub enable_pattern_detection: bool,
    /// Enable result quality anomaly detection.
    pub enable_quality_detection: bool,
    /// Enable performance anomaly detection.
    pub enable_performance_detection: bool,
    /// Enable user behavior anomaly detection.
    pub enable_behavior_detection: bool,
    /// Alert threshold for high-severity anomalies.
    pub alert_threshold: f64,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_size_minutes: 60, // 1 hour window
            anomaly_threshold_std: 2.0, // 2 standard deviations
            min_samples: 10,
            enable_pattern_detection: true,
            enable_quality_detection: true,
            enable_performance_detection: true,
            enable_behavior_detection: true,
            alert_threshold: 0.8,
        }
    }
}

/// Anomaly detection system.
pub struct AnomalyDetection {
    /// Configuration.
    config: AnomalyDetectionConfig,
    /// Search pattern detector.
    pattern_detector: SearchPatternDetector,
    /// Result quality detector.
    quality_detector: ResultQualityDetector,
    /// Performance detector.
    performance_detector: PerformanceDetector,
    /// User behavior detector.
    behavior_detector: UserBehaviorDetector,
    /// Anomaly history for tracking.
    anomaly_history: VecDeque<AnomalyEvent>,
}

impl AnomalyDetection {
    /// Create a new anomaly detection system.
    pub fn new(config: AnomalyDetectionConfig) -> Self {
        Self {
            config: config.clone(),
            pattern_detector: SearchPatternDetector::new(&config),
            quality_detector: ResultQualityDetector::new(&config),
            performance_detector: PerformanceDetector::new(&config),
            behavior_detector: UserBehaviorDetector::new(&config),
            anomaly_history: VecDeque::new(),
        }
    }

    /// Detect anomalies in search request and results.
    pub fn detect_anomalies(
        &mut self,
        query: &str,
        results: &SearchResults,
        context: &MLContext,
        response_time_ms: u64,
    ) -> Result<Vec<AnomalyEvent>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let mut anomalies = Vec::new();
        let timestamp = chrono::Utc::now();

        // Detect search pattern anomalies
        if self.config.enable_pattern_detection {
            if let Some(pattern_anomaly) = self.pattern_detector.detect_anomaly(query, &timestamp)? {
                anomalies.push(pattern_anomaly);
            }
        }

        // Detect result quality anomalies
        if self.config.enable_quality_detection {
            if let Some(quality_anomaly) = self.quality_detector.detect_anomaly(query, results, &timestamp)? {
                anomalies.push(quality_anomaly);
            }
        }

        // Detect performance anomalies
        if self.config.enable_performance_detection {
            if let Some(perf_anomaly) = self.performance_detector.detect_anomaly(response_time_ms, &timestamp)? {
                anomalies.push(perf_anomaly);
            }
        }

        // Detect user behavior anomalies
        if self.config.enable_behavior_detection {
            if let Some(behavior_anomaly) = self.behavior_detector.detect_anomaly(context, &timestamp)? {
                anomalies.push(behavior_anomaly);
            }
        }

        // Store anomalies in history
        for anomaly in &anomalies {
            self.anomaly_history.push_back(anomaly.clone());
            
            // Limit history size
            while self.anomaly_history.len() > 1000 {
                self.anomaly_history.pop_front();
            }
        }

        Ok(anomalies)
    }

    /// Process user feedback for anomaly detection.
    pub fn process_feedback(&mut self, feedback: &FeedbackSignal) -> Result<()> {
        self.quality_detector.add_feedback(feedback)?;
        self.behavior_detector.add_feedback(feedback)?;
        Ok(())
    }

    /// Get recent anomaly statistics.
    pub fn get_anomaly_stats(&self, last_hours: u64) -> AnomalyStats {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::hours(last_hours as i64);
        
        let recent_anomalies: Vec<_> = self.anomaly_history.iter()
            .filter(|a| a.timestamp >= cutoff_time)
            .collect();

        let mut stats = AnomalyStats {
            total_anomalies: recent_anomalies.len(),
            pattern_anomalies: 0,
            quality_anomalies: 0,
            performance_anomalies: 0,
            behavior_anomalies: 0,
            high_severity_count: 0,
            average_severity: 0.0,
        };

        for anomaly in &recent_anomalies {
            match anomaly.anomaly_type {
                AnomalyType::SearchPattern => stats.pattern_anomalies += 1,
                AnomalyType::ResultQuality => stats.quality_anomalies += 1,
                AnomalyType::Performance => stats.performance_anomalies += 1,
                AnomalyType::UserBehavior => stats.behavior_anomalies += 1,
            }

            if anomaly.severity >= self.config.alert_threshold {
                stats.high_severity_count += 1;
            }

            stats.average_severity += anomaly.severity;
        }

        if !recent_anomalies.is_empty() {
            stats.average_severity /= recent_anomalies.len() as f64;
        }

        stats
    }

    /// Check if system health is degraded.
    pub fn is_system_healthy(&self) -> bool {
        let stats = self.get_anomaly_stats(1); // Last hour
        
        // Simple health check based on recent high-severity anomalies
        stats.high_severity_count < 5 && stats.average_severity < self.config.alert_threshold
    }
}

/// Search pattern anomaly detector.
#[derive(Debug)]
struct SearchPatternDetector {
    query_frequency: HashMap<String, VecDeque<chrono::DateTime<chrono::Utc>>>,
    query_stats: MovingStatistics,
    config: AnomalyDetectionConfig,
}

impl SearchPatternDetector {
    fn new(config: &AnomalyDetectionConfig) -> Self {
        Self {
            query_frequency: HashMap::new(),
            query_stats: MovingStatistics::new(config.window_size_minutes),
            config: config.clone(),
        }
    }

    fn detect_anomaly(
        &mut self,
        query: &str,
        timestamp: &chrono::DateTime<chrono::Utc>,
    ) -> Result<Option<AnomalyEvent>> {
        // Track query frequency
        let query_times = self.query_frequency
            .entry(query.to_string())
            .or_insert_with(VecDeque::new);

        query_times.push_back(*timestamp);

        // Remove old entries
        let cutoff = *timestamp - chrono::Duration::minutes(self.config.window_size_minutes as i64);
        while query_times.front().map(|&t| t < cutoff).unwrap_or(false) {
            query_times.pop_front();
        }

        // Check for suspicious frequency patterns
        if query_times.len() >= 20 { // More than 20 identical queries in window
            return Ok(Some(AnomalyEvent {
                anomaly_type: AnomalyType::SearchPattern,
                severity: 0.8,
                message: format!("High frequency of identical query: '{}'", query),
                details: HashMap::from([
                    ("query".to_string(), query.to_string()),
                    ("frequency".to_string(), query_times.len().to_string()),
                ]),
                timestamp: *timestamp,
            }));
        }

        // Update general query statistics
        self.query_stats.add_value(1.0, *timestamp);

        // Check for overall query volume anomalies
        if self.query_stats.sample_count() >= self.config.min_samples {
            let current_rate = self.query_stats.current_value();
            let mean = self.query_stats.mean();
            let std_dev = self.query_stats.std_deviation();

            if std_dev > 0.0 {
                let z_score = (current_rate - mean) / std_dev;
                
                if z_score.abs() > self.config.anomaly_threshold_std {
                    return Ok(Some(AnomalyEvent {
                        anomaly_type: AnomalyType::SearchPattern,
                        severity: (z_score.abs() / self.config.anomaly_threshold_std).min(1.0),
                        message: "Unusual query volume detected".to_string(),
                        details: HashMap::from([
                            ("z_score".to_string(), z_score.to_string()),
                            ("current_rate".to_string(), current_rate.to_string()),
                            ("mean".to_string(), mean.to_string()),
                        ]),
                        timestamp: *timestamp,
                    }));
                }
            }
        }

        Ok(None)
    }
}

/// Result quality anomaly detector.
#[derive(Debug)]
struct ResultQualityDetector {
    click_through_rates: MovingStatistics,
    result_counts: MovingStatistics,
    relevance_scores: MovingStatistics,
    config: AnomalyDetectionConfig,
}

impl ResultQualityDetector {
    fn new(config: &AnomalyDetectionConfig) -> Self {
        Self {
            click_through_rates: MovingStatistics::new(config.window_size_minutes),
            result_counts: MovingStatistics::new(config.window_size_minutes),
            relevance_scores: MovingStatistics::new(config.window_size_minutes),
            config: config.clone(),
        }
    }

    fn detect_anomaly(
        &mut self,
        query: &str,
        results: &SearchResults,
        timestamp: &chrono::DateTime<chrono::Utc>,
    ) -> Result<Option<AnomalyEvent>> {
        // Update statistics
        self.result_counts.add_value(results.hits.len() as f64, *timestamp);

        // Check for unusual result count
        if self.result_counts.sample_count() >= self.config.min_samples {
            let current_count = results.hits.len() as f64;
            let mean = self.result_counts.mean();
            let std_dev = self.result_counts.std_deviation();

            if std_dev > 0.0 {
                let z_score = (current_count - mean) / std_dev;
                
                if z_score < -self.config.anomaly_threshold_std {
                    return Ok(Some(AnomalyEvent {
                        anomaly_type: AnomalyType::ResultQuality,
                        severity: (z_score.abs() / self.config.anomaly_threshold_std).min(1.0),
                        message: "Unusually low search result count".to_string(),
                        details: HashMap::from([
                            ("query".to_string(), query.to_string()),
                            ("result_count".to_string(), current_count.to_string()),
                            ("expected_mean".to_string(), mean.to_string()),
                        ]),
                        timestamp: *timestamp,
                    }));
                }
            }
        }

        Ok(None)
    }

    fn add_feedback(&mut self, feedback: &FeedbackSignal) -> Result<()> {
        self.relevance_scores.add_value(feedback.relevance_score, feedback.timestamp);
        Ok(())
    }
}

/// Performance anomaly detector.
#[derive(Debug)]
struct PerformanceDetector {
    response_times: MovingStatistics,
    config: AnomalyDetectionConfig,
}

impl PerformanceDetector {
    fn new(config: &AnomalyDetectionConfig) -> Self {
        Self {
            response_times: MovingStatistics::new(config.window_size_minutes),
            config: config.clone(),
        }
    }

    fn detect_anomaly(
        &mut self,
        response_time_ms: u64,
        timestamp: &chrono::DateTime<chrono::Utc>,
    ) -> Result<Option<AnomalyEvent>> {
        let response_time = response_time_ms as f64;
        self.response_times.add_value(response_time, *timestamp);

        if self.response_times.sample_count() >= self.config.min_samples {
            let mean = self.response_times.mean();
            let std_dev = self.response_times.std_deviation();

            if std_dev > 0.0 {
                let z_score = (response_time - mean) / std_dev;
                
                if z_score > self.config.anomaly_threshold_std {
                    return Ok(Some(AnomalyEvent {
                        anomaly_type: AnomalyType::Performance,
                        severity: (z_score / self.config.anomaly_threshold_std).min(1.0),
                        message: "Unusual response time detected".to_string(),
                        details: HashMap::from([
                            ("response_time_ms".to_string(), response_time_ms.to_string()),
                            ("expected_mean".to_string(), mean.to_string()),
                            ("z_score".to_string(), z_score.to_string()),
                        ]),
                        timestamp: *timestamp,
                    }));
                }
            }
        }

        Ok(None)
    }
}

/// User behavior anomaly detector.
#[derive(Debug)]
struct UserBehaviorDetector {
    session_lengths: MovingStatistics,
    query_patterns: HashMap<String, usize>,
    config: AnomalyDetectionConfig,
}

impl UserBehaviorDetector {
    fn new(config: &AnomalyDetectionConfig) -> Self {
        Self {
            session_lengths: MovingStatistics::new(config.window_size_minutes),
            query_patterns: HashMap::new(),
            config: config.clone(),
        }
    }

    fn detect_anomaly(
        &mut self,
        context: &MLContext,
        timestamp: &chrono::DateTime<chrono::Utc>,
    ) -> Result<Option<AnomalyEvent>> {
        // Check for suspicious query patterns
        if context.search_history.len() > 10 {
            let recent_queries: Vec<_> = context.search_history.iter()
                .rev()
                .take(10)
                .map(|item| &item.query)
                .collect();

            // Check for repetitive patterns
            let unique_queries: std::collections::HashSet<_> = recent_queries.iter().collect();
            
            if unique_queries.len() <= 2 && recent_queries.len() >= 10 {
                return Ok(Some(AnomalyEvent {
                    anomaly_type: AnomalyType::UserBehavior,
                    severity: 0.6,
                    message: "Repetitive query pattern detected".to_string(),
                    details: HashMap::from([
                        ("unique_queries".to_string(), unique_queries.len().to_string()),
                        ("total_queries".to_string(), recent_queries.len().to_string()),
                    ]),
                    timestamp: *timestamp,
                }));
            }
        }

        Ok(None)
    }

    fn add_feedback(&mut self, feedback: &FeedbackSignal) -> Result<()> {
        // Track feedback patterns for anomaly detection
        *self.query_patterns.entry(feedback.query.clone()).or_insert(0) += 1;
        Ok(())
    }
}

/// Moving statistics calculator for anomaly detection.
#[derive(Debug)]
struct MovingStatistics {
    values: VecDeque<(f64, chrono::DateTime<chrono::Utc>)>,
    window_size_minutes: u64,
    sum: f64,
    sum_squared: f64,
}

impl MovingStatistics {
    fn new(window_size_minutes: u64) -> Self {
        Self {
            values: VecDeque::new(),
            window_size_minutes,
            sum: 0.0,
            sum_squared: 0.0,
        }
    }

    fn add_value(&mut self, value: f64, timestamp: chrono::DateTime<chrono::Utc>) {
        // Remove old values
        let cutoff = timestamp - chrono::Duration::minutes(self.window_size_minutes as i64);
        while self.values.front().map(|(_, t)| *t < cutoff).unwrap_or(false) {
            let (old_value, _) = self.values.pop_front().unwrap();
            self.sum -= old_value;
            self.sum_squared -= old_value * old_value;
        }

        // Add new value
        self.values.push_back((value, timestamp));
        self.sum += value;
        self.sum_squared += value * value;
    }

    fn sample_count(&self) -> usize {
        self.values.len()
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    fn std_deviation(&self) -> f64 {
        let n = self.values.len();
        if n <= 1 {
            return 0.0;
        }

        let mean = self.mean();
        let variance = (self.sum_squared / n as f64) - (mean * mean);
        variance.max(0.0).sqrt()
    }

    fn current_value(&self) -> f64 {
        self.values.back().map(|(v, _)| *v).unwrap_or(0.0)
    }
}

/// Anomaly event data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    /// Type of anomaly detected.
    pub anomaly_type: AnomalyType,
    /// Severity score (0.0 - 1.0).
    pub severity: f64,
    /// Human-readable message.
    pub message: String,
    /// Additional details about the anomaly.
    pub details: HashMap<String, String>,
    /// When the anomaly was detected.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of anomalies that can be detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Search pattern anomaly.
    SearchPattern,
    /// Result quality anomaly.
    ResultQuality,
    /// Performance anomaly.
    Performance,
    /// User behavior anomaly.
    UserBehavior,
}

/// Anomaly detection statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyStats {
    /// Total number of anomalies detected.
    pub total_anomalies: usize,
    /// Number of search pattern anomalies.
    pub pattern_anomalies: usize,
    /// Number of result quality anomalies.
    pub quality_anomalies: usize,
    /// Number of performance anomalies.
    pub performance_anomalies: usize,
    /// Number of user behavior anomalies.
    pub behavior_anomalies: usize,
    /// Number of high-severity anomalies.
    pub high_severity_count: usize,
    /// Average severity of detected anomalies.
    pub average_severity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::FeedbackType;
    use crate::query::SearchHit;

    #[test]
    fn test_anomaly_detection_config_default() {
        let config = AnomalyDetectionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.window_size_minutes, 60);
        assert_eq!(config.anomaly_threshold_std, 2.0);
    }

    #[test]
    fn test_anomaly_detection_creation() {
        let config = AnomalyDetectionConfig::default();
        let detector = AnomalyDetection::new(config);
        assert!(detector.config.enabled);
    }

    #[test]
    fn test_moving_statistics() {
        let mut stats = MovingStatistics::new(60);
        let timestamp = chrono::Utc::now();
        
        stats.add_value(10.0, timestamp);
        stats.add_value(20.0, timestamp);
        stats.add_value(30.0, timestamp);
        
        assert_eq!(stats.sample_count(), 3);
        assert_eq!(stats.mean(), 20.0);
        assert!(stats.std_deviation() > 0.0);
    }

    #[test]
    fn test_anomaly_event_creation() {
        let event = AnomalyEvent {
            anomaly_type: AnomalyType::Performance,
            severity: 0.8,
            message: "High response time".to_string(),
            details: HashMap::from([
                ("response_time".to_string(), "1000ms".to_string()),
            ]),
            timestamp: chrono::Utc::now(),
        };
        
        assert_eq!(event.anomaly_type, AnomalyType::Performance);
        assert_eq!(event.severity, 0.8);
    }

    #[test]
    fn test_anomaly_detection_with_feedback() {
        let mut detector = AnomalyDetection::new(AnomalyDetectionConfig::default());
        
        let feedback = FeedbackSignal {
            query: "test query".to_string(),
            document_id: "doc1".to_string(),
            feedback_type: FeedbackType::Click,
            relevance_score: 0.5,
            timestamp: chrono::Utc::now(),
        };
        
        let result = detector.process_feedback(&feedback);
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_health_check() {
        let detector = AnomalyDetection::new(AnomalyDetectionConfig::default());
        
        // Should be healthy with no anomalies
        assert!(detector.is_system_healthy());
    }

    #[test]
    fn test_anomaly_stats() {
        let detector = AnomalyDetection::new(AnomalyDetectionConfig::default());
        let stats = detector.get_anomaly_stats(24);
        
        assert_eq!(stats.total_anomalies, 0);
        assert_eq!(stats.average_severity, 0.0);
    }
}