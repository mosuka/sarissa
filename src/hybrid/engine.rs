//! Hybrid search engine implementation.
//!
//! This module provides a placeholder for hybrid search engine functionality.
//! The full implementation will combine lexical and vector search with configurable
//! fusion algorithms.

use crate::error::Result;
use crate::hybrid::config::HybridSearchConfig;
use crate::hybrid::search::merger::ResultMerger;

/// Hybrid search engine that combines keyword and vector search.
///
/// Note: This is currently a minimal implementation. Full hybrid search functionality
/// requires integration with both lexical and vector indexes.
pub struct HybridSearchEngine {
    /// Configuration for hybrid search.
    config: HybridSearchConfig,
    /// Result merger for combining search results.
    merger: ResultMerger,
}

impl HybridSearchEngine {
    /// Create a new hybrid search engine.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for hybrid search behavior
    ///
    /// # Returns
    ///
    /// A new `HybridSearchEngine` instance
    pub fn new(config: HybridSearchConfig) -> Result<Self> {
        let merger = ResultMerger::new(config.clone());

        Ok(Self { config, merger })
    }

    /// Get the search configuration.
    ///
    /// # Returns
    ///
    /// A reference to the hybrid search configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::hybrid::config::HybridSearchConfig;
    /// use yatagarasu::hybrid::engine::HybridSearchEngine;
    ///
    /// # fn example() -> yatagarasu::error::Result<()> {
    /// let config = HybridSearchConfig::default();
    /// let engine = HybridSearchEngine::new(config)?;
    ///
    /// let config_ref = engine.config();
    /// println!("Keyword weight: {}", config_ref.keyword_weight);
    /// println!("Vector weight: {}", config_ref.vector_weight);
    /// # Ok(())
    /// # }
    /// ```
    pub fn config(&self) -> &HybridSearchConfig {
        &self.config
    }

    /// Get a reference to the result merger.
    ///
    /// The result merger is responsible for combining keyword and vector
    /// search results according to the configured strategy.
    ///
    /// # Returns
    ///
    /// A reference to the internal `ResultMerger`
    pub fn merger(&self) -> &ResultMerger {
        &self.merger
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_search_engine_creation() {
        let config = HybridSearchConfig::default();
        let engine = HybridSearchEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_engine_config_access() {
        let config = HybridSearchConfig::default();
        let engine = HybridSearchEngine::new(config.clone()).unwrap();
        assert_eq!(engine.config().keyword_weight, config.keyword_weight);
    }
}
