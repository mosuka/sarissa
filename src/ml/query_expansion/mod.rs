//! Query expansion system for improving search recall.
//!
//! This module provides automatic query expansion using various techniques:
//! - Synonym expansion
//! - Word embeddings for semantic expansion
//! - Statistical co-occurrence expansion
//!
//! # Architecture
//!
//! The query expansion system uses a Strategy pattern with the following components:
//! - `QueryExpander` trait: Defines the interface for expansion strategies
//! - `SynonymQueryExpander`: Dictionary-based synonym expansion
//! - `SemanticQueryExpander`: Embedding-based semantic expansion
//! - `StatisticalQueryExpander`: Co-occurrence based statistical expansion
//! - `QueryExpansionBuilder`: Fluent API for building expansion pipelines
//!
//! # Example
//!
//! ```rust,no_run
//! use sarissa::ml::query_expansion::QueryExpansionBuilder;
//! use sarissa::analysis::StandardAnalyzer;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let analyzer = Arc::new(StandardAnalyzer::new()?);
//!
//! let expansion = QueryExpansionBuilder::new(analyzer)
//!     .with_synonyms(Some("synonyms.json"), 0.5)?
//!     .with_statistical(0.3)
//!     .max_expansions(10)
//!     .build()?;
//! # Ok(())
//! # }
//! ```

mod builder;
mod core;
mod r#trait;
mod types;

// Expander implementations
mod semantic;
mod statistical;
mod synonym;

// Public exports
pub use builder::QueryExpansionBuilder;
pub use core::QueryExpansion;
pub use r#trait::QueryExpander;
pub use semantic::{SemanticQueryExpander, WordEmbeddings};
pub use statistical::{CoOccurrenceModel, StatisticalQueryExpander};
pub use synonym::{SynonymDictionary, SynonymQueryExpander};
pub use types::{ExpandedQuery, ExpandedQueryClause, ExpansionType, QueryIntent};
