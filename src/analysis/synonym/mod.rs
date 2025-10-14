//! Core synonym functionality shared across token filters and query expansion.
//!
//! This module provides the fundamental building blocks for synonym handling:
//! - Dictionary management
//! - Token graph construction
//! - Graph traversal and path extraction

pub mod dictionary;
pub mod graph_builder;
pub mod graph_traverser;

pub use dictionary::SynonymDictionary;
pub use graph_builder::SynonymGraphBuilder;
pub use graph_traverser::SynonymGraphTraverser;
