//! Core synonym functionality shared across token filters and query expansion.
//!
//! This module provides the fundamental building blocks for synonym handling
//! in text analysis. Synonyms allow matching semantically equivalent terms
//! (e.g., "quick" and "fast") to improve search recall.
//!
//! # Components
//!
//! - [`dictionary`] - Synonym dictionary management and parsing
//! - [`graph_builder`] - Token graph construction with synonym paths
//! - [`graph_traverser`] - Graph traversal to extract all possible paths
//!
//! # Token Graph Concept
//!
//! When synonyms are inserted into a token stream, they create a graph structure
//! where multiple tokens can occupy the same position. For example:
//!
//! ```text
//! Input: "quick brown fox"
//! With synonym: quick → fast
//!
//! Token Graph:
//!   Position 0: "quick" ──┐
//!                         ├──> Position 1: "brown" -> Position 2: "fox"
//!   Position 0: "fast"  ──┘
//! ```
//!
//! # Examples
//!
//! ```
//! use yatagarasu::analysis::synonym::dictionary::SynonymDictionary;
//!
//! let mut dict = SynonymDictionary::new();
//! dict.add_synonym("quick", vec!["fast", "rapid"]);
//! dict.add_synonym("big", vec!["large", "huge"]);
//!
//! assert!(dict.get_synonyms("quick").is_some());
//! assert_eq!(dict.len(), 2);
//! ```

pub mod dictionary;
pub mod graph_builder;
pub mod graph_traverser;
