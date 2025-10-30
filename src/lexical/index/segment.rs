//! Segment management for lexical indexes.
//!
//! This module handles segment operations:
//! - Segment manager for coordinating segments
//! - Merge engine for combining segments
//! - Merge policy for determining when to merge

pub mod manager;
pub mod merge_engine;
pub mod merge_policy;
