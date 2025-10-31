//! Segment management for inverted indexes.
//!
//! This module handles segment operations for inverted indexes:
//! - Segment manager for coordinating segments
//! - Merge engine for combining segments
//! - Merge policy for determining when to merge
//! - Type definitions for segment metadata

pub mod manager;
pub mod merge_engine;
pub mod merge_policy;
pub mod types;
