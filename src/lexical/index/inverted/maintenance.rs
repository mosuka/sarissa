//! Index maintenance operations for inverted indexes.
//!
//! This module provides maintenance functionality for inverted indexes:
//! - Background tasks for async operations
//! - Deletion management
//! - Optimization strategies
//! - Transaction support

pub mod background_tasks;
pub mod deletion;
pub mod optimization;
pub mod transaction;
