//! Lexical indexing module for building and maintaining lexical indexes.
//!
//! This module handles all lexical index construction, maintenance, and optimization:
//! - Building inverted indexes
//! - Document indexing and analysis
//! - Segment management and merging
//! - Index optimization and maintenance
//!
//! # Module Structure
//!
//! - `config`: Index configuration
//! - `factory`: Index factory for creating and opening indexes
//! - `traits`: Index trait definitions
//! - `inverted`: Inverted index implementation
//! - `segment`: Segment management
//! - `maintenance`: Index maintenance operations

pub mod config;
pub mod factory;
pub mod traits;

pub mod inverted;
pub mod segment;
pub mod maintenance;
