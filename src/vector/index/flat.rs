//! Flat vector index implementation.
//!
//! This module provides a simple flat (brute-force) vector index that stores
//! all vectors in memory and performs exact nearest neighbor search through
//! exhaustive comparison.

pub mod reader;
pub mod writer;
