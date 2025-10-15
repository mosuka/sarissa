//! Storage abstraction layer for Sage.
//!
//! This module provides a pluggable storage system similar to Whoosh's storage architecture.
//! It supports different storage backends like file system, memory, and potentially remote storage.

pub mod column;
pub mod file;
pub mod memory;
pub mod mmap;
pub mod structured;
pub mod traits;

// Re-export commonly used types
pub use column::*;
pub use file::*;
pub use memory::*;
pub use mmap::*;
pub use structured::*;
pub use traits::*;
