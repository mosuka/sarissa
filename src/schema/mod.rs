//! Schema module for Sarissa.
//!
//! This module provides schema definition and field type functionality,
//! similar to Whoosh's schema system.

pub mod field;
#[allow(clippy::module_inception)]
pub mod schema;

// Re-export commonly used types
pub use field::*;
pub use schema::*;

// Re-export field types for convenience
pub use field::{
    BooleanField, DateTimeField, FieldDefinition, FieldType, GeoField, IdField, KeywordField,
    NumericField, NumericType, StoredField, TextField,
};
