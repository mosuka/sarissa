//! Column storage for fast field access.
//!
//! This module provides columnar storage capabilities for efficient
//! faceting, sorting, and aggregation operations.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, RwLock};

use anyhow;
use byteorder::{BigEndian, ByteOrder};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::storage::Storage;

/// Column value types supported by the column storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColumnValue {
    /// String value
    String(String),
    /// 32-bit integer
    I32(i32),
    /// 64-bit integer
    I64(i64),
    /// 32-bit unsigned integer
    U32(u32),
    /// 64-bit unsigned integer
    U64(u64),
    /// 32-bit float
    F32(f32),
    /// 64-bit float
    F64(f64),
    /// Boolean value
    Bool(bool),
    /// DateTime as Unix timestamp (seconds since epoch)
    DateTime(i64),
    /// Null value
    Null,
}

impl Eq for ColumnValue {}

impl std::hash::Hash for ColumnValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ColumnValue::String(s) => {
                0u8.hash(state);
                s.hash(state);
            }
            ColumnValue::I32(v) => {
                1u8.hash(state);
                v.hash(state);
            }
            ColumnValue::I64(v) => {
                2u8.hash(state);
                v.hash(state);
            }
            ColumnValue::U32(v) => {
                3u8.hash(state);
                v.hash(state);
            }
            ColumnValue::U64(v) => {
                4u8.hash(state);
                v.hash(state);
            }
            ColumnValue::F32(v) => {
                5u8.hash(state);
                v.to_bits().hash(state);
            }
            ColumnValue::F64(v) => {
                6u8.hash(state);
                v.to_bits().hash(state);
            }
            ColumnValue::Bool(v) => {
                7u8.hash(state);
                v.hash(state);
            }
            ColumnValue::DateTime(v) => {
                8u8.hash(state);
                v.hash(state);
            }
            ColumnValue::Null => {
                255u8.hash(state);
            }
        }
    }
}

impl ColumnValue {
    /// Get the type name for this column value.
    pub fn type_name(&self) -> &'static str {
        match self {
            ColumnValue::String(_) => "string",
            ColumnValue::I32(_) => "i32",
            ColumnValue::I64(_) => "i64",
            ColumnValue::U32(_) => "u32",
            ColumnValue::U64(_) => "u64",
            ColumnValue::F32(_) => "f32",
            ColumnValue::F64(_) => "f64",
            ColumnValue::Bool(_) => "bool",
            ColumnValue::DateTime(_) => "datetime",
            ColumnValue::Null => "null",
        }
    }

    /// Check if this value can be compared with another value.
    pub fn is_comparable_with(&self, other: &ColumnValue) -> bool {
        match (self, other) {
            (ColumnValue::Null, _) | (_, ColumnValue::Null) => true,
            (ColumnValue::String(_), ColumnValue::String(_)) => true,
            (ColumnValue::I32(_), ColumnValue::I32(_)) => true,
            (ColumnValue::I64(_), ColumnValue::I64(_)) => true,
            (ColumnValue::U32(_), ColumnValue::U32(_)) => true,
            (ColumnValue::U64(_), ColumnValue::U64(_)) => true,
            (ColumnValue::F32(_), ColumnValue::F32(_)) => true,
            (ColumnValue::F64(_), ColumnValue::F64(_)) => true,
            (ColumnValue::Bool(_), ColumnValue::Bool(_)) => true,
            (ColumnValue::DateTime(_), ColumnValue::DateTime(_)) => true,
            // Allow numeric cross-comparisons
            (ColumnValue::I32(_), ColumnValue::I64(_))
            | (ColumnValue::I64(_), ColumnValue::I32(_))
            | (ColumnValue::U32(_), ColumnValue::U64(_))
            | (ColumnValue::U64(_), ColumnValue::U32(_))
            | (ColumnValue::F32(_), ColumnValue::F64(_))
            | (ColumnValue::F64(_), ColumnValue::F32(_)) => true,
            _ => false,
        }
    }

    /// Serialize this value to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();

        match self {
            ColumnValue::String(s) => {
                bytes.push(0); // Type marker
                let str_bytes = s.as_bytes();
                bytes.extend_from_slice(&(str_bytes.len() as u32).to_be_bytes());
                bytes.extend_from_slice(str_bytes);
            }
            ColumnValue::I32(v) => {
                bytes.push(1);
                bytes.extend_from_slice(&v.to_be_bytes());
            }
            ColumnValue::I64(v) => {
                bytes.push(2);
                bytes.extend_from_slice(&v.to_be_bytes());
            }
            ColumnValue::U32(v) => {
                bytes.push(3);
                bytes.extend_from_slice(&v.to_be_bytes());
            }
            ColumnValue::U64(v) => {
                bytes.push(4);
                bytes.extend_from_slice(&v.to_be_bytes());
            }
            ColumnValue::F32(v) => {
                bytes.push(5);
                bytes.extend_from_slice(&v.to_be_bytes());
            }
            ColumnValue::F64(v) => {
                bytes.push(6);
                bytes.extend_from_slice(&v.to_be_bytes());
            }
            ColumnValue::Bool(v) => {
                bytes.push(7);
                bytes.push(if *v { 1 } else { 0 });
            }
            ColumnValue::DateTime(v) => {
                bytes.push(8);
                bytes.extend_from_slice(&v.to_be_bytes());
            }
            ColumnValue::Null => {
                bytes.push(255); // Null marker
            }
        }

        Ok(bytes)
    }

    /// Deserialize a value from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Ok(ColumnValue::Null);
        }

        let type_marker = bytes[0];
        match type_marker {
            0 => {
                // String
                if bytes.len() < 5 {
                    return Err(anyhow::anyhow!("Invalid string value bytes").into());
                }
                let len = BigEndian::read_u32(&bytes[1..5]) as usize;
                if bytes.len() < 5 + len {
                    return Err(anyhow::anyhow!("Truncated string value").into());
                }
                let s = String::from_utf8(bytes[5..5 + len].to_vec())
                    .map_err(|e| anyhow::anyhow!("UTF8 conversion error: {e}"))?;
                Ok(ColumnValue::String(s))
            }
            1 => {
                if bytes.len() < 5 {
                    return Err(anyhow::anyhow!("Invalid i32 value bytes").into());
                }
                let v = BigEndian::read_i32(&bytes[1..5]);
                Ok(ColumnValue::I32(v))
            }
            2 => {
                if bytes.len() < 9 {
                    return Err(anyhow::anyhow!("Invalid i64 value bytes").into());
                }
                let v = BigEndian::read_i64(&bytes[1..9]);
                Ok(ColumnValue::I64(v))
            }
            3 => {
                if bytes.len() < 5 {
                    return Err(anyhow::anyhow!("Invalid u32 value bytes").into());
                }
                let v = BigEndian::read_u32(&bytes[1..5]);
                Ok(ColumnValue::U32(v))
            }
            4 => {
                if bytes.len() < 9 {
                    return Err(anyhow::anyhow!("Invalid u64 value bytes").into());
                }
                let v = BigEndian::read_u64(&bytes[1..9]);
                Ok(ColumnValue::U64(v))
            }
            5 => {
                if bytes.len() < 5 {
                    return Err(anyhow::anyhow!("Invalid f32 value bytes").into());
                }
                let v = BigEndian::read_f32(&bytes[1..5]);
                Ok(ColumnValue::F32(v))
            }
            6 => {
                if bytes.len() < 9 {
                    return Err(anyhow::anyhow!("Invalid f64 value bytes").into());
                }
                let v = BigEndian::read_f64(&bytes[1..9]);
                Ok(ColumnValue::F64(v))
            }
            7 => {
                if bytes.len() < 2 {
                    return Err(anyhow::anyhow!("Invalid bool value bytes").into());
                }
                let v = bytes[1] != 0;
                Ok(ColumnValue::Bool(v))
            }
            8 => {
                if bytes.len() < 9 {
                    return Err(anyhow::anyhow!("Invalid datetime value bytes").into());
                }
                let v = BigEndian::read_i64(&bytes[1..9]);
                Ok(ColumnValue::DateTime(v))
            }
            255 => Ok(ColumnValue::Null),
            _ => Err(anyhow::anyhow!("Unknown column value type: {type_marker}").into()),
        }
    }
}

impl PartialOrd for ColumnValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match (self, other) {
            (ColumnValue::Null, ColumnValue::Null) => Some(Ordering::Equal),
            (ColumnValue::Null, _) => Some(Ordering::Less),
            (_, ColumnValue::Null) => Some(Ordering::Greater),
            (ColumnValue::String(a), ColumnValue::String(b)) => a.partial_cmp(b),
            (ColumnValue::I32(a), ColumnValue::I32(b)) => a.partial_cmp(b),
            (ColumnValue::I64(a), ColumnValue::I64(b)) => a.partial_cmp(b),
            (ColumnValue::U32(a), ColumnValue::U32(b)) => a.partial_cmp(b),
            (ColumnValue::U64(a), ColumnValue::U64(b)) => a.partial_cmp(b),
            (ColumnValue::F32(a), ColumnValue::F32(b)) => a.partial_cmp(b),
            (ColumnValue::F64(a), ColumnValue::F64(b)) => a.partial_cmp(b),
            (ColumnValue::Bool(a), ColumnValue::Bool(b)) => a.partial_cmp(b),
            (ColumnValue::DateTime(a), ColumnValue::DateTime(b)) => a.partial_cmp(b),
            // Cross-type numeric comparisons
            (ColumnValue::I32(a), ColumnValue::I64(b)) => (*a as i64).partial_cmp(b),
            (ColumnValue::I64(a), ColumnValue::I32(b)) => a.partial_cmp(&(*b as i64)),
            (ColumnValue::U32(a), ColumnValue::U64(b)) => (*a as u64).partial_cmp(b),
            (ColumnValue::U64(a), ColumnValue::U32(b)) => a.partial_cmp(&(*b as u64)),
            (ColumnValue::F32(a), ColumnValue::F64(b)) => (*a as f64).partial_cmp(b),
            (ColumnValue::F64(a), ColumnValue::F32(b)) => a.partial_cmp(&(*b as f64)),
            _ => None,
        }
    }
}

/// A column stores values for a specific field across documents.
#[derive(Debug)]
pub struct Column {
    /// Field name this column represents
    field_name: String,
    /// Values indexed by document ID
    values: RwLock<HashMap<u32, ColumnValue>>,
    /// Next document ID to assign
    next_doc_id: RwLock<u32>,
}

impl Column {
    /// Create a new column for the given field.
    pub fn new(field_name: String) -> Self {
        Column {
            field_name,
            values: RwLock::new(HashMap::new()),
            next_doc_id: RwLock::new(0),
        }
    }

    /// Get the field name for this column.
    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Add a value for a document.
    pub fn add_value(&self, doc_id: u32, value: ColumnValue) -> Result<()> {
        let mut values = self.values.write().unwrap();
        values.insert(doc_id, value);

        let mut next_id = self.next_doc_id.write().unwrap();
        if doc_id >= *next_id {
            *next_id = doc_id + 1;
        }

        Ok(())
    }

    /// Get a value for a document.
    pub fn get_value(&self, doc_id: u32) -> Option<ColumnValue> {
        let values = self.values.read().unwrap();
        values.get(&doc_id).cloned()
    }

    /// Get all values in document ID order.
    pub fn get_all_values(&self) -> Vec<(u32, ColumnValue)> {
        let values = self.values.read().unwrap();
        let mut result: Vec<_> = values
            .iter()
            .map(|(&id, value)| (id, value.clone()))
            .collect();
        result.sort_by_key(|(id, _)| *id);
        result
    }

    /// Get values for a range of documents.
    pub fn get_values_in_range(&self, start_doc: u32, end_doc: u32) -> Vec<(u32, ColumnValue)> {
        let values = self.values.read().unwrap();
        let mut result = Vec::new();

        for doc_id in start_doc..=end_doc {
            if let Some(value) = values.get(&doc_id) {
                result.push((doc_id, value.clone()));
            }
        }

        result
    }

    /// Get document count.
    pub fn doc_count(&self) -> u32 {
        let values = self.values.read().unwrap();
        values.len() as u32
    }

    /// Get unique values and their document frequencies.
    pub fn get_value_frequencies(&self) -> HashMap<ColumnValue, u32> {
        let values = self.values.read().unwrap();
        let mut frequencies = HashMap::new();

        for value in values.values() {
            *frequencies.entry(value.clone()).or_insert(0) += 1;
        }

        frequencies
    }

    /// Find documents with specific value.
    pub fn find_documents_with_value(&self, target_value: &ColumnValue) -> Vec<u32> {
        let values = self.values.read().unwrap();
        let mut result = Vec::new();

        for (&doc_id, value) in values.iter() {
            if value == target_value {
                result.push(doc_id);
            }
        }

        result.sort();
        result
    }

    /// Find documents within value range.
    pub fn find_documents_in_range(
        &self,
        min_value: &ColumnValue,
        max_value: &ColumnValue,
    ) -> Vec<u32> {
        let values = self.values.read().unwrap();
        let mut result = Vec::new();

        for (&doc_id, value) in values.iter() {
            if value >= min_value && value <= max_value {
                result.push(doc_id);
            }
        }

        result.sort();
        result
    }
}

/// Column storage manages multiple columns for fast field access.
#[derive(Debug)]
pub struct ColumnStorage {
    /// Storage backend
    storage: Arc<dyn Storage>,
    /// Columns indexed by field name
    columns: RwLock<HashMap<String, Arc<Column>>>,
}

impl ColumnStorage {
    /// Create a new column storage.
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        ColumnStorage {
            storage,
            columns: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create a column for a field.
    pub fn get_column(&self, field_name: &str) -> Arc<Column> {
        let mut columns = self.columns.write().unwrap();

        if let Some(column) = columns.get(field_name) {
            return Arc::clone(column);
        }

        let column = Arc::new(Column::new(field_name.to_string()));
        columns.insert(field_name.to_string(), Arc::clone(&column));
        column
    }

    /// Add a value to a column.
    pub fn add_value(&self, field_name: &str, doc_id: u32, value: ColumnValue) -> Result<()> {
        let column = self.get_column(field_name);
        column.add_value(doc_id, value)
    }

    /// Get a value from a column.
    pub fn get_value(&self, field_name: &str, doc_id: u32) -> Option<ColumnValue> {
        let columns = self.columns.read().unwrap();
        if let Some(column) = columns.get(field_name) {
            column.get_value(doc_id)
        } else {
            None
        }
    }

    /// Get all field names.
    pub fn get_field_names(&self) -> Vec<String> {
        let columns = self.columns.read().unwrap();
        columns.keys().cloned().collect()
    }

    /// Get column statistics.
    pub fn get_column_stats(&self, field_name: &str) -> Option<ColumnStats> {
        let columns = self.columns.read().unwrap();
        if let Some(column) = columns.get(field_name) {
            let doc_count = column.doc_count();
            let value_frequencies = column.get_value_frequencies();
            let unique_values = value_frequencies.len() as u32;

            Some(ColumnStats {
                field_name: field_name.to_string(),
                doc_count,
                unique_values,
                value_frequencies,
            })
        } else {
            None
        }
    }

    /// Persist columns to storage.
    pub fn flush(&self) -> Result<()> {
        let columns = self.columns.read().unwrap();

        for (field_name, column) in columns.iter() {
            let values = column.get_all_values();
            let serialized = serde_json::to_vec(&values)?;

            let column_file = format!("columns/{field_name}.json");
            let mut output = self.storage.create_output(&column_file)?;
            output.write_all(&serialized)?;
            output.flush()?;
        }

        Ok(())
    }

    /// Load columns from storage.
    pub fn load(&self) -> Result<()> {
        // Implementation would load column data from storage
        // This is a simplified version
        Ok(())
    }
}

/// Statistics for a column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    /// Field name
    pub field_name: String,
    /// Number of documents with values in this column
    pub doc_count: u32,
    /// Number of unique values
    pub unique_values: u32,
    /// Value frequencies
    pub value_frequencies: HashMap<ColumnValue, u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryStorage;

    #[test]
    fn test_column_value_serialization() {
        let values = vec![
            ColumnValue::String("hello".to_string()),
            ColumnValue::I32(42),
            ColumnValue::I64(-1000),
            ColumnValue::U32(100),
            ColumnValue::U64(99999),
            ColumnValue::F32(std::f32::consts::PI),
            ColumnValue::F64(std::f64::consts::E),
            ColumnValue::Bool(true),
            ColumnValue::Bool(false),
            ColumnValue::DateTime(1609459200), // 2021-01-01 00:00:00 UTC
            ColumnValue::Null,
        ];

        for value in values {
            let bytes = value.to_bytes().unwrap();
            let deserialized = ColumnValue::from_bytes(&bytes).unwrap();
            assert_eq!(value, deserialized);
        }
    }

    #[test]
    fn test_column_value_comparison() {
        assert!(ColumnValue::I32(5) < ColumnValue::I32(10));
        assert!(
            ColumnValue::String("apple".to_string()) < ColumnValue::String("banana".to_string())
        );
        assert!(ColumnValue::Bool(false) < ColumnValue::Bool(true));
        assert!(ColumnValue::Null < ColumnValue::I32(0));

        // Cross-type numeric comparison
        assert!(ColumnValue::I32(5) < ColumnValue::I64(10));
        assert!(ColumnValue::U32(5) < ColumnValue::U64(10));
        assert!(ColumnValue::F32(std::f32::consts::PI) < ColumnValue::F64(3.15));
    }

    #[test]
    fn test_column_operations() {
        let column = Column::new("test_field".to_string());

        // Add some values
        column
            .add_value(1, ColumnValue::String("apple".to_string()))
            .unwrap();
        column
            .add_value(2, ColumnValue::String("banana".to_string()))
            .unwrap();
        column
            .add_value(3, ColumnValue::String("apple".to_string()))
            .unwrap();
        column.add_value(4, ColumnValue::Null).unwrap();

        assert_eq!(column.doc_count(), 4);
        assert_eq!(
            column.get_value(1),
            Some(ColumnValue::String("apple".to_string()))
        );
        assert_eq!(column.get_value(5), None);

        let frequencies = column.get_value_frequencies();
        assert_eq!(
            frequencies.get(&ColumnValue::String("apple".to_string())),
            Some(&2)
        );
        assert_eq!(
            frequencies.get(&ColumnValue::String("banana".to_string())),
            Some(&1)
        );
        assert_eq!(frequencies.get(&ColumnValue::Null), Some(&1));

        let apple_docs =
            column.find_documents_with_value(&ColumnValue::String("apple".to_string()));
        assert_eq!(apple_docs, vec![1, 3]);
    }

    #[test]
    fn test_column_storage() {
        let storage = Arc::new(MemoryStorage::new(
            crate::storage::memory::MemoryStorageConfig::default(),
        ));
        let column_storage = ColumnStorage::new(storage);

        // Add values to different fields
        column_storage
            .add_value("title", 1, ColumnValue::String("Document 1".to_string()))
            .unwrap();
        column_storage
            .add_value("title", 2, ColumnValue::String("Document 2".to_string()))
            .unwrap();
        column_storage
            .add_value("score", 1, ColumnValue::F32(0.85))
            .unwrap();
        column_storage
            .add_value("score", 2, ColumnValue::F32(0.92))
            .unwrap();

        assert_eq!(
            column_storage.get_value("title", 1),
            Some(ColumnValue::String("Document 1".to_string()))
        );
        assert_eq!(
            column_storage.get_value("score", 2),
            Some(ColumnValue::F32(0.92))
        );

        let field_names = column_storage.get_field_names();
        assert!(field_names.contains(&"title".to_string()));
        assert!(field_names.contains(&"score".to_string()));

        let title_stats = column_storage.get_column_stats("title").unwrap();
        assert_eq!(title_stats.doc_count, 2);
        assert_eq!(title_stats.unique_values, 2);
    }

    #[test]
    fn test_column_range_queries() {
        let column = Column::new("score".to_string());

        column.add_value(1, ColumnValue::F32(0.1)).unwrap();
        column.add_value(2, ColumnValue::F32(0.5)).unwrap();
        column.add_value(3, ColumnValue::F32(0.8)).unwrap();
        column.add_value(4, ColumnValue::F32(0.9)).unwrap();
        column.add_value(5, ColumnValue::F32(1.0)).unwrap();

        let docs_in_range =
            column.find_documents_in_range(&ColumnValue::F32(0.4), &ColumnValue::F32(0.85));
        assert_eq!(docs_in_range, vec![2, 3]);

        let values_in_range = column.get_values_in_range(2, 4);
        assert_eq!(values_in_range.len(), 3);
        assert_eq!(values_in_range[0].0, 2);
        assert_eq!(values_in_range[1].0, 3);
        assert_eq!(values_in_range[2].0, 4);
    }
}
