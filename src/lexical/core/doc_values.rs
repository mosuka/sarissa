//! DocValues implementation for efficient field access during sorting and aggregations.
//!
//! DocValues are column-oriented storage for field values, optimized for:
//! - Sorting search results by field values
//! - Faceting and aggregations
//! - Field-based scoring
//!
//! Unlike stored fields (row-oriented), DocValues store values in a columnar format
//! where accessing all values of a single field is very efficient.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;

use crate::error::{SarissaError, Result};
use crate::lexical::document::field::FieldValue;
use crate::storage::Storage;

/// DocValues file extension
const DOC_VALUES_EXTENSION: &str = ".dv";

/// DocValues format for a single field.
/// Stores a mapping from document ID to field value.
#[derive(Debug, Clone)]
pub struct FieldDocValues {
    /// Field name
    pub field_name: String,
    /// Mapping from doc_id to field value
    /// Using Vec with sparse storage (None for missing values)
    values: Vec<Option<FieldValue>>,
}

impl FieldDocValues {
    /// Create a new FieldDocValues
    pub fn new(field_name: String) -> Self {
        FieldDocValues {
            field_name,
            values: Vec::new(),
        }
    }

    /// Set a value for a document
    pub fn set(&mut self, doc_id: u64, value: FieldValue) {
        let doc_id = doc_id as usize;

        // Expand vector if needed
        if doc_id >= self.values.len() {
            self.values.resize(doc_id + 1, None);
        }

        self.values[doc_id] = Some(value);
    }

    /// Get a value for a document
    pub fn get(&self, doc_id: u64) -> Option<&FieldValue> {
        let doc_id = doc_id as usize;
        if doc_id < self.values.len() {
            self.values[doc_id].as_ref()
        } else {
            None
        }
    }

    /// Get the number of values
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Writer for DocValues
pub struct DocValuesWriter {
    /// Storage for writing DocValues
    storage: Arc<dyn Storage>,
    /// Segment name
    segment_name: String,
    /// Field DocValues being built (field_name -> FieldDocValues)
    fields: HashMap<String, FieldDocValues>,
}

impl DocValuesWriter {
    /// Create a new DocValuesWriter
    pub fn new(storage: Arc<dyn Storage>, segment_name: String) -> Self {
        DocValuesWriter {
            storage,
            segment_name,
            fields: HashMap::new(),
        }
    }

    /// Add a field value for a document
    pub fn add_value(&mut self, doc_id: u64, field_name: &str, value: FieldValue) {
        self.fields
            .entry(field_name.to_string())
            .or_insert_with(|| FieldDocValues::new(field_name.to_string()))
            .set(doc_id, value);
    }

    /// Write DocValues to storage
    pub fn write(&self) -> Result<()> {
        let dv_filename = format!("{}{}", self.segment_name, DOC_VALUES_EXTENSION);
        let mut output = self.storage.create_output(&dv_filename)?;

        // Write magic number and version
        output.write_all(b"DVFF")?; // DocValues File Format
        output.write_all(&[1u8, 0u8])?; // Version 1.0

        // Write number of fields
        let num_fields = self.fields.len() as u32;
        output.write_all(&num_fields.to_le_bytes())?;

        // Write each field's DocValues
        for (field_name, field_dv) in &self.fields {
            // Write field name length and name
            let name_bytes = field_name.as_bytes();
            output.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            output.write_all(name_bytes)?;

            // Write number of values
            let num_values = field_dv.values.len() as u64;
            output.write_all(&num_values.to_le_bytes())?;

            // Write values using bincode 2.0
            let serialized =
                bincode::serde::encode_to_vec(&field_dv.values, bincode::config::standard())
                    .map_err(|e| {
                        SarissaError::Index(format!("Failed to serialize DocValues: {}", e))
                    })?;

            output.write_all(&(serialized.len() as u64).to_le_bytes())?;
            output.write_all(&serialized)?;
        }

        output.flush()?;
        Ok(())
    }
}

/// Reader for DocValues
#[derive(Debug)]
pub struct DocValuesReader {
    /// Field DocValues (field_name -> FieldDocValues)
    fields: HashMap<String, FieldDocValues>,
}

impl DocValuesReader {
    /// Load DocValues from storage
    pub fn load(storage: Arc<dyn Storage>, segment_name: &str) -> Result<Self> {
        let dv_filename = format!("{}{}", segment_name, DOC_VALUES_EXTENSION);

        // Try to open the DocValues file
        let mut input = match storage.open_input(&dv_filename) {
            Ok(input) => input,
            Err(_) => {
                // If DocValues file doesn't exist, return empty reader
                return Ok(DocValuesReader {
                    fields: HashMap::new(),
                });
            }
        };

        // Read and verify magic number
        let mut magic = [0u8; 4];
        input.read_exact(&mut magic)?;
        if &magic != b"DVFF" {
            return Err(SarissaError::Index(
                "Invalid DocValues file format".to_string(),
            ));
        }

        // Read version
        let mut version = [0u8; 2];
        input.read_exact(&mut version)?;
        if version[0] != 1 {
            return Err(SarissaError::Index(format!(
                "Unsupported DocValues version: {}.{}",
                version[0], version[1]
            )));
        }

        // Read number of fields
        let mut num_fields_bytes = [0u8; 4];
        input.read_exact(&mut num_fields_bytes)?;
        let num_fields = u32::from_le_bytes(num_fields_bytes);

        let mut fields = HashMap::new();

        // Read each field's DocValues
        for _ in 0..num_fields {
            // Read field name
            let mut name_len_bytes = [0u8; 4];
            input.read_exact(&mut name_len_bytes)?;
            let name_len = u32::from_le_bytes(name_len_bytes) as usize;

            let mut name_bytes = vec![0u8; name_len];
            input.read_exact(&mut name_bytes)?;
            let field_name = String::from_utf8(name_bytes)
                .map_err(|e| SarissaError::Index(format!("Invalid field name: {}", e)))?;

            // Read number of values
            let mut num_values_bytes = [0u8; 8];
            input.read_exact(&mut num_values_bytes)?;
            let _num_values = u64::from_le_bytes(num_values_bytes);

            // Read serialized values
            let mut data_len_bytes = [0u8; 8];
            input.read_exact(&mut data_len_bytes)?;
            let data_len = u64::from_le_bytes(data_len_bytes) as usize;

            let mut data = vec![0u8; data_len];
            input.read_exact(&mut data)?;

            let (values, _): (Vec<Option<FieldValue>>, _) =
                bincode::serde::decode_from_slice(&data, bincode::config::standard()).map_err(
                    |e| SarissaError::Index(format!("Failed to deserialize DocValues: {}", e)),
                )?;

            fields.insert(field_name.clone(), FieldDocValues { field_name, values });
        }

        Ok(DocValuesReader { fields })
    }

    /// Get DocValues for a field
    pub fn get_field(&self, field_name: &str) -> Option<&FieldDocValues> {
        self.fields.get(field_name)
    }

    /// Get a value for a document and field
    pub fn get_value(&self, field_name: &str, doc_id: u64) -> Option<&FieldValue> {
        self.fields.get(field_name).and_then(|dv| dv.get(doc_id))
    }

    /// Check if a field has DocValues
    pub fn has_field(&self, field_name: &str) -> bool {
        self.fields.contains_key(field_name)
    }

    /// Get all field names with DocValues
    pub fn field_names(&self) -> Vec<String> {
        self.fields.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;

    #[test]
    fn test_field_doc_values() {
        let mut dv = FieldDocValues::new("test_field".to_string());

        // Set some values
        dv.set(0, FieldValue::Integer(100));
        dv.set(1, FieldValue::Text("hello".to_string()));
        dv.set(5, FieldValue::Float(3.15));

        // Get values
        assert_eq!(dv.get(0), Some(&FieldValue::Integer(100)));
        assert_eq!(dv.get(1), Some(&FieldValue::Text("hello".to_string())));
        assert_eq!(dv.get(2), None);
        assert_eq!(dv.get(5), Some(&FieldValue::Float(3.15)));
    }

    #[test]
    fn test_doc_values_write_read() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let segment_name = "segment_0".to_string();

        // Write DocValues
        {
            let mut writer = DocValuesWriter::new(storage.clone(), segment_name.clone());
            writer.add_value(0, "year", FieldValue::Integer(2023));
            writer.add_value(1, "year", FieldValue::Integer(2024));
            writer.add_value(0, "rating", FieldValue::Float(4.5));
            writer.add_value(1, "rating", FieldValue::Float(5.0));
            writer.write().unwrap();
        }

        // Read DocValues
        {
            let reader = DocValuesReader::load(storage.clone(), &segment_name).unwrap();
            assert!(reader.has_field("year"));
            assert!(reader.has_field("rating"));
            assert!(!reader.has_field("unknown"));

            assert_eq!(
                reader.get_value("year", 0),
                Some(&FieldValue::Integer(2023))
            );
            assert_eq!(
                reader.get_value("year", 1),
                Some(&FieldValue::Integer(2024))
            );
            assert_eq!(reader.get_value("rating", 0), Some(&FieldValue::Float(4.5)));
            assert_eq!(reader.get_value("rating", 1), Some(&FieldValue::Float(5.0)));
        }
    }
}
