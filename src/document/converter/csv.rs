//! CSV format document converter.
//!
//! This module provides [`CsvDocumentConverter`] for converting CSV (Comma-Separated Values)
//! files into [`Document`] objects. The first row is treated as the header containing
//! field names, and each subsequent row becomes a document.
//!
//! # Format Requirements
//!
//! - First row must contain field names (header)
//! - Fields are separated by commas (configurable)
//! - Empty fields are skipped
//! - Automatic type inference for integers, floats, and booleans
//!
//! # CSV Format Example
//!
//! ```csv
//! title,year,price,active
//! Rust Programming,2024,19.99,true
//! Python Basics,2023,15.50,false
//! ```
//!
//! # Geographic Coordinates
//!
//! Geographic coordinates can be specified using dotted field names:
//!
//! ```csv
//! name,location.lat,location.lon
//! Tokyo Tower,35.6762,139.6503
//! Eiffel Tower,48.8584,2.2945
//! ```
//!
//! The converter will automatically combine `location.lat` and `location.lon`
//! into a single `GeoPoint` field named `location`.
//!
//! # Type Inference
//!
//! Field values are automatically inferred:
//! - **Boolean**: "true", "false", "yes", "no", "t", "f", "y", "n", "1", "0", "on", "off" (case-insensitive)
//! - **Integer**: Valid i64 values (e.g., "42", "-100")
//! - **Float**: Valid f64 values (e.g., "3.14", "-0.5")
//! - **Text**: Everything else (default)
//!
//! # Examples
//!
//! Basic CSV conversion:
//!
//! ```no_run
//! use platypus::document::converter::DocumentConverter;
//! use platypus::document::converter::csv::CsvDocumentConverter;
//!
//! let converter = CsvDocumentConverter::new();
//!
//! for doc in converter.convert("books.csv").unwrap() {
//!     let doc = doc.unwrap();
//!     println!("Title: {:?}", doc.get_field("title"));
//! }
//! ```
//!
//! Custom delimiter (TSV):
//!
//! ```no_run
//! use platypus::document::converter::DocumentConverter;
//! use platypus::document::converter::csv::CsvDocumentConverter;
//!
//! let converter = CsvDocumentConverter::new()
//!     .with_delimiter('\t');
//!
//! for doc in converter.convert("data.tsv").unwrap() {
//!     // Process tab-separated document...
//! }
//! ```
//!
//! Flexible mode (allow varying field counts):
//!
//! ```no_run
//! use platypus::document::converter::DocumentConverter;
//! use platypus::document::converter::csv::CsvDocumentConverter;
//!
//! let converter = CsvDocumentConverter::new()
//!     .with_flexible(true);
//!
//! for doc in converter.convert("irregular.csv").unwrap() {
//!     // Process documents with varying field counts...
//! }
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use csv::{Reader, ReaderBuilder, StringRecord};

use crate::document::converter::DocumentConverter;
use crate::document::document::Document;
use crate::document::field::FieldValue;
use crate::error::{PlatypusError, Result};
use crate::lexical::index::inverted::query::geo::GeoPoint;

/// A document converter for CSV format.
///
/// The first row is treated as the header containing field names.
/// Each subsequent row becomes a Document.
///
/// Supports:
/// - Text fields
/// - Integer fields (auto-detected)
/// - Float fields (auto-detected)
/// - Boolean fields (true/false, auto-detected)
#[derive(Clone, Debug)]
pub struct CsvDocumentConverter {
    /// CSV delimiter character (default: ',')
    delimiter: u8,
    /// Whether to trim whitespace from fields
    trim: bool,
    /// Whether to allow flexible field counts
    flexible: bool,
}

impl Default for CsvDocumentConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl CsvDocumentConverter {
    /// Create a new CSV converter with comma delimiter.
    pub fn new() -> Self {
        CsvDocumentConverter {
            delimiter: b',',
            trim: true,
            flexible: false,
        }
    }

    /// Set a custom delimiter character.
    pub fn with_delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = delimiter as u8;
        self
    }

    /// Set whether to trim whitespace from fields.
    pub fn with_trim(mut self, trim: bool) -> Self {
        self.trim = trim;
        self
    }

    /// Set whether to allow flexible field counts.
    pub fn with_flexible(mut self, flexible: bool) -> Self {
        self.flexible = flexible;
        self
    }

    /// Infer the field value type from a string.
    fn infer_field_value(&self, value: &str) -> FieldValue {
        // Try boolean
        if value.eq_ignore_ascii_case("true") {
            return FieldValue::Boolean(true);
        }
        if value.eq_ignore_ascii_case("false") {
            return FieldValue::Boolean(false);
        }

        // Try integer
        if let Ok(int_val) = value.parse::<i64>() {
            return FieldValue::Integer(int_val);
        }

        // Try float
        if let Ok(float_val) = value.parse::<f64>() {
            return FieldValue::Float(float_val);
        }

        // Default to text
        FieldValue::Text(value.to_string())
    }
}

/// Iterator over CSV documents.
pub struct CsvDocumentIterator {
    reader: Reader<File>,
    headers: StringRecord,
    converter: Arc<CsvDocumentConverter>,
}

impl Iterator for CsvDocumentIterator {
    type Item = Result<Document>;

    fn next(&mut self) -> Option<Self::Item> {
        let record = match self.reader.records().next()? {
            Ok(record) => record,
            Err(e) => {
                return Some(Err(PlatypusError::parse(format!(
                    "Failed to read CSV record: {}",
                    e
                ))));
            }
        };

        if !self.converter.flexible && record.len() != self.headers.len() {
            return Some(Err(PlatypusError::parse(format!(
                "CSV field count mismatch: expected {} fields, found {}",
                self.headers.len(),
                record.len()
            ))));
        }

        let mut doc = Document::new();
        let mut geo_fields: HashMap<String, (Option<f64>, Option<f64>)> = HashMap::new();

        // First pass: collect all fields and identify geo coordinates
        for (header, value) in self.headers.iter().zip(record.iter()) {
            if value.is_empty() {
                continue;
            }

            // Check if this is a dotted field name (e.g., "location.lat")
            if let Some(dot_pos) = header.find('.') {
                let base_name = &header[..dot_pos];
                let suffix = &header[dot_pos + 1..];

                if suffix == "lat" || suffix == "lon" {
                    let entry = geo_fields
                        .entry(base_name.to_string())
                        .or_insert((None, None));

                    if let Ok(float_val) = value.parse::<f64>() {
                        if suffix == "lat" {
                            entry.0 = Some(float_val);
                        } else {
                            entry.1 = Some(float_val);
                        }
                    }
                    continue; // Skip adding this as a regular field
                }
            }

            // Regular field
            let field_value = self.converter.infer_field_value(value);
            doc.add_field_value(header, field_value);
        }

        // Second pass: create GeoPoint fields from collected lat/lon pairs
        for (base_name, (lat, lon)) in geo_fields {
            if let (Some(lat_val), Some(lon_val)) = (lat, lon) {
                doc.add_field_value(
                    base_name,
                    FieldValue::Geo(GeoPoint {
                        lat: lat_val,
                        lon: lon_val,
                    }),
                );
            }
        }

        Some(Ok(doc))
    }
}

impl DocumentConverter for CsvDocumentConverter {
    type Iter = CsvDocumentIterator;

    fn convert<P: AsRef<Path>>(&self, path: P) -> Result<Self::Iter> {
        let file = File::open(path.as_ref())
            .map_err(|e| PlatypusError::parse(format!("Failed to open CSV file: {}", e)))?;

        let mut reader = ReaderBuilder::new()
            .delimiter(self.delimiter)
            .trim(csv::Trim::All)
            .flexible(self.flexible)
            .from_reader(file);

        // Get headers
        let headers = reader
            .headers()
            .map_err(|e| PlatypusError::parse(format!("Failed to read CSV headers: {}", e)))?
            .clone();

        if headers.is_empty() {
            return Err(PlatypusError::parse("CSV header is empty"));
        }

        Ok(CsvDocumentIterator {
            reader,
            headers,
            converter: Arc::new(self.clone()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_basic_parsing() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "title,year,price").unwrap();
        writeln!(file, "Rust Programming,2024,19.99").unwrap();
        file.flush().unwrap();

        let converter = CsvDocumentConverter::new();
        let mut iter = converter.convert(file.path()).unwrap();
        let doc = iter.next().unwrap().unwrap();

        assert_eq!(
            doc.get_field("title").unwrap().value.as_text().unwrap(),
            "Rust Programming"
        );
        assert!(matches!(
            &doc.get_field("year").unwrap().value,
            FieldValue::Integer(2024)
        ));
        assert!(matches!(
            &doc.get_field("price").unwrap().value,
            FieldValue::Float(_)
        ));
    }

    #[test]
    fn test_csv_multiple_rows() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "title,year,price").unwrap();
        writeln!(file, "Rust Programming,2024,19.99").unwrap();
        writeln!(file, "Python Basics,2023,15.50").unwrap();
        file.flush().unwrap();

        let converter = CsvDocumentConverter::new();
        let docs: Vec<_> = converter.convert(file.path()).unwrap().collect();

        assert_eq!(docs.len(), 2);

        let doc1 = docs[0].as_ref().unwrap();
        assert_eq!(
            doc1.get_field("title").unwrap().value.as_text().unwrap(),
            "Rust Programming"
        );

        let doc2 = docs[1].as_ref().unwrap();
        assert_eq!(
            doc2.get_field("title").unwrap().value.as_text().unwrap(),
            "Python Basics"
        );
    }

    #[test]
    fn test_csv_type_inference() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "title,year,price,active").unwrap();
        writeln!(file, "Test,2024,19.99,true").unwrap();
        file.flush().unwrap();

        let converter = CsvDocumentConverter::new();
        let mut iter = converter.convert(file.path()).unwrap();
        let doc = iter.next().unwrap().unwrap();

        assert!(matches!(
            &doc.get_field("title").unwrap().value,
            FieldValue::Text(_)
        ));
        assert!(matches!(
            &doc.get_field("year").unwrap().value,
            FieldValue::Integer(2024)
        ));
        assert!(matches!(
            &doc.get_field("price").unwrap().value,
            FieldValue::Float(_)
        ));
        assert!(matches!(
            &doc.get_field("active").unwrap().value,
            FieldValue::Boolean(true)
        ));
    }

    #[test]
    fn test_csv_empty_fields() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "title,year,price").unwrap();
        writeln!(file, "Rust Programming,,19.99").unwrap();
        file.flush().unwrap();

        let converter = CsvDocumentConverter::new();
        let mut iter = converter.convert(file.path()).unwrap();
        let doc = iter.next().unwrap().unwrap();

        assert!(doc.has_field("title"));
        assert!(!doc.has_field("year")); // Empty field should not be added
        assert!(doc.has_field("price"));
    }

    #[test]
    fn test_csv_file_not_found() {
        let converter = CsvDocumentConverter::new();
        let result = converter.convert("nonexistent.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_geo_point_with_dot_notation() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id,title,location.lat,location.lon,city").unwrap();
        writeln!(file, "doc001,Tokyo Tower,35.6762,139.6503,Tokyo").unwrap();
        writeln!(file, "doc002,Eiffel Tower,48.8584,2.2945,Paris").unwrap();
        file.flush().unwrap();

        let converter = CsvDocumentConverter::new();
        let docs: Vec<_> = converter.convert(file.path()).unwrap().collect();

        assert_eq!(docs.len(), 2);

        let doc1 = docs[0].as_ref().unwrap();
        assert_eq!(
            doc1.get_field("title").unwrap().value.as_text().unwrap(),
            "Tokyo Tower"
        );
        assert!(matches!(
            &doc1.get_field("location").unwrap().value,
            FieldValue::Geo(_)
        ));

        if let FieldValue::Geo(geo) = &doc1.get_field("location").unwrap().value {
            assert_eq!(geo.lat, 35.6762);
            assert_eq!(geo.lon, 139.6503);
        }

        let doc2 = docs[1].as_ref().unwrap();
        assert_eq!(
            doc2.get_field("title").unwrap().value.as_text().unwrap(),
            "Eiffel Tower"
        );
        assert!(matches!(
            &doc2.get_field("location").unwrap().value,
            FieldValue::Geo(_)
        ));
    }

    #[test]
    fn test_csv_multiple_geo_fields() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "id,origin.lat,origin.lon,destination.lat,destination.lon"
        )
        .unwrap();
        writeln!(file, "route001,35.6762,139.6503,51.5074,-0.1278").unwrap();
        file.flush().unwrap();

        let converter = CsvDocumentConverter::new();
        let mut iter = converter.convert(file.path()).unwrap();
        let doc = iter.next().unwrap().unwrap();

        assert!(matches!(
            &doc.get_field("origin").unwrap().value,
            FieldValue::Geo(_)
        ));
        assert!(matches!(
            &doc.get_field("destination").unwrap().value,
            FieldValue::Geo(_)
        ));

        if let FieldValue::Geo(origin) = &doc.get_field("origin").unwrap().value {
            assert_eq!(origin.lat, 35.6762);
            assert_eq!(origin.lon, 139.6503);
        }

        if let FieldValue::Geo(dest) = &doc.get_field("destination").unwrap().value {
            assert_eq!(dest.lat, 51.5074);
            assert_eq!(dest.lon, -0.1278);
        }
    }

    #[test]
    fn test_csv_geo_incomplete_coordinates() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id,title,location.lat,city").unwrap();
        writeln!(file, "doc001,Test,35.6762,Tokyo").unwrap();
        file.flush().unwrap();

        let converter = CsvDocumentConverter::new();
        let mut iter = converter.convert(file.path()).unwrap();
        let doc = iter.next().unwrap().unwrap();

        // location field should not exist because lon is missing
        assert!(!doc.has_field("location"));
        assert!(doc.has_field("title"));
        assert!(doc.has_field("city"));
    }
}
