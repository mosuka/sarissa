//! CSV format document converter.
//!
//! Converts CSV files into Documents where the first row contains field names:
//! ```csv
//! title,year,price,active
//! Rust Programming,2024,19.99,true
//! Python Basics,2023,15.50,false
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use csv::{Reader, ReaderBuilder, StringRecord};

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::document::converter::DocumentConverter;
use crate::document::document::Document;
use crate::document::field_value::FieldValue;
use crate::error::{Result, SageError};
use crate::query::geo::GeoPoint;

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
#[derive(Clone)]
pub struct CsvDocumentConverter {
    /// Analyzer to use for text fields (optional).
    analyzer: Option<Arc<dyn Analyzer>>,
    /// CSV delimiter character (default: ',')
    delimiter: u8,
    /// Whether to trim whitespace from fields
    trim: bool,
    /// Whether to allow flexible field counts
    flexible: bool,
}

impl std::fmt::Debug for CsvDocumentConverter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsvDocumentConverter")
            .field("analyzer", &self.analyzer.as_ref().map(|_| "<Analyzer>"))
            .field("delimiter", &(self.delimiter as char))
            .field("trim", &self.trim)
            .field("flexible", &self.flexible)
            .finish()
    }
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
            analyzer: None,
            delimiter: b',',
            trim: true,
            flexible: false,
        }
    }

    /// Create a CSV converter with a custom analyzer.
    pub fn with_analyzer(analyzer: Arc<dyn Analyzer>) -> Self {
        CsvDocumentConverter {
            analyzer: Some(analyzer),
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
            Err(e) => return Some(Err(SageError::parse(format!("Failed to read CSV record: {}", e)))),
        };

        if !self.converter.flexible && record.len() != self.headers.len() {
            return Some(Err(SageError::parse(format!(
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
                    let entry = geo_fields.entry(base_name.to_string()).or_insert((None, None));

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
            doc.add_field(header, field_value);
        }

        // Second pass: create GeoPoint fields from collected lat/lon pairs
        for (base_name, (lat, lon)) in geo_fields {
            if let (Some(lat_val), Some(lon_val)) = (lat, lon) {
                doc.add_field(
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
            .map_err(|e| SageError::parse(format!("Failed to open CSV file: {}", e)))?;

        let mut reader = ReaderBuilder::new()
            .delimiter(self.delimiter)
            .trim(csv::Trim::All)
            .flexible(self.flexible)
            .from_reader(file);

        // Get headers
        let headers = reader
            .headers()
            .map_err(|e| SageError::parse(format!("Failed to read CSV headers: {}", e)))?
            .clone();

        if headers.is_empty() {
            return Err(SageError::parse("CSV header is empty"));
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
            doc.get_field("title").unwrap().as_text().unwrap(),
            "Rust Programming"
        );
        assert!(matches!(
            doc.get_field("year").unwrap(),
            FieldValue::Integer(2024)
        ));
        assert!(matches!(
            doc.get_field("price").unwrap(),
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
        assert_eq!(doc1.get_field("title").unwrap().as_text().unwrap(), "Rust Programming");

        let doc2 = docs[1].as_ref().unwrap();
        assert_eq!(doc2.get_field("title").unwrap().as_text().unwrap(), "Python Basics");
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
            doc.get_field("title").unwrap(),
            FieldValue::Text(_)
        ));
        assert!(matches!(
            doc.get_field("year").unwrap(),
            FieldValue::Integer(2024)
        ));
        assert!(matches!(
            doc.get_field("price").unwrap(),
            FieldValue::Float(_)
        ));
        assert!(matches!(
            doc.get_field("active").unwrap(),
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
        assert_eq!(doc1.get_field("title").unwrap().as_text().unwrap(), "Tokyo Tower");
        assert!(matches!(
            doc1.get_field("location").unwrap(),
            FieldValue::Geo(_)
        ));

        if let FieldValue::Geo(geo) = doc1.get_field("location").unwrap() {
            assert_eq!(geo.lat, 35.6762);
            assert_eq!(geo.lon, 139.6503);
        }

        let doc2 = docs[1].as_ref().unwrap();
        assert_eq!(doc2.get_field("title").unwrap().as_text().unwrap(), "Eiffel Tower");
        assert!(matches!(
            doc2.get_field("location").unwrap(),
            FieldValue::Geo(_)
        ));
    }

    #[test]
    fn test_csv_multiple_geo_fields() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id,origin.lat,origin.lon,destination.lat,destination.lon").unwrap();
        writeln!(file, "route001,35.6762,139.6503,51.5074,-0.1278").unwrap();
        file.flush().unwrap();

        let converter = CsvDocumentConverter::new();
        let mut iter = converter.convert(file.path()).unwrap();
        let doc = iter.next().unwrap().unwrap();

        assert!(matches!(
            doc.get_field("origin").unwrap(),
            FieldValue::Geo(_)
        ));
        assert!(matches!(
            doc.get_field("destination").unwrap(),
            FieldValue::Geo(_)
        ));

        if let FieldValue::Geo(origin) = doc.get_field("origin").unwrap() {
            assert_eq!(origin.lat, 35.6762);
            assert_eq!(origin.lon, 139.6503);
        }

        if let FieldValue::Geo(dest) = doc.get_field("destination").unwrap() {
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
