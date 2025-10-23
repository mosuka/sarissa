//! CSV format document converter.
//!
//! Converts CSV data into Documents where the first row contains field names:
//! ```csv
//! title,year,price,active
//! Rust Programming,2024,19.99,true
//! Python Basics,2023,15.50,false
//! ```

use std::sync::Arc;

use csv::ReaderBuilder;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::document::converter::DocumentConverter;
use crate::document::document::Document;
use crate::document::field_value::FieldValue;
use crate::error::{Result, SageError};

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

impl DocumentConverter for CsvDocumentConverter {
    fn convert(&self, input: &str) -> Result<Document> {
        let mut reader = ReaderBuilder::new()
            .delimiter(self.delimiter)
            .trim(csv::Trim::All)
            .flexible(self.flexible)
            .from_reader(input.as_bytes());

        // Get headers
        let headers = reader
            .headers()
            .map_err(|e| SageError::parse(format!("Failed to read CSV headers: {}", e)))?
            .clone();

        if headers.is_empty() {
            return Err(SageError::parse("CSV header is empty"));
        }

        // Read the first data record
        let mut records = reader.records();
        let record = records
            .next()
            .ok_or_else(|| SageError::parse("CSV has only header, no data rows to convert"))?
            .map_err(|e| SageError::parse(format!("Failed to read CSV record: {}", e)))?;

        if record.len() != headers.len() {
            return Err(SageError::parse(format!(
                "CSV field count mismatch: expected {} fields, found {}",
                headers.len(),
                record.len()
            )));
        }

        let mut doc = Document::new();

        for (header, value) in headers.iter().zip(record.iter()) {
            if !value.is_empty() {
                let field_value = self.infer_field_value(value);
                doc.add_field(header, field_value);
            }
        }

        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_basic_parsing() {
        let converter = CsvDocumentConverter::new();
        let csv = "title,year,price\nRust Programming,2024,19.99";
        let doc = converter.convert(csv).unwrap();

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
    fn test_csv_type_inference() {
        let converter = CsvDocumentConverter::new();
        let csv = "title,year,price,active\nTest,2024,19.99,true";
        let doc = converter.convert(csv).unwrap();

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
    fn test_csv_quoted_fields() {
        let converter = CsvDocumentConverter::new();
        let csv = r#"title,description
"Rust, Programming","A book about Rust, the language""#;
        let doc = converter.convert(csv).unwrap();

        assert_eq!(
            doc.get_field("title").unwrap().as_text().unwrap(),
            "Rust, Programming"
        );
        assert_eq!(
            doc.get_field("description").unwrap().as_text().unwrap(),
            "A book about Rust, the language"
        );
    }

    #[test]
    fn test_csv_empty_fields() {
        let converter = CsvDocumentConverter::new();
        let csv = "title,year,price\nRust Programming,,19.99";
        let doc = converter.convert(csv).unwrap();

        assert!(doc.has_field("title"));
        assert!(!doc.has_field("year")); // Empty field should not be added
        assert!(doc.has_field("price"));
    }

    #[test]
    fn test_csv_custom_delimiter() {
        let converter = CsvDocumentConverter::new().with_delimiter('\t');
        let csv = "title\tyear\tprice\nRust Programming\t2024\t19.99";
        let doc = converter.convert(csv).unwrap();

        assert_eq!(
            doc.get_field("title").unwrap().as_text().unwrap(),
            "Rust Programming"
        );
        assert!(matches!(
            doc.get_field("year").unwrap(),
            FieldValue::Integer(2024)
        ));
    }

    #[test]
    fn test_csv_empty_input() {
        let converter = CsvDocumentConverter::new();
        let result = converter.convert("");
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_header_only() {
        let converter = CsvDocumentConverter::new();
        let result = converter.convert("title,year,price");
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_field_count_mismatch() {
        let converter = CsvDocumentConverter::new();
        let csv = "title,year,price\nRust Programming,2024";
        let result = converter.convert(csv);
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_escaped_quotes() {
        let converter = CsvDocumentConverter::new();
        let csv = r#"title,description
"Book with ""quotes""","Description with ""special"" chars""#;
        let doc = converter.convert(csv).unwrap();

        assert_eq!(
            doc.get_field("title").unwrap().as_text().unwrap(),
            r#"Book with "quotes""#
        );
        assert_eq!(
            doc.get_field("description").unwrap().as_text().unwrap(),
            r#"Description with "special" chars"#
        );
    }

    #[test]
    fn test_csv_with_trim() {
        let converter = CsvDocumentConverter::new().with_trim(true);
        let csv = "title, year, price\n  Rust Programming  , 2024 , 19.99  ";
        let doc = converter.convert(csv).unwrap();

        assert_eq!(
            doc.get_field("title").unwrap().as_text().unwrap(),
            "Rust Programming"
        );
        assert!(matches!(
            doc.get_field("year").unwrap(),
            FieldValue::Integer(2024)
        ));
    }
}
