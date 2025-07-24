//! Error types for the Sarissa library.

use anyhow;
use std::io;
use thiserror::Error;

/// The main error type for Sarissa operations.
#[derive(Error, Debug)]
pub enum SarissaError {
    /// I/O errors (file operations, network, etc.)
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Index-related errors
    #[error("Index error: {0}")]
    Index(String),

    /// Schema-related errors
    #[error("Schema error: {0}")]
    Schema(String),

    /// Analysis-related errors (tokenization, filtering, etc.)
    #[error("Analysis error: {0}")]
    Analysis(String),

    /// Query-related errors (parsing, invalid queries, etc.)
    #[error("Query error: {0}")]
    Query(String),

    /// Storage-related errors
    #[error("Storage error: {0}")]
    Storage(String),

    /// Field-related errors
    #[error("Field error: {0}")]
    Field(String),

    /// Benchmark-related errors
    #[error("Benchmark error: {0}")]
    BenchmarkFailed(String),

    /// Thread join errors
    #[error("Thread join error: {0}")]
    ThreadJoinError(String),

    /// Operation cancelled
    #[error("Operation cancelled: {0}")]
    OperationCancelled(String),

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Generic error for other cases
    #[error("Error: {0}")]
    Other(String),

    /// Generic anyhow error
    #[error("Anyhow error: {0}")]
    Anyhow(#[from] anyhow::Error),

    /// Machine learning error
    #[error("ML error: {0}")]
    ML(String),
}

/// Result type alias for operations that may fail with SarissaError.
pub type Result<T> = std::result::Result<T, SarissaError>;

/// Implement From<MLError> for SarissaError
// Temporarily disabled while ML module is disabled
// impl From<crate::ml::MLError> for SarissaError {
//     fn from(err: crate::ml::MLError) -> Self {
//         SarissaError::ML(err.to_string())
//     }
// }
impl SarissaError {
    /// Create a new index error.
    pub fn index<S: Into<String>>(msg: S) -> Self {
        SarissaError::Index(msg.into())
    }

    /// Create a new schema error.
    pub fn schema<S: Into<String>>(msg: S) -> Self {
        SarissaError::Schema(msg.into())
    }

    /// Create a new analysis error.
    pub fn analysis<S: Into<String>>(msg: S) -> Self {
        SarissaError::Analysis(msg.into())
    }

    /// Create a new query error.
    pub fn query<S: Into<String>>(msg: S) -> Self {
        SarissaError::Query(msg.into())
    }

    /// Create a new storage error.
    pub fn storage<S: Into<String>>(msg: S) -> Self {
        SarissaError::Storage(msg.into())
    }

    /// Create a new field error.
    pub fn field<S: Into<String>>(msg: S) -> Self {
        SarissaError::Field(msg.into())
    }

    /// Create a new ML error.
    pub fn ml<S: Into<String>>(msg: S) -> Self {
        SarissaError::ML(msg.into())
    }

    /// Create a new generic error.
    pub fn other<S: Into<String>>(msg: S) -> Self {
        SarissaError::Other(msg.into())
    }

    /// Create a new timeout error.
    pub fn timeout<S: Into<String>>(msg: S) -> Self {
        SarissaError::Other(format!("Timeout: {}", msg.into()))
    }

    /// Create a new invalid argument error.
    pub fn invalid_argument<S: Into<String>>(msg: S) -> Self {
        SarissaError::Other(format!("Invalid argument: {}", msg.into()))
    }

    /// Create a new internal error.
    pub fn internal<S: Into<String>>(msg: S) -> Self {
        SarissaError::Other(format!("Internal error: {}", msg.into()))
    }

    /// Create a new not found error.
    pub fn not_found<S: Into<String>>(msg: S) -> Self {
        SarissaError::Other(format!("Not found: {}", msg.into()))
    }

    /// Create a new cancelled error.
    pub fn cancelled<S: Into<String>>(msg: S) -> Self {
        SarissaError::OperationCancelled(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_construction() {
        let error = SarissaError::index("Test index error");
        assert_eq!(error.to_string(), "Index error: Test index error");

        let error = SarissaError::schema("Test schema error");
        assert_eq!(error.to_string(), "Schema error: Test schema error");

        let error = SarissaError::analysis("Test analysis error");
        assert_eq!(error.to_string(), "Analysis error: Test analysis error");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "File not found");
        let sarissa_error = SarissaError::from(io_error);

        match sarissa_error {
            SarissaError::Io(_) => {} // Expected
            _ => panic!("Expected IO error variant"),
        }
    }
}
