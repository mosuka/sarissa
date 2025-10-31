//! Error types for the Yatagarasu library.

use std::io;

use anyhow;
use thiserror::Error;

/// The main error type for Yatagarasu operations.
#[derive(Error, Debug)]
pub enum YatagarasuError {
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

    /// Resource exhausted
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

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

/// Result type alias for operations that may fail with YatagarasuError.
pub type Result<T> = std::result::Result<T, YatagarasuError>;

/// Implement `From<MLError>` for YatagarasuError
impl From<crate::ml::MLError> for YatagarasuError {
    fn from(err: crate::ml::MLError) -> Self {
        YatagarasuError::ML(err.to_string())
    }
}
impl YatagarasuError {
    /// Create a new index error.
    pub fn index<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Index(msg.into())
    }

    /// Create a new schema error.
    pub fn schema<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Schema(msg.into())
    }

    /// Create a new analysis error.
    pub fn analysis<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Analysis(msg.into())
    }

    /// Create a new query error.
    pub fn query<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Query(msg.into())
    }

    /// Create a new parse error.
    pub fn parse<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Query(msg.into()) // Parse errors are treated as query errors
    }

    /// Create a new storage error.
    pub fn storage<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Storage(msg.into())
    }

    /// Create a new field error.
    pub fn field<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Field(msg.into())
    }

    /// Create a new ML error.
    pub fn ml<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::ML(msg.into())
    }

    /// Create a new generic error.
    pub fn other<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Other(msg.into())
    }

    /// Create a new timeout error.
    pub fn timeout<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Other(format!("Timeout: {}", msg.into()))
    }

    /// Create a new invalid config error.
    pub fn invalid_config<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Other(format!("Invalid configuration: {}", msg.into()))
    }

    /// Create a new invalid argument error.
    pub fn invalid_argument<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Other(format!("Invalid argument: {}", msg.into()))
    }

    /// Create a new internal error.
    pub fn internal<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Other(format!("Internal error: {}", msg.into()))
    }

    /// Create a new not found error.
    pub fn not_found<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::Other(format!("Not found: {}", msg.into()))
    }

    /// Create a new cancelled error.
    pub fn cancelled<S: Into<String>>(msg: S) -> Self {
        YatagarasuError::OperationCancelled(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_construction() {
        let error = YatagarasuError::index("Test index error");
        assert_eq!(error.to_string(), "Index error: Test index error");

        let error = YatagarasuError::schema("Test schema error");
        assert_eq!(error.to_string(), "Schema error: Test schema error");

        let error = YatagarasuError::analysis("Test analysis error");
        assert_eq!(error.to_string(), "Analysis error: Test analysis error");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "File not found");
        let yatagarasu_error = YatagarasuError::from(io_error);

        match yatagarasu_error {
            YatagarasuError::Io(_) => {} // Expected
            _ => panic!("Expected IO error variant"),
        }
    }
}
