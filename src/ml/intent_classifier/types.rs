//! Common types for intent classification.

use serde::{Deserialize, Serialize};

/// Training sample for intent classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentSample {
    /// Query text.
    pub query: String,
    /// Intent label.
    pub intent: String,
}

/// Query intent classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryIntent {
    /// Informational query (seeking knowledge).
    Informational,
    /// Navigational query (seeking specific resource).
    Navigational,
    /// Transactional query (seeking to perform action).
    Transactional,
    /// Unknown intent.
    Unknown,
}
