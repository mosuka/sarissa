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
