//! Shared JSON → `Document` parsing for the CLI's document-ingestion commands.
//!
//! Every command that accepts a document as a JSON string — `put doc`,
//! `add doc`, the `put docs`/`add docs` JSONL entries (via
//! [`crate::commands::bulk`]), and the REPL's `add doc`/`put doc` — goes
//! through [`parse_document_json`], which delegates to
//! [`laurus::json_to_document`]. That is the same canonical converter used
//! by the laurus-server HTTP gateway and laurus-mcp, so all JSON-accepting
//! transports agree on one document shape: `{"fields": {"name": value, ...}}`
//! with plain (untagged) field values, no `{"Text": ...}`-style type tags
//! and no `document` wrapper.

use anyhow::{Context, Result};

/// Convert an already-parsed JSON value of the shape `{"fields": {...}}`
/// into a [`laurus::Document`].
///
/// Shared by [`parse_document_json`] (which parses the JSON text itself)
/// and [`crate::commands::bulk::parse_entry`] (which needs the parsed
/// [`serde_json::Value`] anyway, to also read the sibling `"id"` key).
///
/// # Errors
///
/// Returns an error if [`laurus::json_to_document`] rejects the value's
/// structure (missing/non-object `fields`) or any individual field value.
/// When the underlying error looks like it came from the pre-migration
/// type-tagged format (e.g. `{"title": {"Text": "Hello"}}`) — which always
/// fails at the "only supported as geographic points or bytes" object-shape
/// check — a hint pointing at the new plain-value format is appended.
pub(crate) fn document_from_value(value: &serde_json::Value) -> Result<laurus::Document> {
    laurus::json_to_document(value).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("geographic points or bytes") {
            anyhow::anyhow!(
                "{msg}\n\n\
                 hint: document field values are now plain JSON — \
                 `\"title\": \"Hello\"`, not the old tagged format \
                 `\"title\": {{\"Text\": \"Hello\"}}`."
            )
        } else {
            anyhow::anyhow!(msg)
        }
    })
}

/// Parse a JSON string of the shape `{"fields": {...}}` into a [`laurus::Document`].
///
/// # Arguments
///
/// * `data_json` - The raw JSON text to parse.
///
/// # Errors
///
/// Returns an error if `data_json` is not valid JSON, or from
/// [`document_from_value`] for a structurally invalid or unparseable value.
pub fn parse_document_json(data_json: &str) -> Result<laurus::Document> {
    let value: serde_json::Value =
        serde_json::from_str(data_json).context("input is not valid JSON")?;
    document_from_value(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_values() {
        let doc = parse_document_json(r#"{"fields": {"title": "Hello", "views": 42}}"#).unwrap();
        assert_eq!(
            doc.fields.get("title"),
            Some(&laurus::DataValue::Text("Hello".into()))
        );
        assert_eq!(doc.fields.get("views"), Some(&laurus::DataValue::Int64(42)));
    }

    #[test]
    fn rejects_invalid_json_with_context() {
        let err = parse_document_json("{not json").unwrap_err();
        assert!(err.to_string().contains("not valid JSON"));
    }

    #[test]
    fn old_tagged_format_gets_a_migration_hint() {
        let err = parse_document_json(r#"{"fields": {"title": {"Text": "Hello"}}}"#).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("hint"), "must include a migration hint: {msg}");
        assert!(
            msg.contains("no longer") || msg.contains("plain JSON"),
            "hint must explain the new format: {msg}"
        );
    }
}
