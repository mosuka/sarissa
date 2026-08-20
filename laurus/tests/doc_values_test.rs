//! Integration tests for issue #943 — DocValues availability and
//! freshness at the reader level.

use std::sync::Arc;

use laurus::lexical::LexicalIndexWriter;
use laurus::lexical::{InvertedIndexWriter, InvertedIndexWriterConfig};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};

/// #943: `has_doc_values` must answer correctly as the very first
/// operation on a fresh reader — it used to report `false` until some
/// other call happened to load the DocValues cache.
#[test]
fn has_doc_values_is_correct_on_a_fresh_reader() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut writer =
        InvertedIndexWriter::new(storage, InvertedIndexWriterConfig::default()).unwrap();

    let doc = Document::builder()
        .add_field("popularity", DataValue::Int64(42))
        .add_field("body", DataValue::Text("alpha".into()))
        .build();
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    let reader = writer.build_reader().unwrap();

    assert!(
        reader.has_doc_values("popularity"),
        "doc values exist on disk — a fresh reader must report them"
    );
    assert!(!reader.has_doc_values("no_such_field"));
}
