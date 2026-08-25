//! Integration tests for issue #943 — DocValues availability and
//! freshness at the reader level.

use std::sync::Arc;

use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};

/// A writer registered with a real index (#1024): a standalone
/// `InvertedIndexWriter` is ephemeral — its segments enter no manifest and
/// `build_reader` sees nothing — so durable fixtures go through
/// `InvertedIndex::create` + `writer()`.
fn index_writer(
    storage: Arc<dyn laurus::storage::Storage>,
) -> Box<dyn laurus::lexical::writer::LexicalIndexWriter> {
    let index =
        laurus::lexical::index::inverted::InvertedIndex::create(storage, Default::default())
            .unwrap();
    use laurus::lexical::index::LexicalIndex;
    index.writer().unwrap()
}

/// #943: `has_doc_values` must answer correctly as the very first
/// operation on a fresh reader — it used to report `false` until some
/// other call happened to load the DocValues cache.
#[test]
fn has_doc_values_is_correct_on_a_fresh_reader() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut writer = index_writer(storage);

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
