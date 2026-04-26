//! End-to-end integration test for 3D ECEF geo field handling in the
//! inverted index writer/reader pair.
//!
//! Covers the full path that #299 wires up:
//! - `DataValue::GeoEcef` ingested via `add_geo_ecef` flows through the
//!   document parser into `point_values` of length 3.
//! - `BKDWriter` materializes a 3D BKD because the points are 3-element.
//! - The stored-field deserializer reads tag 12 back as
//!   `FieldValue::GeoEcef`.
//! - A `range_search` over the BKD returns the expected doc ids and the
//!   round-tripped `GeoEcefPoint` values match what was indexed.
//!
//! Sample points are deliberately spread across an order-of-magnitude
//! range on each axis so a narrow query box exercises the BKD's pruning
//! paths and confirms that all three dimensions are honoured.

use laurus::lexical::LexicalIndexWriter;
use laurus::lexical::{InvertedIndexWriter, InvertedIndexWriterConfig};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document, GeoEcefPoint};
use std::sync::Arc;

#[test]
fn geo3d_round_trip_through_writer_and_reader() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = InvertedIndexWriterConfig {
        max_buffered_docs: 10,
        ..Default::default()
    };
    let mut writer = InvertedIndexWriter::new(storage.clone(), config).unwrap();

    // Three ECEF points, picked so that the per-axis ranges differ by an
    // order of magnitude (this is what makes widest-axis splitting
    // observable, and what would expose any 2D-vs-3D dim mismatch).
    let p_a = GeoEcefPoint::new(1_000_000.0, 2_000_000.0, 3_000_000.0);
    let p_b = GeoEcefPoint::new(1_000_500.0, 2_005_000.0, 3_050_000.0);
    let p_c = GeoEcefPoint::new(2_000_000.0, 4_000_000.0, 6_000_000.0);

    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_a))
                .add_field("name", DataValue::Text("A".into()))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_b))
                .add_field("name", DataValue::Text("B".into()))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_c))
                .add_field("name", DataValue::Text("C".into()))
                .build(),
        )
        .unwrap();

    writer.commit().unwrap();

    // The per-field BKD is named after the segment + field name. Confirm
    // it landed on disk before going further; if we miss this the next
    // assertion will fail with a less obvious message.
    assert!(
        storage.file_exists("segment_000000.position.bkd"),
        "expected the writer to materialize a 3D BKD for the position field"
    );

    let reader = writer.build_reader().unwrap();

    // Stored-field round-trip: docs come back with their GeoEcef intact.
    // This exercises the new tag-12 path on the reader side.
    let doc_a = reader.document(0).unwrap().expect("doc 0 must exist");
    let doc_b = reader.document(1).unwrap().expect("doc 1 must exist");
    let doc_c = reader.document(2).unwrap().expect("doc 2 must exist");
    assert_eq!(
        doc_a.get("position").and_then(|v| v.as_geo_ecef()),
        Some(p_a)
    );
    assert_eq!(
        doc_b.get("position").and_then(|v| v.as_geo_ecef()),
        Some(p_b)
    );
    assert_eq!(
        doc_c.get("position").and_then(|v| v.as_geo_ecef()),
        Some(p_c)
    );

    // BKD round-trip: a narrow 3D range that should match only A and B
    // (their x/y/z all sit in the lower half of every axis).
    let bkd_tree = reader
        .get_bkd_tree("position")
        .unwrap()
        .expect("position field must have a BKD tree");
    let mut hits = bkd_tree
        .range_search(
            &[Some(0.0), Some(0.0), Some(0.0)],
            &[Some(1_500_000.0), Some(2_500_000.0), Some(3_500_000.0)],
            true,
            true,
        )
        .unwrap();
    hits.sort_unstable();
    assert_eq!(hits, vec![0, 1]);

    // Wider query: include C (which lives in the upper octant).
    let mut all_hits = bkd_tree
        .range_search(
            &[Some(0.0), Some(0.0), Some(0.0)],
            &[Some(3_000_000.0), Some(5_000_000.0), Some(7_000_000.0)],
            true,
            true,
        )
        .unwrap();
    all_hits.sort_unstable();
    assert_eq!(all_hits, vec![0, 1, 2]);
}

#[test]
fn geo3d_dimension_observable_through_bkd_header() {
    use laurus::lexical::index::structures::bkd_tree::BKDReader;

    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut writer = InvertedIndexWriter::new(
        storage.clone(),
        InvertedIndexWriterConfig {
            max_buffered_docs: 10,
            ..Default::default()
        },
    )
    .unwrap();

    writer
        .add_document(
            Document::builder()
                .add_field(
                    "position",
                    DataValue::GeoEcef(GeoEcefPoint::new(1.0, 2.0, 3.0)),
                )
                .build(),
        )
        .unwrap();
    writer.commit().unwrap();

    // Pop the BKD reader off the segment directly so we can inspect the
    // dimensionality the writer chose. A 3D BKD is exactly what we want;
    // a 2D one would mean the GeoEcef branch fell back to the 2D Geo flow.
    let bkd = BKDReader::open(
        storage.clone() as Arc<dyn Storage>,
        "segment_000000.position.bkd",
    )
    .unwrap();
    let header = bkd.header();
    assert_eq!(header.num_dims, 3, "ECEF must produce a 3D BKD");
    assert_eq!(header.bytes_per_dim, 8);
    assert_eq!(header.total_point_count, 1);
}
