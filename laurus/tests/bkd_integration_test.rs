use chrono::{TimeZone, Utc};
use laurus::lexical::NumericRangeQuery;
use laurus::lexical::NumericType;
use laurus::lexical::Query;
use laurus::lexical::{GeoDistanceQuery, GeoPoint};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};
use std::sync::Arc;

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

#[test]
fn test_bkd_file_creation_and_query() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut writer = index_writer(storage.clone());

    // Doc 1: age=30, score=95.5
    let doc1 = Document::builder()
        .add_field("age", DataValue::Int64(30))
        .add_field("score", DataValue::Float64(95.5))
        .add_field(
            "created_at",
            DataValue::DateTime(Utc.timestamp_opt(1600000000, 0).unwrap()),
        )
        .add_field("description", DataValue::Text("User profile 1".into()))
        .build();
    writer.add_document(doc1).unwrap();

    // Doc 2: age=20, score=80.0
    let doc2 = Document::builder()
        .add_field("age", DataValue::Int64(20))
        .add_field("score", DataValue::Float64(80.0))
        .add_field(
            "created_at",
            DataValue::DateTime(Utc.timestamp_opt(1500000000, 0).unwrap()),
        )
        .add_field("description", DataValue::Text("User profile 2".into()))
        .build();
    writer.add_document(doc2).unwrap();

    // Doc 3: age=40, score=100.0
    let doc3 = Document::builder()
        .add_field("age", DataValue::Int64(40))
        .add_field("score", DataValue::Float64(100.0))
        .add_field(
            "created_at",
            DataValue::DateTime(Utc.timestamp_opt(1700000000, 0).unwrap()),
        )
        .add_field("description", DataValue::Text("User profile 3".into()))
        .build();
    writer.add_document(doc3).unwrap();

    // Commit to flush segment
    writer.commit().unwrap();

    // Verify files existed
    let age_bkd = "segment_000000.age.bkd";
    assert!(
        storage.file_exists(age_bkd),
        "BKD file for age should exist"
    );

    // Open Reader
    let reader = writer.build_reader().unwrap();

    // Query 1: Age [25, 35] -> Should match Doc 1 (age 30) -> ID 0
    let query_age = NumericRangeQuery::new(
        "age",
        NumericType::Integer,
        Some(25.0),
        Some(35.0),
        true,
        true,
    );

    let matched_age = collect_matcher_results(query_age.matcher(&*reader).unwrap());

    assert_eq!(matched_age, vec![0]);

    // Query 2: Score >= 90.0 -> Doc 1 (95.5), Doc 3 (100.0) -> IDs 0, 2
    let query_score =
        NumericRangeQuery::new("score", NumericType::Float, Some(90.0), None, true, true);
    let matched_score = collect_matcher_results(query_score.matcher(&*reader).unwrap());

    assert_eq!(matched_score, vec![0, 2]);

    // Query 3: Created At < 1600000000 -> Doc 2 (1500000000) -> ID 1
    let query_date = NumericRangeQuery::new(
        "created_at",
        NumericType::Integer,
        None,
        Some(1600000000.0),
        false,
        false,
    );
    let matched_date = collect_matcher_results(query_date.matcher(&*reader).unwrap());

    assert_eq!(matched_date, vec![1]);
}

#[test]
fn test_geo_bkd_query() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut writer = index_writer(storage.clone());

    // Tokyo: 35.6812, 139.7671
    let tokyo = GeoPoint::new(35.6812, 139.7671);
    // Yokohama: 35.4437, 139.6380
    let yokohama = GeoPoint::new(35.4437, 139.6380);
    // Osaka: 34.6937, 135.5023
    let osaka = GeoPoint::new(34.6937, 135.5023);

    writer
        .add_document(
            Document::builder()
                .add_field("location", DataValue::Geo(tokyo))
                .add_field("city", DataValue::Text("Tokyo".into()))
                .build(),
        )
        .unwrap();

    writer
        .add_document(
            Document::builder()
                .add_field("location", DataValue::Geo(yokohama))
                .add_field("city", DataValue::Text("Yokohama".into()))
                .build(),
        )
        .unwrap();

    writer
        .add_document(
            Document::builder()
                .add_field("location", DataValue::Geo(osaka))
                .add_field("city", DataValue::Text("Osaka".into()))
                .build(),
        )
        .unwrap();

    writer.commit().unwrap();

    // Verify BKD file existed
    assert!(storage.file_exists("segment_000000.location.bkd"));

    let reader = writer.build_reader().unwrap();

    // Distance query: Near Tokyo (within 50 km) -> Should match Tokyo (0 km) and Yokohama (~30 km)
    let query = GeoDistanceQuery::new("location", tokyo, 50_000.0);
    let matched_docs = collect_matcher_results(query.matcher(&*reader).unwrap());
    assert_eq!(matched_docs, vec![0, 1]);

    // Near Osaka (within 20 km) -> Should match Osaka
    let query_osaka = GeoDistanceQuery::new("location", osaka, 20_000.0);
    let matched_osaka = collect_matcher_results(query_osaka.matcher(&*reader).unwrap());
    assert_eq!(matched_osaka, vec![2]);
}

fn collect_matcher_results(mut m: Box<dyn laurus::lexical::query::matcher::Matcher>) -> Vec<u64> {
    let mut docs = Vec::new();
    while !m.is_exhausted() {
        let doc_id = m.doc_id();
        if doc_id == u64::MAX {
            break;
        }
        docs.push(doc_id);
        if !m.next().unwrap() {
            break;
        }
    }
    docs
}
