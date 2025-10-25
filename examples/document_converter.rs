//! Example demonstrating DocumentConverter usage.
//!
//! This example shows how to use DocumentConverter to create documents from files,
//! supporting both CSV and JSONL formats with various features including GeoPoint support.

use std::io::Write;
use tempfile::{NamedTempFile, TempDir};

use sage::document::converter::{
    csv::CsvDocumentConverter, jsonl::JsonlDocumentConverter, DocumentConverter,
};
use sage::document::field_value::FieldValue;
use sage::error::Result;
use sage::lexical::engine::LexicalEngine;
use sage::lexical::index::IndexConfig;

fn main() -> Result<()> {
    println!("=== DocumentConverter Example ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}\n", temp_dir.path());

    // Create search engine
    let mut engine = LexicalEngine::create_in_dir(temp_dir.path(), IndexConfig::default())?;

    // ===================================================================
    // Example 1: JSONL Format with nested GeoPoint objects
    // ===================================================================
    println!("=== Example 1: JSONL Format with GeoPoint ===\n");

    let mut jsonl_file = NamedTempFile::new().unwrap();
    writeln!(
        jsonl_file,
        r#"{{"id": "LOC-001", "title": "Tokyo Tower", "city": "Tokyo", "location": {{"lat": 35.6762, "lon": 139.6503}}}}"#
    )
    .unwrap();
    writeln!(
        jsonl_file,
        r#"{{"id": "LOC-002", "title": "Eiffel Tower", "city": "Paris", "location": {{"lat": 48.8584, "lon": 2.2945}}}}"#
    )
    .unwrap();
    writeln!(
        jsonl_file,
        r#"{{"id": "LOC-003", "title": "Statue of Liberty", "city": "New York", "location": {{"lat": 40.6892, "lon": -74.0445}}}}"#
    )
    .unwrap();
    jsonl_file.flush().unwrap();

    let jsonl_converter = JsonlDocumentConverter::new();

    let mut doc_count = 0;
    for result in jsonl_converter.convert(jsonl_file.path())? {
        let doc = result?;
        doc_count += 1;

        if let Some(title) = doc.get_field("title").and_then(|f| f.as_text()) {
            print!("  Loaded: {}", title);
            if let Some(FieldValue::Geo(geo)) = doc.get_field("location") {
                println!(" at ({:.4}, {:.4})", geo.lat, geo.lon);
            } else {
                println!();
            }
        }

        engine.add_document(doc)?;
    }

    engine.commit()?;
    println!("✓ {} documents from JSONL indexed\n", doc_count);

    // ===================================================================
    // Example 2: CSV Format with dot notation for GeoPoint
    // ===================================================================
    println!("=== Example 2: CSV Format with location.lat/lon ===\n");

    let mut csv_file = NamedTempFile::new().unwrap();
    writeln!(csv_file, "id,name,city,location.lat,location.lon").unwrap();
    writeln!(csv_file, "RES-001,Sushi Restaurant,Tokyo,35.6812,139.7671").unwrap();
    writeln!(csv_file, "RES-002,Pizza Place,New York,40.7580,-73.9855").unwrap();
    writeln!(csv_file, "RES-003,Cafe de Paris,Paris,48.8606,2.3376").unwrap();
    csv_file.flush().unwrap();

    let csv_converter = CsvDocumentConverter::new();

    doc_count = 0;
    for result in csv_converter.convert(csv_file.path())? {
        let doc = result?;
        doc_count += 1;

        if let Some(name) = doc.get_field("name").and_then(|f| f.as_text()) {
            print!("  Loaded: {}", name);
            if let Some(FieldValue::Geo(geo)) = doc.get_field("location") {
                println!(" at ({:.4}, {:.4})", geo.lat, geo.lon);
            } else {
                println!();
            }
        }

        engine.add_document(doc)?;
    }

    engine.commit()?;
    println!("✓ {} documents from CSV indexed\n", doc_count);

    // ===================================================================
    // Example 3: Type inference demonstration
    // ===================================================================
    println!("=== Example 3: Type Inference ===\n");

    let mut types_file = NamedTempFile::new().unwrap();
    writeln!(types_file, "name,count,price,active,rating").unwrap();
    writeln!(types_file, "Product A,42,19.99,true,4.5").unwrap();
    writeln!(types_file, "Product B,100,29.99,false,4.8").unwrap();
    csv_file.flush().unwrap();

    doc_count = 0;
    for result in csv_converter.convert(types_file.path())? {
        let doc = result?;
        doc_count += 1;

        if doc_count == 1 {
            // Show types for first document
            println!("  Field types detected:");
            if let Some(name) = doc.get_field("name") {
                println!("    name: {:?} (Text)", name);
            }
            if let Some(count) = doc.get_field("count") {
                println!("    count: {:?} (Integer)", count);
            }
            if let Some(price) = doc.get_field("price") {
                println!("    price: {:?} (Float)", price);
            }
            if let Some(active) = doc.get_field("active") {
                println!("    active: {:?} (Boolean)", active);
            }
            if let Some(rating) = doc.get_field("rating") {
                println!("    rating: {:?} (Float)", rating);
            }
        }

        engine.add_document(doc)?;
    }

    engine.commit()?;
    println!("\n✓ {} documents with type inference indexed\n", doc_count);

    // ===================================================================
    // Example 4: Using resources files (if they exist)
    // ===================================================================
    println!("=== Example 4: Loading from resources/ (if available) ===\n");

    if std::path::Path::new("resources/documents.csv").exists() {
        println!("Loading from resources/documents.csv...");
        let mut count = 0;
        for result in csv_converter.convert("resources/documents.csv")? {
            let doc = result?;
            count += 1;
            if count <= 3 {
                if let Some(title) = doc.get_field("title").and_then(|f| f.as_text()) {
                    print!("  {}: {}", count, title);
                    if let Some(city) = doc.get_field("city").and_then(|f| f.as_text()) {
                        print!(" ({})", city);
                    }
                    if let Some(FieldValue::Geo(geo)) = doc.get_field("location") {
                        print!(" [{:.2}, {:.2}]", geo.lat, geo.lon);
                    }
                    println!();
                }
            }
        }
        println!("  ... (loaded {} total documents)", count);
    } else {
        println!("  resources/documents.csv not found (skipping)");
    }

    if std::path::Path::new("resources/documents.jsonl").exists() {
        println!("\nLoading from resources/documents.jsonl...");
        let mut count = 0;
        for result in jsonl_converter.convert("resources/documents.jsonl")? {
            let doc = result?;
            count += 1;
            if count <= 3 {
                if let Some(title) = doc.get_field("title").and_then(|f| f.as_text()) {
                    print!("  {}: {}", count, title);
                    if let Some(city) = doc.get_field("city").and_then(|f| f.as_text()) {
                        print!(" ({})", city);
                    }
                    if let Some(FieldValue::Geo(geo)) = doc.get_field("location") {
                        print!(" [{:.2}, {:.2}]", geo.lat, geo.lon);
                    }
                    println!();
                }
            }
        }
        println!("  ... (loaded {} total documents)", count);
    } else {
        println!("  resources/documents.jsonl not found (skipping)");
    }

    // ===================================================================
    // Summary
    // ===================================================================
    println!("\n=== Summary ===\n");
    println!("DocumentConverter features:");
    println!("  ✓ JSONL format: One JSON object per line");
    println!("  ✓ CSV format: First row as header, subsequent rows as data");
    println!("  ✓ Automatic type inference (Integer, Float, Boolean, Text)");
    println!("  ✓ GeoPoint support:");
    println!("    - JSONL: Nested {{\"lat\": 35.6, \"lon\": 139.7}} objects");
    println!("    - CSV: Dot notation (location.lat, location.lon columns)");
    println!("  ✓ Iterator-based streaming for memory efficiency");
    println!("  ✓ Multiple GeoPoint fields per document supported");

    Ok(())
}
