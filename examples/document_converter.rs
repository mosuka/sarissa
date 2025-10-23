//! Example demonstrating DocumentConverter usage.
//!
//! This example shows how to use DocumentConverter to create documents from strings,
//! which can then be analyzed with DocumentParser (similar to QueryParser).

use tempfile::TempDir;

use sage::document::converter::{
    DocumentConverter, field_value::FieldValueDocumentConverter, json::JsonDocumentConverter,
};
use sage::error::Result;
use sage::lexical::engine::LexicalEngine;
use sage::lexical::inverted_index::IndexConfig;
use sage::lexical::types::SearchRequest;

fn main() -> Result<()> {
    println!("=== DocumentConverter Example ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}\n", temp_dir.path());

    // Create search engine first
    let mut engine = LexicalEngine::create_in_dir(temp_dir.path(), IndexConfig::default())?;

    println!("=== Example 1: Field:Value Format ===\n");

    // Create field:value document converter
    let converter = FieldValueDocumentConverter::new();

    // Parse documents from field:value format
    let doc1_text = r#"
id:PROD-001
title:Wireless Headphones
category:electronics
description:High-quality wireless headphones with noise cancellation
price:199.99
in_stock:true
    "#;

    let doc1 = converter.convert(doc1_text)?;
    println!("Converted document 1:");
    println!("  Fields: {:?}", doc1.field_names());
    engine.add_document(doc1)?;

    let doc2_text = r#"
id:BOOK-002
title:Rust Programming Language
category:books
description:The official Rust book for learning Rust
price:39.99
pages:552
in_stock:true
    "#;

    let doc2 = converter.convert(doc2_text)?;
    println!("\nConverted document 2:");
    println!("  Fields: {:?}", doc2.field_names());
    engine.add_document(doc2)?;

    let doc3_text = r#"
id:SOFT-003
title:Sage Search Engine
category:software
description:Fast and flexible full-text search engine in Rust
price:0.0
in_stock:true
    "#;

    let doc3 = converter.convert(doc3_text)?;
    println!("\nConverted document 3:");
    println!("  Fields: {:?}", doc3.field_names());
    engine.add_document(doc3)?;

    engine.commit()?;
    println!("\n✓ All documents indexed successfully\n");

    println!("=== Example 2: JSON Format ===\n");

    // Create JSON document converter
    let json_converter = JsonDocumentConverter::new();

    let json_doc = r#"{
        "id": "GAME-004",
        "title": "Puzzle Game Collection",
        "category": "games",
        "description": "A collection of classic puzzle games",
        "price": 29.99,
        "in_stock": false
    }"#;

    let doc4 = json_converter.convert(json_doc)?;
    println!("Converted JSON document:");
    println!("  Fields: {:?}", doc4.field_names());
    engine.add_document(doc4)?;
    engine.commit()?;

    println!("\n=== Example 3: Search Parsed Documents ===\n");

    // Search by category (using KeywordAnalyzer - exact match)
    println!("Search: category:electronics");
    let query_parser = sage::query::parser::QueryParser::new();
    let query = query_parser.parse_field("category", "electronics")?;
    let results = engine.search(SearchRequest::new(query).load_documents(true))?;

    println!("Found {} results:", results.total_hits);
    for hit in results.hits.iter() {
        if let Some(doc) = &hit.document
            && let Some(title) = doc.get_field("title").and_then(|f| f.as_text())
        {
            println!("  - {title}");
        }
    }

    // Search in description
    println!("\nSearch: description:rust");
    let query = query_parser.parse_field("description", "rust")?;
    let results = engine.search(SearchRequest::new(query).load_documents(true))?;

    println!("Found {} results:", results.total_hits);
    for hit in results.hits.iter() {
        if let Some(doc) = &hit.document
            && let Some(title) = doc.get_field("title").and_then(|f| f.as_text())
        {
            println!("  - {title}");
        }
    }

    println!("\n=== Example 4: Type Inference ===\n");

    let type_test = r#"
name:Test Document
count:42
rating:4.5
active:true
inactive:false
    "#;

    let type_doc = converter.convert(type_test)?;
    println!("Type inference demonstration:");
    println!("  name (text): {:?}", type_doc.get_field("name"));
    println!("  count (integer): {:?}", type_doc.get_field("count"));
    println!("  rating (float): {:?}", type_doc.get_field("rating"));
    println!("  active (boolean): {:?}", type_doc.get_field("active"));
    println!("  inactive (boolean): {:?}", type_doc.get_field("inactive"));

    println!("\n=== Symmetry with QueryParser ===\n");
    println!("Document side:");
    println!("  String → DocumentConverter → Document → DocumentParser → AnalyzedDocument");
    println!("\nQuery side:");
    println!("  String → QueryParser → Query");
    println!("\n✓ Both parsers use PerFieldAnalyzer for consistent analysis!");

    Ok(())
}
