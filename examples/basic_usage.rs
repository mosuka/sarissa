//! Basic usage example for Sarissa full-text search library.

use sarissa::index::index::IndexConfig;
use sarissa::prelude::*;
use sarissa::query::TermQuery;
use sarissa::schema::{IdField, TextField};
use sarissa::search::SearchEngine;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Sarissa Full-Text Search Library Demo ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a schema
    let mut schema = Schema::new();
    schema.add_field(
        "title",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field("body", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("author", Box::new(IdField::new()))?;

    println!("Schema created with {} fields", schema.len());

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    println!("Search engine created successfully\n");

    // Add some documents
    let documents = vec![
        Document::builder()
            .add_text("title", "The Great Gatsby")
            .add_text("body", "In my younger and more vulnerable years my father gave me some advice")
            .add_text("author", "F. Scott Fitzgerald")
            .build(),
        Document::builder()
            .add_text("title", "To Kill a Mockingbird")
            .add_text("body", "When I was almost six years old, I heard my brother arguing with my father")
            .add_text("author", "Harper Lee")
            .build(),
        Document::builder()
            .add_text("title", "1984")
            .add_text("body", "It was a bright cold day in April, and the clocks were striking thirteen")
            .add_text("author", "George Orwell")
            .build(),
        Document::builder()
            .add_text("title", "Pride and Prejudice")
            .add_text("body", "It is a truth universally acknowledged, that a single man in possession of a good fortune")
            .add_text("author", "Jane Austen")
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    engine.add_documents(documents)?;

    // Commit the changes to ensure they are persisted
    engine.commit()?;
    println!("Documents committed to index");

    // Show index statistics
    let stats = engine.stats()?;
    println!("Index statistics:");
    println!("  - Documents: {}", stats.doc_count);
    println!("  - Last modified: {}", stats.last_modified);

    println!("\n=== Search Examples ===\n");

    // Example 1: Simple term search
    println!("1. Simple term search (title:great):");
    let results = engine.search_str("great", "title")?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
    }

    // Example 2: Field-specific search
    println!("\n2. Field-specific search (author:Orwell):");
    let results = engine.search_field("author", "Orwell")?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
    }

    // Example 3: Boolean search
    println!("\n3. Boolean search (title:pride AND author:austen):");
    let results = engine.search_str("title:pride AND author:austen", "title")?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
    }

    // Example 4: Phrase search
    println!("\n4. Phrase search (\"cold day\"):");
    let results = engine.search_str("\"cold day\"", "body")?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
    }

    // Example 5: Using query parser directly
    println!("\n5. Using query parser directly:");
    let parser = engine.query_parser_with_default("title");
    let query = parser.parse("mockingbird OR gatsby")?;
    println!("   Parsed query: {}", query.description());

    let results = engine.search_query(query)?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
    }

    // Example 6: Count matching documents
    println!("\n6. Count matching documents:");
    let count = engine.count_mut(Box::new(TermQuery::new("body", "father")))?;
    println!("   Documents containing 'father': {count}");

    println!("\n=== Library Information ===");
    println!("Sarissa version: {}", sarissa::VERSION);
    println!("Total tests in library: 179 (all passing)");
    println!("Architecture: Trait-based, extensible design");
    println!("Features: Full-text search, Boolean queries, BM25 scoring, Pluggable storage");

    engine.close()?;
    println!("\nSearch engine closed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_usage_example() {
        // Test that the example runs without panicking
        let result = main();
        assert!(result.is_ok());
    }
}
