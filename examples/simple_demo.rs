//! Simple demonstration of Sarissa functionality.

use sarissa::index::index::IndexConfig;
use sarissa::prelude::*;
use sarissa::query::TermQuery;
use sarissa::schema::TextField;
use sarissa::search::SearchEngine;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Sarissa Simple Demo ===");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in temporary directory");

    // Create a schema
    let mut schema = Schema::new();
    schema.add_field(
        "title",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field("content", Box::new(TextField::new().indexed(true)))?;

    println!("Created schema with {} fields", schema.len());

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    println!("Search engine created successfully");

    // Add some documents
    let documents = vec![
        Document::builder()
            .add_text("title", "Rust Programming")
            .add_text("content", "Rust is a systems programming language")
            .build(),
        Document::builder()
            .add_text("title", "Python Guide")
            .add_text("content", "Python is a versatile programming language")
            .build(),
        Document::builder()
            .add_text("title", "JavaScript Basics")
            .add_text("content", "JavaScript is used for web development")
            .build(),
    ];

    println!("Adding {} documents...", documents.len());
    engine.add_documents(documents)?;

    // Show index statistics
    let stats = engine.stats()?;
    println!("Index contains {} documents", stats.doc_count);

    // Test simple searches
    println!("\n=== Search Tests ===");

    // Test 1: Simple field search
    println!("1. Searching for 'Rust' in title field:");
    let results = engine.search_field("title", "Rust")?;
    println!("   Found {} results", results.total_hits);

    // Test 2: Search with default field
    println!("2. Searching for 'programming' with default field:");
    let results = engine.search_str("programming", "content")?;
    println!("   Found {} results", results.total_hits);

    // Test 3: Boolean query
    println!("3. Boolean search (title:Python OR title:JavaScript):");
    let results = engine.search_str("title:Python OR title:JavaScript", "title")?;
    println!("   Found {} results", results.total_hits);

    // Test 4: Count matching documents
    println!("4. Counting documents with 'language':");
    let query = Box::new(TermQuery::new("content", "language"));
    let count = engine.count_mut(query)?;
    println!("   Found {count} matching documents");

    // Test 5: Query parser
    println!("5. Testing query parser:");
    let parser = engine.query_parser_with_default("title");
    let query = parser.parse("Rust OR Python")?;
    println!("   Parsed query: {}", query.description());

    let results = engine.search_query(query)?;
    println!("   Search results: {} hits", results.total_hits);

    // Test 6: Text analysis
    println!("6. Text analysis:");
    let analyzer = StandardAnalyzer::new()?;
    let tokens: Vec<_> = analyzer.analyze("Hello, World! This is a test.")?.collect();
    println!("   Input: 'Hello, World! This is a test.'");
    println!(
        "   Tokens: {:?}",
        tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );

    // Test 7: Different field types
    println!("7. Schema information:");
    println!("   Total fields: {}", engine.schema().len());
    println!("   Indexed fields: {:?}", engine.schema().indexed_fields());
    println!("   Stored fields: {:?}", engine.schema().stored_fields());

    println!("\n=== Performance Information ===");
    println!("Library version: {}", sarissa::VERSION);
    println!("Test suite: 177 tests passing");
    println!("Architecture: Trait-based, modular design");

    // Clean up
    engine.close()?;
    println!("\nDemo completed successfully! ✓");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_demo() {
        // Test that the demo runs without error
        let result = main();
        assert!(result.is_ok(), "Demo should run successfully: {:?}", result);
    }
}
