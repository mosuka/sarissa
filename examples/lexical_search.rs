//! Lexical search example demonstrating Platypus's keyword tooling.
//!
//! This walkthrough exercises every built-in query type exposed by the
//! `lexical::index::inverted::query` module, including:
//! - TermQuery: Simple term searches
//! - PhraseQuery: Exact phrase matching
//! - BooleanQuery: Complex boolean combinations (AND, OR, NOT)
//! - NumericRangeQuery: Range queries on numeric fields
//! - FuzzyQuery: Approximate string matching with edit distance
//! - WildcardQuery: Pattern matching with * and ?
//! - GeoDistanceQuery: Geographic proximity searches
//! - GeoBoundingBoxQuery: Geographic bounding box searches
//!
//! Field Sorting Features:
//! - Sort by field values (ascending/descending)
//! - DocValues: Column-oriented storage for efficient sorting
//! - TopFieldCollector: Lucene-style collection-time sorting
//! - Supports all field types: Text, Integer, Float, Boolean, DateTime, etc.

use std::sync::Arc;

use tempfile::TempDir;

use platypus::analysis::analyzer::analyzer::Analyzer;
use platypus::analysis::analyzer::keyword::KeywordAnalyzer;
use platypus::analysis::analyzer::per_field::PerFieldAnalyzer;
use platypus::analysis::analyzer::standard::StandardAnalyzer;
use platypus::document::converter::DocumentConverter;
use platypus::document::converter::jsonl::JsonlDocumentConverter;
use platypus::document::field::FieldValue;
use platypus::error::Result;
use platypus::lexical::engine::LexicalEngine;
use platypus::lexical::index::config::{InvertedIndexConfig, LexicalIndexConfig};
use platypus::lexical::index::factory::LexicalIndexFactory;
use platypus::lexical::index::inverted::query::Query;
use platypus::lexical::index::inverted::query::boolean::BooleanQuery;
use platypus::lexical::index::inverted::query::fuzzy::FuzzyQuery;
use platypus::lexical::index::inverted::query::geo::{
    GeoBoundingBox, GeoBoundingBoxQuery, GeoDistanceQuery, GeoPoint,
};
use platypus::lexical::index::inverted::query::phrase::PhraseQuery;
use platypus::lexical::index::inverted::query::range::NumericRangeQuery;
use platypus::lexical::index::inverted::query::term::TermQuery;
use platypus::lexical::index::inverted::query::wildcard::WildcardQuery;
use platypus::lexical::search::searcher::LexicalSearchRequest;
use platypus::storage::file::FileStorageConfig;
use platypus::storage::{StorageConfig, StorageFactory};

fn main() -> Result<()> {
    println!("=== Comprehensive Lexical Search Example ===\n");

    // Step 1: Create a lexical search index
    println!("Step 1: Create a lexical search index...");

    // Create a storage backend
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    // Create an analyzer
    let standard_analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);
    let keyword_analyzer: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
    let mut per_field_analyzer = PerFieldAnalyzer::new(Arc::clone(&standard_analyzer));
    per_field_analyzer.add_analyzer("id", Arc::clone(&keyword_analyzer));

    // Create a lexical index
    let lexical_index_config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer: Arc::new(per_field_analyzer.clone()),
        ..InvertedIndexConfig::default()
    });
    let lexical_index = LexicalIndexFactory::create(storage, lexical_index_config)?;

    // Create a lexical engine
    let mut lexical_engine = LexicalEngine::new(lexical_index)?;
    println!();

    // Step 2: Load documents from JSONL file
    println!("Step 2: Loading documents from resources/documents.jsonl...");
    let converter = JsonlDocumentConverter::new();
    let doc_iter = converter.convert("resources/documents.jsonl")?;

    // Add documents one by one using iterator (memory efficient)
    // Note: The engine caches the writer, so calling add_document() repeatedly
    // is efficient and doesn't create a new writer each time
    let mut count = 0;
    for doc_result in doc_iter {
        lexical_engine.add_document(doc_result?)?;
        count += 1;
    }
    println!("  Added {} documents from JSONL file", count);

    // Commit the changes to make them searchable
    lexical_engine.commit()?;
    println!("  Documents committed to index\n");

    // Step 3: Demonstrate all query types
    println!("Step 3: Demonstrating ALL query types...\n");
    println!("{}", "=".repeat(80));

    // 1. TermQuery - Simple term search
    println!("\n[1] TermQuery - Simple term search");
    println!("{}", "-".repeat(80));
    println!("Description: Searches for exact term matches in a specific field");
    println!("\nExample: Search for 'rust' in title field");
    let query = TermQuery::new("title", "rust");
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!("\nResults: Found {} matching documents", results.total_hits);
    display_results(&results);

    // 2. PhraseQuery - Exact phrase matching
    println!("\n[2] PhraseQuery - Exact phrase matching");
    println!("{}", "-".repeat(80));
    println!("Description: Searches for exact phrase (words in specific order)");
    println!("\nExample: Search for phrase 'machine learning' in body");
    let query = PhraseQuery::new("body", vec!["machine".to_string(), "learning".to_string()]);
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!("\nResults: Found {} matching documents", results.total_hits);
    display_results(&results);

    // 3. BooleanQuery - AND condition
    println!("\n[3] BooleanQuery (AND) - Boolean combination with AND");
    println!("{}", "-".repeat(80));
    println!("Description: Combines multiple queries with boolean logic (AND)");
    println!("\nExample: Documents with both 'rust' AND 'programming'");
    let mut query = BooleanQuery::new();
    query.add_must(Box::new(TermQuery::new("body", "rust")));
    query.add_must(Box::new(TermQuery::new("body", "programming")));
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!("\nResults: Found {} matching documents", results.total_hits);
    display_results(&results);

    // 4. BooleanQuery - OR condition
    println!("\n[4] BooleanQuery (OR) - Boolean combination with OR");
    println!("{}", "-".repeat(80));
    println!("Description: Combines multiple queries with boolean logic (OR)");
    println!("\nExample: Documents with 'python' OR 'javascript'");
    let mut query = BooleanQuery::new();
    query.add_should(Box::new(TermQuery::new("body", "python")));
    query.add_should(Box::new(TermQuery::new("body", "javascript")));
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!("\nResults: Found {} matching documents", results.total_hits);
    display_results(&results);

    // 5. BooleanQuery - NOT condition
    println!("\n[5] BooleanQuery (NOT) - Boolean combination with NOT");
    println!("{}", "-".repeat(80));
    println!("Description: Combines queries with exclusion logic");
    println!("\nExample: Documents with 'web' OR 'database' but NOT 'python'");
    let mut query = BooleanQuery::new();
    query.add_should(Box::new(TermQuery::new("body", "web")));
    query.add_should(Box::new(TermQuery::new("body", "database")));
    query.add_must_not(Box::new(TermQuery::new("body", "python")));
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!("\nResults: Found {} matching documents", results.total_hits);
    display_results(&results);

    // 6. NumericRangeQuery - Range query on integers
    println!("\n[6] NumericRangeQuery - Range query on numeric fields");
    println!("{}", "-".repeat(80));
    println!("Description: Searches for documents with numeric values in a range");
    println!("\nExample: Documents from year 2023 and later");
    let query = NumericRangeQuery::i64_range("year", Some(2023), None);
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!("\nResults: Found {} matching documents", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            print!("  {}. Score: {:.4} - ", i + 1, hit.score);
            if let Some(field) = doc.get_field("title")
                && let FieldValue::Text(title) = &field.value
            {
                print!("Title: {}", title);
            }
            if let Some(field) = doc.get_field("year")
                && let FieldValue::Integer(year) = &field.value
            {
                print!(" (Year: {})", year);
            }
            println!();
        }
    }

    // 7. NumericRangeQuery - Rating range
    println!("\n[7] NumericRangeQuery - Rating range query");
    println!("{}", "-".repeat(80));
    println!("Description: Find documents with rating between 4 and 5");
    println!("\nExample: Documents with rating >= 4 and <= 5");
    let query = NumericRangeQuery::i64_range("rating", Some(4), Some(5));
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!("\nResults: Found {} matching documents", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            print!("  {}. Score: {:.4} - ", i + 1, hit.score);
            if let Some(field) = doc.get_field("title")
                && let FieldValue::Text(title) = &field.value
            {
                print!("Title: {}", title);
            }
            if let Some(field) = doc.get_field("rating")
                && let FieldValue::Integer(rating) = &field.value
            {
                print!(" (Rating: {})", rating);
            }
            println!();
        }
    }

    // 8. FuzzyQuery - Approximate string matching
    println!("\n[8] FuzzyQuery - Approximate string matching (typo tolerance)");
    println!("{}", "-".repeat(80));
    println!("Description: Finds terms similar to the query (handles typos)");
    println!("\nExample: Search for 'rust' (will also match 'rast' with 1 edit distance)");
    let query = FuzzyQuery::new("tags", "rust").max_edits(2);
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!(
        "\nResults: Found {} matching documents (including typos)",
        results.total_hits
    );
    display_results(&results);

    // 9. FuzzyQuery - Another example with different term
    println!("\n[9] FuzzyQuery - Search for 'python' (will match 'pyhton')");
    println!("{}", "-".repeat(80));
    println!("Description: Demonstrates fuzzy matching with different term");
    println!("\nExample: Fuzzy search for 'python' in tags");
    let query = FuzzyQuery::new("tags", "python").max_edits(2);
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!("\nResults: Found {} matching documents", results.total_hits);
    display_results(&results);

    // 10. WildcardQuery - Pattern matching with *
    println!("\n[10] WildcardQuery - Pattern matching with wildcards");
    println!("{}", "-".repeat(80));
    println!("Description: Matches patterns using * (any chars) and ? (one char)");
    println!("\nExample: Search for tags starting with 'web' (web*)");
    let query = WildcardQuery::new("tags", "web*")?;
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!("\nResults: Found {} matching documents", results.total_hits);
    display_results(&results);

    // 11. WildcardQuery - Pattern matching with ?
    println!("\n[11] WildcardQuery - Single character wildcard");
    println!("{}", "-".repeat(80));
    println!("Description: Using ? to match exactly one character");
    println!("\nExample: Search for 'ru?t' (matches 'rust')");
    let query = WildcardQuery::new("tags", "ru?t")?;
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!("\nResults: Found {} matching documents", results.total_hits);
    display_results(&results);

    // 12. GeoDistanceQuery - Geographic proximity search
    println!("\n[12] GeoDistanceQuery - Geographic proximity search");
    println!("{}", "-".repeat(80));
    println!("Description: Finds documents within a certain distance from a point");
    println!("\nExample: Documents within 50km of Tokyo (35.6762, 139.6503)");
    let tokyo = GeoPoint::new(35.6762, 139.6503)?;
    let query = GeoDistanceQuery::new("location", tokyo, 50.0); // 50km radius
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!(
        "\nResults: Found {} matching documents within 50km",
        results.total_hits
    );
    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            print!("  {}. Score: {:.4} - ", i + 1, hit.score);
            if let Some(field) = doc.get_field("title")
                && let FieldValue::Text(title) = &field.value
            {
                print!("Title: {}", title);
            }
            if let Some(field) = doc.get_field("location")
                && let FieldValue::Geo(geo) = &field.value
            {
                let distance = tokyo.distance_to(&GeoPoint::new(geo.lat, geo.lon)?);
                print!(" (Distance: {:.2}km)", distance);
            }
            println!();
        }
    }

    // 13. GeoDistanceQuery - Larger radius
    println!("\n[13] GeoDistanceQuery - Larger search radius");
    println!("{}", "-".repeat(80));
    println!("Description: Geographic search with larger radius");
    println!("\nExample: Documents within 2000km of Tokyo");
    let query = GeoDistanceQuery::new("location", tokyo, 2000.0); // 2000km radius
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!(
        "\nResults: Found {} matching documents within 2000km",
        results.total_hits
    );
    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            print!("  {}. Score: {:.4} - ", i + 1, hit.score);
            if let Some(field) = doc.get_field("title")
                && let FieldValue::Text(title) = &field.value
            {
                print!("Title: {}", title);
            }
            if let Some(field) = doc.get_field("location")
                && let FieldValue::Geo(geo) = &field.value
            {
                let distance = tokyo.distance_to(&GeoPoint::new(geo.lat, geo.lon)?);
                print!(" (Distance: {:.2}km)", distance);
            }
            println!();
        }
    }

    // 14. GeoBoundingBoxQuery - Rectangular geographic search
    println!("\n[14] GeoBoundingBoxQuery - Rectangular geographic area search");
    println!("{}", "-".repeat(80));
    println!("Description: Finds documents within a rectangular bounding box");
    println!("\nExample: Documents in Europe (approx bounding box)");
    let top_left = GeoPoint::new(60.0, -10.0)?; // North-West corner
    let bottom_right = GeoPoint::new(35.0, 40.0)?; // South-East corner
    let bbox = GeoBoundingBox::new(top_left, bottom_right)?;
    let query = GeoBoundingBoxQuery::new("location", bbox);
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!(
        "\nResults: Found {} matching documents in Europe",
        results.total_hits
    );
    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            print!("  {}. Score: {:.4} - ", i + 1, hit.score);
            if let Some(field) = doc.get_field("title")
                && let FieldValue::Text(title) = &field.value
            {
                print!("Title: {}", title);
            }
            if let Some(field) = doc.get_field("location")
                && let FieldValue::Geo(geo) = &field.value
            {
                print!(" (Location: {:.2}, {:.2})", geo.lat, geo.lon);
            }
            println!();
        }
    }

    // 15. Complex combined query
    println!("\n[15] Complex Combined Query - Multiple query types together");
    println!("{}", "-".repeat(80));
    println!("Description: Combines multiple query types with boolean logic");
    println!("\nExample: (fuzzy 'rust' OR 'python') AND year >= 2023 AND rating >= 4");
    let mut query = BooleanQuery::new();
    query.add_should(Box::new(FuzzyQuery::new("tags", "rust").max_edits(2)));
    query.add_should(Box::new(FuzzyQuery::new("tags", "python").max_edits(2)));
    query.add_must(Box::new(NumericRangeQuery::i64_range(
        "year",
        Some(2023),
        None,
    )));
    query.add_must(Box::new(NumericRangeQuery::i64_range(
        "rating",
        Some(4),
        None,
    )));
    let query = query.with_minimum_should_match(1); // At least one of the should clauses must match
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;
    println!("\nResults: Found {} matching documents", results.total_hits);
    display_results(&results);

    // 16. Sorting by field values using DocValues
    println!("\n[16] Field-Based Sorting - Sort by year (descending)");
    println!("{}", "-".repeat(80));
    println!("Description: Sort results by field values instead of score");
    println!("             Uses DocValues (column-oriented storage) for efficient sorting");
    println!("             Lucene-style: sorting happens during collection, not after");
    println!("\nExample: Sort all documents by year (newest first)");
    let query = TermQuery::new("category", "programming"); // Just to have a query
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>)
        .load_documents(true)
        .sort_by_field_desc("year")
        .max_docs(20);
    let results = lexical_engine.search(request)?;
    println!(
        "\nResults: Found {} matching documents (sorted by year desc)",
        results.total_hits
    );
    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            print!("  {}. ", i + 1);
            if let Some(field) = doc.get_field("title")
                && let FieldValue::Text(title) = &field.value
            {
                print!("Title: {}", title);
            }
            if let Some(field) = doc.get_field("year")
                && let FieldValue::Integer(year) = &field.value
            {
                print!(" (Year: {})", year);
            }
            println!();
        }
    }

    // 17. Sorting by numeric fields
    println!("\n[17] Field-Based Sorting - Sort by rating (ascending)");
    println!("{}", "-".repeat(80));
    println!("Description: Sort numeric field values using DocValues");
    println!("             DocValues avoid loading full documents during sorting");
    println!("\nExample: Sort documents by rating (lowest first)");
    let query = NumericRangeQuery::i64_range("year", Some(2023), None);
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>)
        .load_documents(true)
        .sort_by_field_asc("rating")
        .max_docs(20);
    let results = lexical_engine.search(request)?;
    println!(
        "\nResults: Found {} matching documents (sorted by rating asc)",
        results.total_hits
    );
    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            print!("  {}. ", i + 1);
            if let Some(field) = doc.get_field("title")
                && let FieldValue::Text(title) = &field.value
            {
                print!("Title: {}", title);
            }
            if let Some(field) = doc.get_field("rating")
                && let FieldValue::Integer(rating) = &field.value
            {
                print!(" (Rating: {})", rating);
            }
            println!();
        }
    }

    // 18. Sorting by text field (ascending)
    println!("\n[18] Field-Based Sorting - Sort by author name (ascending)");
    println!("{}", "-".repeat(80));
    println!("Description: Sort text field values alphabetically in ascending order");
    println!("             Uses DocValues for efficient column-oriented field access");
    println!("\nExample: Sort documents by author name (A to Z)");
    let query = NumericRangeQuery::i64_range("rating", Some(4), Some(5)); // Get all documents
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>)
        .load_documents(true)
        .sort_by_field_asc("author")
        .max_docs(10);
    let results = lexical_engine.search(request)?;
    println!(
        "\nResults: Found {} matching documents (sorted by author asc)",
        results.total_hits
    );
    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            print!("  {}. ", i + 1);
            if let Some(field) = doc.get_field("author")
                && let FieldValue::Text(author) = &field.value
            {
                print!("Author: {}", author);
            }
            if let Some(field) = doc.get_field("title")
                && let FieldValue::Text(title) = &field.value
            {
                print!(" - Title: {}", title);
            }
            println!();
        }
    }

    // 19. Sorting by text field (descending)
    println!("\n[19] Field-Based Sorting - Sort by author name (descending)");
    println!("{}", "-".repeat(80));
    println!("Description: Sort text field values alphabetically in descending order");
    println!("             Uses DocValues for efficient column-oriented field access");
    println!("\nExample: Sort documents by author name (Z to A)");
    let query = NumericRangeQuery::i64_range("rating", Some(4), Some(5)); // Get all documents
    println!("Query Debug Output:\n{:#?}", query);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>)
        .load_documents(true)
        .sort_by_field_desc("author")
        .max_docs(10);
    let results = lexical_engine.search(request)?;
    println!(
        "\nResults: Found {} matching documents (sorted by author desc)",
        results.total_hits
    );
    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            print!("  {}. ", i + 1);
            if let Some(field) = doc.get_field("author")
                && let FieldValue::Text(author) = &field.value
            {
                print!("Author: {}", author);
            }
            if let Some(field) = doc.get_field("title")
                && let FieldValue::Text(title) = &field.value
            {
                print!(" - Title: {}", title);
            }
            println!();
        }
    }

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("\n=== Summary of Query Types ===");
    println!("✓ TermQuery: Exact term matching");
    println!("✓ PhraseQuery: Exact phrase matching");
    println!("✓ BooleanQuery: AND/OR/NOT combinations");
    println!("✓ NumericRangeQuery: Numeric range searches");
    println!("✓ FuzzyQuery: Approximate matching (typo tolerance)");
    println!("✓ WildcardQuery: Pattern matching with * and ?");
    println!("✓ GeoDistanceQuery: Geographic proximity search");
    println!("✓ GeoBoundingBoxQuery: Geographic bounding box search");
    println!("\n=== Summary of Sorting Features ===");
    println!("✓ Sort by score (default): Relevance-based ranking");
    println!("✓ Sort by field (ascending): Lowest to highest values");
    println!("✓ Sort by field (descending): Highest to lowest values");
    println!("✓ Supports text, numeric, and other field types");
    println!("\n=== DocValues Implementation ===");
    println!("✓ Column-oriented storage for efficient field access");
    println!("✓ Lucene-style: sorting happens during collection");
    println!("✓ TopFieldCollector with BinaryHeap priority queue");
    println!("✓ Required for field-based sorting operations");
    println!("\nAll query types and sorting features demonstrated successfully!");

    // Clean up
    lexical_engine.close()?;
    println!("\nFull-text search example completed successfully!");

    Ok(())
}

/// Helper function to display search results in a formatted way
fn display_results(results: &platypus::lexical::index::inverted::query::SearchResults) {
    for (i, hit) in results.hits.iter().enumerate() {
        println!("  {}. Score: {:.4}", i + 1, hit.score);

        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("id")
                && let Some(id) = field_value.value.as_text()
            {
                println!("     ID: {}", id);
            }

            if let Some(field_value) = doc.get_field("title")
                && let Some(title) = field_value.value.as_text()
            {
                println!("     Title: {}", title);
            }

            if let Some(field_value) = doc.get_field("author")
                && let Some(author) = field_value.value.as_text()
            {
                println!("     Author: {}", author);
            }

            if let Some(field_value) = doc.get_field("category")
                && let Some(category) = field_value.value.as_text()
            {
                println!("     Category: {}", category);
            }

            if let Some(field_value) = doc.get_field("body")
                && let Some(body) = field_value.value.as_text()
            {
                // Display first 100 characters of body
                let preview = if body.len() > 100 {
                    format!("{}...", &body[..100])
                } else {
                    body.to_string()
                };
                println!("     Body: {}", preview);
            }
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_text_search_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
