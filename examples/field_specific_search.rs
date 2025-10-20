//! Field-specific search example - demonstrates searching within specific fields.

use sage::analysis::analyzer::analyzer::Analyzer;
use sage::analysis::analyzer::keyword::KeywordAnalyzer;
use sage::analysis::analyzer::standard::StandardAnalyzer;
use sage::document::document::Document;
use sage::error::Result;
use sage::full_text::index::IndexConfig;
use sage::full_text::search::SearchRequest;
use sage::full_text::search::engine::SearchEngine;
use std::sync::Arc;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Field-Specific Search Example ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), IndexConfig::default())?;

    // Prepare documents
    let documents = vec![
        Document::builder()
            .add_text("title", "The Great Gatsby")
            .add_text("body", "In my younger and more vulnerable years my father gave me some advice")
            .add_text("author", "F. Scott Fitzgerald")
            .add_text("category", "classic")
            .add_integer("year", 1925)
            .add_text("tags", "american jazz-age tragedy")
            .add_text("id", "book001")
            .build(),
        Document::builder()
            .add_text("title", "To Kill a Mockingbird")
            .add_text("body", "When I was almost six years old, I heard my brother arguing with my father")
            .add_text("author", "Harper Lee")
            .add_text("category", "classic")
            .add_integer("year", 1960)
            .add_text("tags", "american southern racism")
            .add_text("id", "book002")
            .build(),
        Document::builder()
            .add_text("title", "1984")
            .add_text("body", "It was a bright cold day in April, and the clocks were striking thirteen")
            .add_text("author", "George Orwell")
            .add_text("category", "dystopian")
            .add_integer("year", 1949)
            .add_text("tags", "british totalitarian surveillance")
            .add_text("id", "book003")
            .build(),
        Document::builder()
            .add_text("title", "Animal Farm")
            .add_text("body", "Mr Jones of Manor Farm, had locked the hen houses for the night")
            .add_text("author", "George Orwell")
            .add_text("category", "satire")
            .add_integer("year", 1945)
            .add_text("tags", "british allegory political")
            .add_text("id", "book004")
            .build(),
        Document::builder()
            .add_text("title", "Pride and Prejudice")
            .add_text("body", "It is a truth universally acknowledged, that a single man in possession of a good fortune")
            .add_text("author", "Jane Austen")
            .add_text("category", "romance")
            .add_integer("year", 1813)
            .add_text("tags", "british regency society")
            .add_text("id", "book005")
            .build(),
        Document::builder()
            .add_text("title", "The Catcher in the Rye")
            .add_text("body", "If you really want to hear about it, the first thing you'll probably want to know")
            .add_text("author", "J.D. Salinger")
            .add_text("category", "coming-of-age")
            .add_integer("year", 1951)
            .add_text("tags", "american teenage rebellion")
            .add_text("id", "book006")
            .build(),
    ];

    println!("Adding documents to the index...");

    // Add documents with per-field analyzer configuration
    // Note: We need to manually configure the writer since SearchEngine doesn't persist
    // analyzer configuration across writer() calls yet
    {
        use sage::analysis::analyzer::per_field::PerFieldAnalyzer;
        use sage::full_text::index::advanced_writer::{AdvancedIndexWriter, AdvancedWriterConfig};

        let storage = engine.storage().clone();

        // Configure field-specific analyzers using PerFieldAnalyzer (Lucene-style)
        // - category and id use KeywordAnalyzer (entire field as one token)
        // - other fields use StandardAnalyzer (default) for tokenized search
        // Note: Reuse analyzer instances with Arc::clone to save memory
        let standard_analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);
        let keyword_analyzer: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
        let mut per_field_analyzer = PerFieldAnalyzer::new(Arc::clone(&standard_analyzer));
        per_field_analyzer.add_analyzer("category", Arc::clone(&keyword_analyzer));
        per_field_analyzer.add_analyzer("id", Arc::clone(&keyword_analyzer));

        let config = AdvancedWriterConfig {
            analyzer: Arc::new(per_field_analyzer),
            ..Default::default()
        };

        let mut writer = AdvancedIndexWriter::new(storage, config)?;

        for doc in documents {
            writer.add_document(doc)?;
        }

        writer.commit()?;
    }

    // Commit changes to engine
    engine.commit()?;

    println!("\n=== Field-Specific Search Examples ===\n");

    // Example 1: Search by author
    println!("1. Search by author (author:Orwell):");
    let parser = sage::query::parser::QueryParser::new();
    let query = parser.parse_field("author", "Orwell")?;
    let results = engine.search(SearchRequest::new(query))?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
    }

    // Example 2: Search by author with document loading
    println!("\n2. Search by author with document details (author:Orwell):");
    let request = SearchRequest::new(Box::new(sage::query::term::TermQuery::new(
        "author", "orwell",
    )))
    .load_documents(true);
    let results = engine.search(request)?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title")
                && let Some(title) = field_value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field_value) = doc.get_field("category")
                && let Some(category) = field_value.as_text()
            {
                println!("      Category: {category}");
            }
            if let Some(sage::document::field_value::FieldValue::Integer(year)) =
                doc.get_field("year")
            {
                println!("      Year: {year}");
            }
        }
    }

    // Example 3: Search by category
    println!("\n3. Search by category (category:classic):");
    let parser = sage::query::parser::QueryParser::new();
    let query = parser.parse_field("category", "classic")?;
    let results = engine.search(SearchRequest::new(query))?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
    }

    // Example 4: Search in tags field
    println!("\n4. Search in tags field (tags:british):");
    let request = SearchRequest::new(Box::new(sage::query::term::TermQuery::new(
        "tags", "british",
    )))
    .load_documents(true);
    let results = engine.search(request)?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title")
                && let Some(title) = field_value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field_value) = doc.get_field("author")
                && let Some(author) = field_value.as_text()
            {
                println!("      Author: {author}");
            }
            if let Some(field_value) = doc.get_field("tags")
                && let Some(tags) = field_value.as_text()
            {
                println!("      Tags: {tags}");
            }
        }
    }

    // Example 5: Search in title field
    println!("\n5. Search in title field (title:farm):");
    let parser = sage::query::parser::QueryParser::new();
    let query = parser.parse_field("title", "farm")?;
    let results = engine.search(SearchRequest::new(query))?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
    }

    // Example 6: Search in body field
    println!("\n6. Search in body field (body:father):");
    let parser = sage::query::parser::QueryParser::new();
    let query = parser.parse_field("body", "father")?;
    let results = engine.search(SearchRequest::new(query))?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
    }

    // Example 7: Multi-field search comparison
    println!("\n7. Multi-field search comparison:");
    println!("   Searching for 'american' in different fields:");

    // Search in tags
    let parser = sage::query::parser::QueryParser::new();
    let query = parser.parse_field("tags", "american")?;
    let tags_results = engine.search(SearchRequest::new(query))?;
    println!("   - In tags field: {} results", tags_results.total_hits);

    // Search in body
    let query = parser.parse_field("body", "american")?;
    let body_results = engine.search(SearchRequest::new(query))?;
    println!("   - In body field: {} results", body_results.total_hits);

    // Search in category
    let query = parser.parse_field("category", "american")?;
    let category_results = engine.search(SearchRequest::new(query))?;
    println!(
        "   - In category field: {} results",
        category_results.total_hits
    );

    // Example 8: Using query parser with field specification
    println!("\n8. Using query parser with field specification:");
    let parser = sage::query::parser::QueryParser::new();

    // Parse field:value syntax
    let query = parser.parse("author:austen OR category:dystopian")?;
    println!("   Query: author:austen OR category:dystopian");

    let request = SearchRequest::new(query).load_documents(true);
    let results = engine.search(request)?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title")
                && let Some(title) = field_value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field_value) = doc.get_field("author")
                && let Some(author) = field_value.as_text()
            {
                println!("      Author: {author}");
            }
            if let Some(field_value) = doc.get_field("category")
                && let Some(category) = field_value.as_text()
            {
                println!("      Category: {category}");
            }
        }
    }

    engine.close()?;
    println!("\nField-specific search example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_specific_search_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
