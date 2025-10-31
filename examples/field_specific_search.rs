//! Field-specific search example - demonstrates searching within specific fields.

use std::sync::Arc;

use tempfile::TempDir;

use yatagarasu::analysis::analyzer::analyzer::Analyzer;
use yatagarasu::analysis::analyzer::keyword::KeywordAnalyzer;
use yatagarasu::analysis::analyzer::per_field::PerFieldAnalyzer;
use yatagarasu::analysis::analyzer::standard::StandardAnalyzer;
use yatagarasu::document::document::Document;
use yatagarasu::document::field_value::FieldValue;
use yatagarasu::error::Result;
use yatagarasu::lexical::engine::LexicalEngine;
use yatagarasu::lexical::index::config::{InvertedIndexConfig, LexicalIndexConfig};
use yatagarasu::lexical::index::factory::LexicalIndexFactory;
use yatagarasu::lexical::types::LexicalSearchRequest;
use yatagarasu::lexical::index::inverted::query::term::TermQuery;
use yatagarasu::storage::file::FileStorageConfig;
use yatagarasu::storage::{StorageConfig, StorageFactory};

fn main() -> Result<()> {
    println!("=== Field-Specific Search Example ===\n");

    // Create a storage backend
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    // Create an analyzer
    let standard_analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);
    let keyword_analyzer: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
    let mut per_field_analyzer = PerFieldAnalyzer::new(Arc::clone(&standard_analyzer));
    per_field_analyzer.add_analyzer("category", Arc::clone(&keyword_analyzer));
    per_field_analyzer.add_analyzer("id", Arc::clone(&keyword_analyzer));

    // Create a lexical index
    let lexical_index_config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer: Arc::new(per_field_analyzer.clone()),
        ..InvertedIndexConfig::default()
    });
    let lexical_index = LexicalIndexFactory::create(storage, lexical_index_config)?;

    // Create a lexical engine
    let mut lexical_engine = LexicalEngine::new(lexical_index)?;

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

    println!("Adding {} documents to the index...", documents.len());

    // Add documents to the lexical engine
    lexical_engine.add_documents(documents)?;

    // Commit changes to engine
    lexical_engine.commit()?;

    println!("\n=== Field-Specific Search Examples ===\n");

    // Example 1: Search by author using DSL string
    println!("1. Search by author using DSL string (author:Orwell):");
    let request = LexicalSearchRequest::new("author:Orwell");
    let results = lexical_engine.search(request)?;
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
    let request =
        LexicalSearchRequest::new(Box::new(TermQuery::new("author", "orwell"))
            as Box<dyn yatagarasu::lexical::index::inverted::query::Query>)
        .load_documents(true);
    let results = lexical_engine.search(request)?;
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
            if let Some(FieldValue::Integer(year)) = doc.get_field("year") {
                println!("      Year: {year}");
            }
        }
    }

    // Example 3: Search by category using DSL string
    println!("\n3. Search by category using DSL string (category:classic):");
    let request = LexicalSearchRequest::new("category:classic");
    let results = lexical_engine.search(request)?;
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
    let request = LexicalSearchRequest::new(
        Box::new(TermQuery::new("tags", "british")) as Box<dyn yatagarasu::lexical::index::inverted::query::Query>
    )
    .load_documents(true);
    let results = lexical_engine.search(request)?;
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

    // Example 5: Search in title field using DSL string
    println!("\n5. Search in title field using DSL string (title:farm):");
    let request = LexicalSearchRequest::new("title:farm");
    let results = lexical_engine.search(request)?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.2}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
    }

    // Example 6: Search in body field using DSL string
    println!("\n6. Search in body field using DSL string (body:father):");
    let request = LexicalSearchRequest::new("body:father");
    let results = lexical_engine.search(request)?;
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
    let tags_results = lexical_engine.search(LexicalSearchRequest::new("tags:american"))?;
    println!("   - In tags field: {} results", tags_results.total_hits);

    // Search in body
    let body_results = lexical_engine.search(LexicalSearchRequest::new("body:american"))?;
    println!("   - In body field: {} results", body_results.total_hits);

    // Search in category
    let category_results = lexical_engine.search(LexicalSearchRequest::new("category:american"))?;
    println!(
        "   - In category field: {} results",
        category_results.total_hits
    );

    // Example 8: Using query parser with field specification (DSL string)
    println!("\n8. Using DSL string with boolean operators:");
    println!("   Query: author:austen OR category:dystopian");

    let request =
        LexicalSearchRequest::new("author:austen OR category:dystopian").load_documents(true);
    let results = lexical_engine.search(request)?;
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

    lexical_engine.close()?;
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
