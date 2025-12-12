//! Field-specific search example - demonstrates searching within specific fields.

use std::sync::Arc;

use tempfile::TempDir;

use platypus::analysis::analyzer::analyzer::Analyzer;
use platypus::analysis::analyzer::keyword::KeywordAnalyzer;
use platypus::analysis::analyzer::per_field::PerFieldAnalyzer;
use platypus::analysis::analyzer::standard::StandardAnalyzer;
use platypus::document::document::Document;
use platypus::document::field::{IntegerOption, TextOption};
use platypus::error::Result;
use platypus::lexical::engine::LexicalEngine;
use platypus::lexical::index::config::{InvertedIndexConfig, LexicalIndexConfig};
use platypus::lexical::index::inverted::query::term::TermQuery;
use platypus::lexical::search::searcher::LexicalSearchRequest;
use platypus::storage::file::FileStorageConfig;
use platypus::storage::{StorageConfig, StorageFactory};

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

    // Create a lexical engine
    let lexical_index_config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer: Arc::new(per_field_analyzer.clone()),
        ..InvertedIndexConfig::default()
    });
    let mut lexical_engine = LexicalEngine::new(storage, lexical_index_config)?;

    // Prepare documents
    let documents = vec![
        Document::builder()
            .add_text("title", "The Great Gatsby", TextOption::default())
            .add_text("body", "In my younger and more vulnerable years my father gave me some advice", TextOption::default())
            .add_text("author", "F. Scott Fitzgerald", TextOption::default())
            .add_text("category", "classic", TextOption::default())
            .add_integer("year", 1925, IntegerOption::default())
            .add_text("tags", "american jazz-age tragedy", TextOption::default())
            .add_text("id", "book001", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "To Kill a Mockingbird", TextOption::default())
            .add_text("body", "When I was almost six years old, I heard my brother arguing with my father", TextOption::default())
            .add_text("author", "Harper Lee", TextOption::default())
            .add_text("category", "classic", TextOption::default())
            .add_integer("year", 1960, IntegerOption::default())
            .add_text("tags", "american southern racism", TextOption::default())
            .add_text("id", "book002", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "1984", TextOption::default())
            .add_text("body", "It was a bright cold day in April, and the clocks were striking thirteen", TextOption::default())
            .add_text("author", "George Orwell", TextOption::default())
            .add_text("category", "dystopian", TextOption::default())
            .add_integer("year", 1949, IntegerOption::default())
            .add_text("tags", "british totalitarian surveillance", TextOption::default())
            .add_text("id", "book003", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Animal Farm", TextOption::default())
            .add_text("body", "Mr Jones of Manor Farm, had locked the hen houses for the night", TextOption::default())
            .add_text("author", "George Orwell", TextOption::default())
            .add_text("category", "satire", TextOption::default())
            .add_integer("year", 1945, IntegerOption::default())
            .add_text("tags", "british allegory political", TextOption::default())
            .add_text("id", "book004", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Pride and Prejudice", TextOption::default())
            .add_text("body", "It is a truth universally acknowledged, that a single man in possession of a good fortune", TextOption::default())
            .add_text("author", "Jane Austen", TextOption::default())
            .add_text("category", "romance", TextOption::default())
            .add_integer("year", 1813, IntegerOption::default())
            .add_text("tags", "british regency society", TextOption::default())
            .add_text("id", "book005", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "The Catcher in the Rye", TextOption::default())
            .add_text("body", "If you really want to hear about it, the first thing you'll probably want to know", TextOption::default())
            .add_text("author", "J.D. Salinger", TextOption::default())
            .add_text("category", "coming-of-age", TextOption::default())
            .add_integer("year", 1951, IntegerOption::default())
            .add_text("tags", "american teenage rebellion", TextOption::default())
            .add_text("id", "book006", TextOption::default())
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());

    // Add documents to the lexical engine
    for doc in documents {
        lexical_engine.add_document(doc)?;
    }

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
    let request = LexicalSearchRequest::new(Box::new(TermQuery::new("author", "orwell"))
        as Box<dyn platypus::lexical::index::inverted::query::Query>)
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
                && let Some(title) = field_value.value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field_value) = doc.get_field("category")
                && let Some(category) = field_value.value.as_text()
            {
                println!("      Category: {category}");
            }
            if let Some(field) = doc.get_field("year")
                && let platypus::document::field::FieldValue::Integer(year) = &field.value
            {
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
    let request = LexicalSearchRequest::new(Box::new(TermQuery::new("tags", "british"))
        as Box<dyn platypus::lexical::index::inverted::query::Query>)
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
                && let Some(title) = field_value.value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field_value) = doc.get_field("author")
                && let Some(author) = field_value.value.as_text()
            {
                println!("      Author: {author}");
            }
            if let Some(field_value) = doc.get_field("tags")
                && let Some(tags) = field_value.value.as_text()
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
                && let Some(title) = field_value.value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field_value) = doc.get_field("author")
                && let Some(author) = field_value.value.as_text()
            {
                println!("      Author: {author}");
            }
            if let Some(field_value) = doc.get_field("category")
                && let Some(category) = field_value.value.as_text()
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
