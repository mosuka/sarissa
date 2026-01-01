//! Lexical Search Example - Basic usage guide
//!
//! This example demonstrates the fundamental steps to use Sarissa for lexical search:
//! 1. Setup storage and analyzer (using PerFieldAnalyzer)
//! 2. Configure the index
//! 3. Add documents
//! 4. Perform a search

use std::sync::Arc;

use sarissa::analysis::analyzer::analyzer::Analyzer;
use sarissa::analysis::analyzer::keyword::KeywordAnalyzer;
use sarissa::analysis::analyzer::per_field::PerFieldAnalyzer;
use sarissa::analysis::analyzer::standard::StandardAnalyzer;
use sarissa::error::Result;
use sarissa::lexical::core::document::Document;
use sarissa::lexical::core::field::TextOption;
use sarissa::lexical::engine::LexicalEngine;
use sarissa::lexical::engine::config::LexicalIndexConfig;
use sarissa::lexical::index::config::InvertedIndexConfig;
use sarissa::lexical::index::inverted::query::Query;
use sarissa::lexical::index::inverted::query::term::TermQuery;
use sarissa::lexical::search::searcher::LexicalSearchRequest;
use sarissa::storage::file::FileStorageConfig;
use sarissa::storage::{StorageConfig, StorageFactory};
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Lexical Search Basic Example ===\n");

    // 1. Setup Storage
    // We use a temporary directory for this example, but in a real app you'd use a persistent path.
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Setup Analyzer (PerFieldAnalyzer)
    // We use PerFieldAnalyzer to apply different analysis strategies to different fields.
    // - Default: StandardAnalyzer (tokenizes, lowercases, removes stop words)
    // - "category": KeywordAnalyzer (treats the entire input as a single token, case-sensitive)
    let default_analyzer = Arc::new(StandardAnalyzer::new()?);
    let mut analyzer = PerFieldAnalyzer::new(default_analyzer);

    // Add specific analyzer for "category" field
    analyzer.add_analyzer("category", Arc::new(KeywordAnalyzer::new()));

    let analyzer_arc: Arc<dyn Analyzer> = Arc::new(analyzer);

    // 3. Configure Index
    // We use the Inverted index type for lexical search.
    let index_config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer: analyzer_arc,
        ..InvertedIndexConfig::default()
    });

    // 4. Create Engine
    let engine = LexicalEngine::new(storage, index_config)?;

    // 5. Add Documents
    // Let's index a few simple documents with a category.
    let documents = vec![
        Document::builder()
            .add_text(
                "title",
                "The Rust Programming Language",
                TextOption::default(),
            )
            .add_text(
                "body",
                "Rust is fast and memory efficient.",
                TextOption::default(),
            )
            .add_text("category", "TECHNOLOGY", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Learning Search Engines", TextOption::default())
            .add_text(
                "body",
                "Search engines are complex but fascinating.",
                TextOption::default(),
            )
            .add_text("category", "EDUCATION", TextOption::default())
            .build(),
        Document::builder()
            .add_text(
                "title",
                "Cooking with Rust (Iron Skillets)",
                TextOption::default(),
            )
            .add_text(
                "body",
                "How to season your cast iron skillet.",
                TextOption::default(),
            )
            .add_text("category", "LIFESTYLE", TextOption::default())
            .build(),
    ];

    println!("Indexing {} documents...", documents.len());
    for doc in documents {
        engine.add_document(doc)?;
    }
    // Don't forget to commit! Changes are not visible until committed.
    engine.commit()?;

    // 6. Search

    // Demo 1: Search in 'title' (StandardAnalyzer)
    println!("\n--- Search 1: 'Rust' in 'title' (Standard Analysis) ---");
    let query = TermQuery::new("title", "rust"); // Lowercase 'rust' matches because StandardAnalyzer lowercases input
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = engine.search(request)?;

    println!("Found {} hits:", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!("{}. Doc ID: {}, Score: {:.4}", i + 1, hit.doc_id, hit.score);
        if let Some(doc) = &hit.document {
            print_doc_summary(doc);
        }
    }

    // Demo 2: Search in 'category' (KeywordAnalyzer)
    println!("\n--- Search 2: 'TECHNOLOGY' in 'category' (Keyword Analysis) ---");
    // KeywordAnalyzer is case-sensitive and requires exact match.
    // "TECHNOLOGY" matches "TECHNOLOGY". "technology" would NOT match.
    let query = TermQuery::new("category", "TECHNOLOGY");
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = engine.search(request)?;

    println!("Found {} hits:", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!("{}. Doc ID: {}, Score: {:.4}", i + 1, hit.doc_id, hit.score);
        if let Some(doc) = &hit.document {
            print_doc_summary(doc);
        }
    }

    Ok(())
}

fn print_doc_summary(doc: &Document) {
    if let Some(title) = doc.get_field("title").and_then(|f| f.value.as_text()) {
        println!("   Title: {}", title);
    }
    if let Some(category) = doc.get_field("category").and_then(|f| f.value.as_text()) {
        println!("   Category: {}", category);
    }
}
