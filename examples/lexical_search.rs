//! Lexical Search Example - Basic usage guide
//!
//! This example demonstrates the fundamental steps to use Sarissa for lexical search:
//! 1. Setup storage and analyzer
//! 2. Configure the index
//! 3. Add documents
//! 4. Perform a search

use std::sync::Arc;

use sarissa::analysis::analyzer::analyzer::Analyzer;
use sarissa::analysis::analyzer::standard::StandardAnalyzer;
use sarissa::document::document::Document;
use sarissa::document::field::TextOption;
use sarissa::error::Result;
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

    // 2. Setup Analyzer
    // The StandardAnalyzer is good for general text (tokenizes, lowercases, removes stop words).
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);

    // 3. Configure Index
    // We use the Inverted index type for lexical search.
    let index_config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer,
        ..InvertedIndexConfig::default()
    });

    // 4. Create Engine
    let engine = LexicalEngine::new(storage, index_config)?;

    // 5. Add Documents
    // Let's index a few simple documents.
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
            .build(),
        Document::builder()
            .add_text("title", "Learning Search Engines", TextOption::default())
            .add_text(
                "body",
                "Search engines are complex but fascinating.",
                TextOption::default(),
            )
            .build(),
    ];

    println!("Indexing {} documents...", documents.len());
    for doc in documents {
        engine.add_document(doc)?;
    }
    // Don't forget to commit! Changes are not visible until committed.
    engine.commit()?;

    // 6. Search
    println!("\nSearching for 'Rust' in 'title' field:");

    // Create a TermQuery: looks for exact term match (after analysis).
    let query = TermQuery::new("title", "rust");

    // Create a search request and execute it.
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true); // We want to retrieve the actual document content.

    let results = engine.search(request)?;

    println!("Found {} hits:", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!("{}. Doc ID: {}, Score: {:.4}", i + 1, hit.doc_id, hit.score);
        if let Some(doc) = &hit.document {
            if let Some(title) = doc.get_field("title").and_then(|f| f.value.as_text()) {
                println!("   Title: {}", title);
            }
        }
    }

    Ok(())
}
