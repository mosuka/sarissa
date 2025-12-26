//! TermQuery example - demonstrates single term exact matching search.

use std::sync::Arc;

use tempfile::TempDir;

use sarissa::analysis::analyzer::analyzer::Analyzer;
use sarissa::analysis::analyzer::keyword::KeywordAnalyzer;
use sarissa::analysis::analyzer::per_field::PerFieldAnalyzer;
use sarissa::analysis::analyzer::standard::StandardAnalyzer;
use sarissa::document::document::Document;
use sarissa::document::field::TextOption;
use sarissa::error::Result;
use sarissa::lexical::engine::LexicalEngine;
use sarissa::lexical::index::config::InvertedIndexConfig;
use sarissa::lexical::index::config::LexicalIndexConfig;
use sarissa::lexical::index::inverted::query::Query;
use sarissa::lexical::index::inverted::query::term::TermQuery;
use sarissa::lexical::search::searcher::LexicalSearchRequest;
use sarissa::storage::StorageConfig;
use sarissa::storage::StorageFactory;
use sarissa::storage::file::FileStorageConfig;

fn main() -> Result<()> {
    println!("=== TermQuery Example - Single Term Exact Matching ===\n");

    // Create a storage backend
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    // Create an analyzer
    let standard_analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);
    let keyword_analyzer: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
    let mut per_field_analyzer = PerFieldAnalyzer::new(Arc::clone(&standard_analyzer));
    per_field_analyzer.add_analyzer("id", Arc::clone(&keyword_analyzer));

    // Create a lexical engine
    let lexical_index_config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer: Arc::new(per_field_analyzer.clone()),
        ..InvertedIndexConfig::default()
    });
    let lexical_engine = LexicalEngine::new(storage, lexical_index_config)?;

    // Add documents with various terms
    let documents = vec![
        Document::builder()
            .add_text("title", "Rust Programming Language", TextOption::default())
            .add_text(
                "body",
                "Rust is a systems programming language focused on safety, speed, and concurrency",
                TextOption::default(),
            )
            .add_text("author", "Steve Klabnik", TextOption::default())
            .add_text("category", "programming", TextOption::default())
            .add_text("id", "doc1", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Python for Beginners", TextOption::default())
            .add_text(
                "body",
                "Python is a versatile and easy-to-learn programming language",
                TextOption::default(),
            )
            .add_text("author", "John Smith", TextOption::default())
            .add_text("category", "programming", TextOption::default())
            .add_text("id", "doc2", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "JavaScript Essentials", TextOption::default())
            .add_text(
                "body",
                "JavaScript is the language of the web, used for frontend and backend development",
                TextOption::default(),
            )
            .add_text("author", "Jane Doe", TextOption::default())
            .add_text("category", "web-development", TextOption::default())
            .add_text("id", "doc3", TextOption::default())
            .build(),
        Document::builder()
            .add_text(
                "title",
                "Machine Learning Fundamentals",
                TextOption::default(),
            )
            .add_text(
                "body",
                "Machine learning is a subset of artificial intelligence focused on algorithms",
                TextOption::default(),
            )
            .add_text("author", "Alice Johnson", TextOption::default())
            .add_text("category", "data-science", TextOption::default())
            .add_text("id", "doc4", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Data Structures in C++", TextOption::default())
            .add_text(
                "body",
                "Understanding data structures is crucial for efficient programming",
                TextOption::default(),
            )
            .add_text("author", "Bob Wilson", TextOption::default())
            .add_text("category", "programming", TextOption::default())
            .add_text("id", "doc5", TextOption::default())
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    for doc in documents {
        lexical_engine.add_document(doc)?;
    }

    lexical_engine.commit()?;

    println!("\n=== TermQuery Examples ===\n");

    // Example 1: Search for exact term in title field
    println!("1. Searching for 'Rust' in title field:");
    let query = TermQuery::new("title", "Rust");
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("title")
            && let Some(title) = field_value.value.as_text()
        {
            println!("      Title: {title}");
        }
    }

    // Example 2: Search for exact term in body field
    println!("\n2. Searching for 'language' in body field:");
    let query = TermQuery::new("body", "language");
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("title")
            && let Some(title) = field_value.value.as_text()
        {
            println!("      Title: {title}");
        }
    }

    // Example 3: Search for exact term in category field
    println!("\n3. Searching for 'programming' in category field:");
    let query = TermQuery::new("category", "programming");
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("title")
            && let Some(title) = field_value.value.as_text()
        {
            println!("      Title: {title}");
        }
    }

    // Example 4: Search for non-existent term
    println!("\n4. Searching for non-existent term 'golang':");
    let query = TermQuery::new("title", "golang");
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>);
    let results = lexical_engine.search(request)?;

    println!("   Found {} results", results.total_hits);

    // Example 5: Case sensitivity demonstration
    println!("\n5. Case sensitivity - searching for 'rust' (lowercase):");
    let query = TermQuery::new("title", "rust");
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>);
    let results = lexical_engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    println!("   Note: TermQuery is case-sensitive by default");

    // Example 6: Author exact match
    println!("\n6. Searching for exact author 'John Smith':");
    let query = TermQuery::new("author", "John Smith");
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
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
        }
    }

    // Example 7: Count matching documents
    println!("\n7. Counting documents containing 'programming':");
    let query = TermQuery::new("body", "programming");
    let count =
        lexical_engine.count(LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>))?;
    println!("   Count: {count} documents");

    lexical_engine.close()?;
    println!("\nTermQuery example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_query_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
