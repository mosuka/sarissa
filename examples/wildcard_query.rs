//! WildcardQuery example - demonstrates pattern matching with * and ? wildcards.

use tempfile::TempDir;

use sage::document::document::Document;
use sage::error::Result;
use sage::lexical::engine::LexicalEngine;
use sage::lexical::index::{LexicalIndexConfig, LexicalIndexFactory};
use sage::lexical::types::LexicalSearchRequest;
use sage::query::wildcard::WildcardQuery;
use sage::storage::file::FileStorage;
use sage::storage::file::FileStorageConfig;
use std::sync::Arc;

fn main() -> Result<()> {
    println!("=== WildcardQuery Example - Pattern Matching with Wildcards ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a search engine
    let config = LexicalIndexConfig::default();
    let storage = Arc::new(FileStorage::new(
        temp_dir.path(),
        FileStorageConfig::new(temp_dir.path()),
    )?);
    let index = LexicalIndexFactory::create(storage, config)?;
    let mut engine = LexicalEngine::new(index)?;

    // Add documents with various patterns for wildcard matching
    let documents = vec![
        Document::builder()
            .add_text("title", "JavaScript Tutorial for Beginners")
            .add_text("filename", "javascript_tutorial.pdf")
            .add_text("description", "Complete JavaScript programming guide")
            .add_text("category", "programming")
            .add_text("extension", "pdf")
            .add_text("id", "file001")
            .build(),
        Document::builder()
            .add_text("title", "Python Programming Reference")
            .add_text("filename", "python_reference.html")
            .add_text("description", "Comprehensive Python programming reference")
            .add_text("category", "programming")
            .add_text("extension", "html")
            .add_text("id", "file002")
            .build(),
        Document::builder()
            .add_text("title", "Machine Learning Algorithms")
            .add_text("filename", "ml_algorithms.docx")
            .add_text("description", "Understanding machine learning techniques")
            .add_text("category", "data-science")
            .add_text("extension", "docx")
            .add_text("id", "file003")
            .build(),
        Document::builder()
            .add_text("title", "Database Design Principles")
            .add_text("filename", "database_design.pptx")
            .add_text("description", "Principles of good database design")
            .add_text("category", "database")
            .add_text("extension", "pptx")
            .add_text("id", "file004")
            .build(),
        Document::builder()
            .add_text("title", "Web Development Best Practices")
            .add_text("filename", "web_dev_practices.txt")
            .add_text("description", "Best practices for web development")
            .add_text("category", "web-development")
            .add_text("extension", "txt")
            .add_text("id", "file005")
            .build(),
        Document::builder()
            .add_text("title", "React Component Patterns")
            .add_text("filename", "react_patterns.jsx")
            .add_text(
                "description",
                "Common patterns in React component development",
            )
            .add_text("category", "frontend")
            .add_text("extension", "jsx")
            .add_text("id", "file006")
            .build(),
        Document::builder()
            .add_text("title", "API Documentation Template")
            .add_text("filename", "api_docs_template.md")
            .add_text("description", "Template for creating API documentation")
            .add_text("category", "documentation")
            .add_text("extension", "md")
            .add_text("id", "file007")
            .build(),
        Document::builder()
            .add_text("title", "Configuration Settings")
            .add_text("filename", "app_config.json")
            .add_text("description", "Application configuration file")
            .add_text("category", "configuration")
            .add_text("extension", "json")
            .add_text("id", "file008")
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    engine.add_documents(documents)?;

    println!("\n=== WildcardQuery Examples ===\n");

    // Example 1: Wildcard at the end (prefix matching)
    println!("1. Files starting with 'java' using 'java*' pattern:");
    let query = WildcardQuery::new("filename", "java*")?;
    let request = LexicalSearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("filename")
            && let Some(filename) = field_value.as_text()
        {
            println!("      Filename: {filename}");
        }
    }

    // Example 2: Wildcard at the beginning (suffix matching)
    println!("\n2. Files ending with '.pdf' using '*.pdf' pattern:");
    let query = WildcardQuery::new("filename", "*.pdf")?;
    let request = LexicalSearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("filename")
            && let Some(filename) = field_value.as_text()
        {
            println!("      Filename: {filename}");
        }
    }

    // Example 3: Wildcard in the middle
    println!("\n3. Files with 'web' followed by anything ending in '.txt' using 'web*.txt':");
    let query = WildcardQuery::new("filename", "web*.txt")?;
    let request = LexicalSearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("filename")
            && let Some(filename) = field_value.as_text()
        {
            println!("      Filename: {filename}");
        }
    }

    // Example 4: Single character wildcard (?)
    println!("\n4. Extensions with pattern '?sx' (jsx, tsx, etc.):");
    let query = WildcardQuery::new("extension", "?sx")?;
    let request = LexicalSearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("extension")
            && let Some(ext) = field_value.as_text()
        {
            println!("      Extension: {ext}");
        }
    }

    // Example 5: Multiple wildcards
    println!("\n5. Categories starting with 'prog' and ending with 'ing' using 'prog*ing':");
    let query = WildcardQuery::new("category", "prog*ing")?;
    let request = LexicalSearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("category")
            && let Some(category) = field_value.as_text()
        {
            println!("      Category: {category}");
        }
    }

    // Example 6: Complex pattern with both wildcards
    println!("\n6. Filenames with pattern '*_*.????' (underscore and 4-char extension):");
    let query = WildcardQuery::new("filename", "*_*.????")?;
    let request = LexicalSearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("filename")
            && let Some(filename) = field_value.as_text()
        {
            println!("      Filename: {filename}");
        }
    }

    // Example 7: Title pattern matching
    println!("\n7. Titles containing 'Development' using '*Development*':");
    let query = WildcardQuery::new("title", "*Development*")?;
    let request = LexicalSearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

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
            && let Some(title) = field_value.as_text()
        {
            println!("      Title: {title}");
        }
    }

    // Example 8: Single character matching
    println!("\n8. Extensions with exactly 3 characters using '???':");
    let query = WildcardQuery::new("extension", "???")?;
    let request = LexicalSearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("extension")
            && let Some(ext) = field_value.as_text()
        {
            println!("      Extension: {ext}");
        }
    }

    // Example 9: Match all files with any extension
    println!("\n9. All files with any extension using '*.*':");
    let query = WildcardQuery::new("filename", "*.*")?;
    let request = LexicalSearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("filename")
            && let Some(filename) = field_value.as_text()
        {
            println!("      Filename: {filename}");
        }
    }

    // Example 10: No matches
    println!("\n10. Pattern with no matches using 'xyz*abc':");
    let query = WildcardQuery::new("filename", "xyz*abc")?;
    let request = LexicalSearchRequest::new(Box::new(query));
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);

    // Example 11: Count matching documents
    println!("\n11. Counting files with 'data' in filename using '*data*':");
    let query = WildcardQuery::new("filename", "*data*")?;
    let count = engine.count(Box::new(query))?;
    println!("    Count: {count} files");

    engine.close()?;
    println!("\nWildcardQuery example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wildcard_query_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
