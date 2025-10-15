//! TermQuery example - demonstrates single term exact matching search.

use sage::full_text::index::IndexConfig;
use sage::full_text_search::SearchEngine;
use sage::full_text_search::SearchRequest;
use sage::prelude::*;
use sage::query::TermQuery;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== TermQuery Example - Single Term Exact Matching ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), IndexConfig::default())?;

    // Add documents with various terms
    let documents = vec![
        Document::builder()
            .add_text("title", "Rust Programming Language")
            .add_text(
                "body",
                "Rust is a systems programming language focused on safety, speed, and concurrency",
            )
            .add_text("author", "Steve Klabnik")
            .add_text("category", "programming")
            .build(),
        Document::builder()
            .add_text("title", "Python for Beginners")
            .add_text(
                "body",
                "Python is a versatile and easy-to-learn programming language",
            )
            .add_text("author", "John Smith")
            .add_text("category", "programming")
            .build(),
        Document::builder()
            .add_text("title", "JavaScript Essentials")
            .add_text(
                "body",
                "JavaScript is the language of the web, used for frontend and backend development",
            )
            .add_text("author", "Jane Doe")
            .add_text("category", "web-development")
            .build(),
        Document::builder()
            .add_text("title", "Machine Learning Fundamentals")
            .add_text(
                "body",
                "Machine learning is a subset of artificial intelligence focused on algorithms",
            )
            .add_text("author", "Alice Johnson")
            .add_text("category", "data-science")
            .build(),
        Document::builder()
            .add_text("title", "Data Structures in C++")
            .add_text(
                "body",
                "Understanding data structures is crucial for efficient programming",
            )
            .add_text("author", "Bob Wilson")
            .add_text("category", "programming")
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    engine.add_documents(documents)?;

    println!("\n=== TermQuery Examples ===\n");

    // Example 1: Search for exact term in title field
    println!("1. Searching for 'Rust' in title field:");
    let query = TermQuery::new("title", "Rust");
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
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

    // Example 2: Search for exact term in body field
    println!("\n2. Searching for 'language' in body field:");
    let query = TermQuery::new("body", "language");
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
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

    // Example 3: Search for exact term in category field
    println!("\n3. Searching for 'programming' in category field:");
    let query = TermQuery::new("category", "programming");
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
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

    // Example 4: Search for non-existent term
    println!("\n4. Searching for non-existent term 'golang':");
    let query = TermQuery::new("title", "golang");
    let request = SearchRequest::new(Box::new(query));
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);

    // Example 5: Case sensitivity demonstration
    println!("\n5. Case sensitivity - searching for 'rust' (lowercase):");
    let query = TermQuery::new("title", "rust");
    let request = SearchRequest::new(Box::new(query));
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    println!("   Note: TermQuery is case-sensitive by default");

    // Example 6: Author exact match
    println!("\n6. Searching for exact author 'John Smith':");
    let query = TermQuery::new("author", "John Smith");
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

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
                && let Some(title) = field_value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field_value) = doc.get_field("author")
                && let Some(author) = field_value.as_text()
            {
                println!("      Author: {author}");
            }
        }
    }

    // Example 7: Count matching documents
    println!("\n7. Counting documents containing 'programming':");
    let query = TermQuery::new("body", "programming");
    let count = engine.count(Box::new(query))?;
    println!("   Count: {count} documents");

    engine.close()?;
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
