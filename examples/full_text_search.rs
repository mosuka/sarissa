//! Full-text search example - demonstrates comprehensive full-text search capabilities.
//!
//! This example showcases the core full-text search features including:
//! - Creating and configuring a search index
//! - Adding documents with multiple fields
//! - Performing various types of text searches
//! - Retrieving and displaying search results
//! - Counting matching documents

use sage::document::document::Document;
use sage::error::Result;
use sage::full_text::index::IndexConfig;
use sage::full_text_search::engine::SearchEngine;
use sage::full_text_search::SearchRequest;
use sage::query::boolean::BooleanQuery;
use sage::query::phrase::PhraseQuery;
use sage::query::range::NumericRangeQuery;
use sage::query::term::TermQuery;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Full-Text Search Example ===\n");

    // Step 1: Create a search index
    println!("Step 1: Creating search index...");
    let temp_dir = TempDir::new().unwrap();
    println!("  Index location: {:?}\n", temp_dir.path());

    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), IndexConfig::default())?;

    // Step 2: Prepare and index documents
    println!("Step 2: Indexing documents...");
    let documents = vec![
        Document::builder()
            .add_text("id", "doc001")
            .add_text("title", "Introduction to Rust Programming")
            .add_text(
                "body",
                "Rust is a modern systems programming language that focuses on safety, \
                 speed, and concurrency. It provides memory safety without garbage collection \
                 and enables developers to write efficient and reliable software.",
            )
            .add_text("author", "Alice Johnson")
            .add_text("category", "programming")
            .add_text("tags", "rust systems-programming memory-safety")
            .add_integer("year", 2023)
            .build(),
        Document::builder()
            .add_text("id", "doc002")
            .add_text("title", "Web Development with Rust")
            .add_text(
                "body",
                "Building web applications with Rust has become increasingly popular. \
                 Frameworks like Actix and Rocket make it easy to create fast and secure \
                 web services. Rust's performance and safety make it ideal for web development.",
            )
            .add_text("author", "Bob Smith")
            .add_text("category", "web-development")
            .add_text("tags", "rust web actix rocket")
            .add_integer("year", 2023)
            .build(),
        Document::builder()
            .add_text("id", "doc003")
            .add_text("title", "Python for Data Science")
            .add_text(
                "body",
                "Python is the most popular language for data science and machine learning. \
                 Libraries like NumPy, Pandas, and Scikit-learn provide powerful tools for \
                 data analysis and statistical computing.",
            )
            .add_text("author", "Carol Williams")
            .add_text("category", "data-science")
            .add_text("tags", "python data-science machine-learning")
            .add_integer("year", 2022)
            .build(),
        Document::builder()
            .add_text("id", "doc004")
            .add_text("title", "Building Microservices with Rust")
            .add_text(
                "body",
                "Microservices architecture has revolutionized how we build distributed systems. \
                 Rust's lightweight runtime and excellent performance make it a great choice for \
                 building scalable microservices.",
            )
            .add_text("author", "David Brown")
            .add_text("category", "architecture")
            .add_text("tags", "rust microservices distributed-systems")
            .add_integer("year", 2024)
            .build(),
        Document::builder()
            .add_text("id", "doc005")
            .add_text("title", "JavaScript and TypeScript Best Practices")
            .add_text(
                "body",
                "Modern JavaScript development relies heavily on TypeScript for type safety. \
                 This guide covers best practices for writing maintainable JavaScript and \
                 TypeScript code for web applications.",
            )
            .add_text("author", "Eve Davis")
            .add_text("category", "web-development")
            .add_text("tags", "javascript typescript web frontend")
            .add_integer("year", 2023)
            .build(),
        Document::builder()
            .add_text("id", "doc006")
            .add_text("title", "Machine Learning with Python")
            .add_text(
                "body",
                "Deep learning and neural networks are transforming artificial intelligence. \
                 Python frameworks like TensorFlow and PyTorch enable developers to build \
                 sophisticated machine learning models.",
            )
            .add_text("author", "Frank Miller")
            .add_text("category", "data-science")
            .add_text("tags", "python machine-learning deep-learning ai")
            .add_integer("year", 2024)
            .build(),
        Document::builder()
            .add_text("id", "doc007")
            .add_text("title", "Concurrent Programming in Rust")
            .add_text(
                "body",
                "Rust's ownership system makes concurrent programming safe and efficient. \
                 Understanding threads, async/await, and message passing is essential for \
                 building high-performance concurrent applications in Rust.",
            )
            .add_text("author", "Grace Taylor")
            .add_text("category", "programming")
            .add_text("tags", "rust concurrency async parallel")
            .add_integer("year", 2024)
            .build(),
        Document::builder()
            .add_text("id", "doc008")
            .add_text("title", "Database Design Principles")
            .add_text(
                "body",
                "Effective database design is crucial for application performance. \
                 This guide covers normalization, indexing strategies, and query optimization \
                 for both SQL and NoSQL databases.",
            )
            .add_text("author", "Henry Wilson")
            .add_text("category", "database")
            .add_text("tags", "database sql nosql design")
            .add_integer("year", 2023)
            .build(),
    ];

    println!("  Indexed {} documents\n", documents.len());
    engine.add_documents(documents)?;

    // Step 3: Perform various full-text searches
    println!("Step 3: Performing full-text searches...\n");

    // Example 1: Simple term search
    // Note: Using lowercase because StandardAnalyzer normalizes text to lowercase
    println!("Example 1: Search for 'rust' in title field");
    let query = TermQuery::new("title", "rust");
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("  Found {} matching documents:", results.total_hits);
    display_results(&results);

    // Example 2: Full-text search in body field
    println!("\nExample 2: Search for 'programming' in body field");
    let query = TermQuery::new("body", "programming");
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("  Found {} matching documents:", results.total_hits);
    display_results(&results);

    // Example 3: Phrase search
    println!("\nExample 3: Search for phrase 'machine learning'");
    let query = PhraseQuery::new("body", vec!["machine".to_string(), "learning".to_string()]);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("  Found {} matching documents:", results.total_hits);
    display_results(&results);

    // Example 4: Boolean query (AND condition)
    println!("\nExample 4: Boolean query - documents with both 'rust' AND 'programming'");
    let mut query = BooleanQuery::new();
    query.add_must(Box::new(TermQuery::new("body", "rust")));
    query.add_must(Box::new(TermQuery::new("body", "programming")));
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("  Found {} matching documents:", results.total_hits);
    display_results(&results);

    // Example 5: Boolean query (OR condition)
    println!("\nExample 5: Boolean query - documents with 'python' OR 'javascript'");
    let mut query = BooleanQuery::new();
    query.add_should(Box::new(TermQuery::new("body", "python")));
    query.add_should(Box::new(TermQuery::new("body", "javascript")));
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("  Found {} matching documents:", results.total_hits);
    display_results(&results);

    // Example 6: Search by category
    println!("\nExample 6: Filter by category 'web' (from 'web-development')");
    let query = TermQuery::new("category", "web");
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("  Found {} matching documents:", results.total_hits);
    display_results(&results);

    // Example 7: Search by author
    println!("\nExample 7: Search for documents by author 'alice johnson'");
    let query = TermQuery::new("author", "alice");
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("  Found {} matching documents:", results.total_hits);
    display_results(&results);

    // Example 8: Search in tags
    println!("\nExample 8: Search for 'science' tag (from 'data-science')");
    let query = TermQuery::new("tags", "science");
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("  Found {} matching documents:", results.total_hits);
    display_results(&results);

    // Example 9: Range query by year
    println!("\nExample 9: Search for documents from 2023 and later");
    let query = NumericRangeQuery::i64_range("year", Some(2023), None);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("  Found {} matching documents:", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            print!("  {}. ", i + 1);
            if let Some(sage::document::field_value::FieldValue::Text(title)) =
                doc.get_field("title")
            {
                print!("Title: {}", title);
            }
            if let Some(sage::document::field_value::FieldValue::Integer(year)) =
                doc.get_field("year")
            {
                print!(" (Year: {})", year);
            }
            println!();
        }
    }

    // Example 10: Count matching documents
    println!("\nExample 10: Count all documents containing 'rust'");
    let query = TermQuery::new("body", "rust");
    let count = engine.count(Box::new(query))?;
    println!("  Total count: {} documents\n", count);

    // Example 11: Complex boolean query
    println!("\nExample 11: Complex query - 'web' OR 'database', but NOT 'python'");
    let mut query = BooleanQuery::new();
    query.add_should(Box::new(TermQuery::new("body", "web")));
    query.add_should(Box::new(TermQuery::new("body", "database")));
    query.add_must_not(Box::new(TermQuery::new("body", "python")));
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("  Found {} matching documents:", results.total_hits);
    display_results(&results);

    // Clean up
    engine.close()?;
    println!("\nFull-text search example completed successfully!");

    Ok(())
}

/// Helper function to display search results in a formatted way
fn display_results(results: &sage::query::SearchResults) {
    for (i, hit) in results.hits.iter().enumerate() {
        println!("  {}. Score: {:.4}", i + 1, hit.score);

        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("id") {
                if let Some(id) = field_value.as_text() {
                    println!("     ID: {}", id);
                }
            }

            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("     Title: {}", title);
                }
            }

            if let Some(field_value) = doc.get_field("author") {
                if let Some(author) = field_value.as_text() {
                    println!("     Author: {}", author);
                }
            }

            if let Some(field_value) = doc.get_field("category") {
                if let Some(category) = field_value.as_text() {
                    println!("     Category: {}", category);
                }
            }

            if let Some(field_value) = doc.get_field("body") {
                if let Some(body) = field_value.as_text() {
                    // Display first 100 characters of body
                    let preview = if body.len() > 100 {
                        format!("{}...", &body[..100])
                    } else {
                        body.to_string()
                    };
                    println!("     Body: {}", preview);
                }
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
