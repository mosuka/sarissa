//! Example demonstrating schema-less indexing (Lucene-style)
//!
//! This example shows how to use Sarissa without predefined schemas,
//! allowing maximum flexibility in document structure and field analyzers.

use sarissa::analysis::{KeywordAnalyzer, NoOpAnalyzer, StandardAnalyzer};
use sarissa::index::writer::{BasicIndexWriter, IndexWriter, WriterConfig};
use sarissa::schema::{Document, FieldValue};
use sarissa::storage::{MemoryStorage, StorageConfig};
use std::sync::Arc;

fn main() -> sarissa::error::Result<()> {
    println!("=== Schema-less Indexing Example ===\n");

    // Create storage and writer configuration
    let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
    let config = WriterConfig::default();

    // Create writer in schema-less mode (no schema required!)
    let mut writer = BasicIndexWriter::new_schemaless(storage, config)?;
    println!("✓ Created schema-less IndexWriter");

    // Prepare analyzers for different field types
    let _standard_analyzer = Arc::new(StandardAnalyzer::new()?);
    let keyword_analyzer = Arc::new(KeywordAnalyzer::new());
    let _noop_analyzer = Arc::new(NoOpAnalyzer::new());

    println!("\n=== Adding E-commerce Product Documents ===");

    // Product 1: Electronics
    let mut product1 = Document::new();
    // Use unified API - specify analyzer only when needed
    product1.add_field_with_analyzer(
        "id",
        FieldValue::Text("ELEC-001".to_string()),
        keyword_analyzer.clone(),
    );
    // Standard fields use default analyzer automatically
    product1.add_field(
        "title",
        FieldValue::Text("Wireless Bluetooth Headphones".to_string()),
    );
    product1.add_field(
        "description",
        FieldValue::Text("High-quality wireless headphones with noise cancellation".to_string()),
    );
    // Category needs exact matching
    product1.add_field_with_analyzer(
        "category",
        FieldValue::Text("electronics".to_string()),
        keyword_analyzer.clone(),
    );
    product1.add_field("price", FieldValue::Float(199.99));
    product1.add_field("in_stock", FieldValue::Boolean(true));

    writer.add_document(product1)?;
    println!("✓ Added electronics product");

    // Product 2: Book (different fields structure)
    let mut product2 = Document::new();
    product2.add_field_with_analyzer(
        "id",
        FieldValue::Text("BOOK-002".to_string()),
        keyword_analyzer.clone(),
    );
    // Text fields automatically get default analyzer (StandardAnalyzer)
    product2.add_field(
        "title",
        FieldValue::Text("The Rust Programming Language".to_string()),
    );
    product2.add_field(
        "author",
        FieldValue::Text("Steve Klabnik and Carol Nichols".to_string()),
    );
    // ISBN needs exact matching
    product2.add_field_with_analyzer(
        "isbn",
        FieldValue::Text("978-1718500440".to_string()),
        keyword_analyzer.clone(),
    );
    product2.add_field_with_analyzer(
        "category",
        FieldValue::Text("books".to_string()),
        keyword_analyzer.clone(),
    );
    product2.add_field("price", FieldValue::Float(39.99));
    product2.add_field("pages", FieldValue::Integer(552));

    writer.add_document(product2)?;
    println!("✓ Added book product");

    // User 1: Customer profile (completely different structure)
    let mut user1 = Document::new();
    user1.add_field_with_analyzer(
        "user_id",
        FieldValue::Text("USER-12345".to_string()),
        keyword_analyzer.clone(),
    );
    user1.add_field_with_analyzer(
        "email",
        FieldValue::Text("john.doe@example.com".to_string()),
        keyword_analyzer.clone(), // Email should be exact match
    );
    // Name and bio get default analyzer (searchable text)
    user1.add_field("full_name", FieldValue::Text("John Doe".to_string()));
    user1.add_field(
        "bio",
        FieldValue::Text(
            "Software engineer passionate about Rust and search technologies".to_string(),
        ),
    );
    user1.add_field("age", FieldValue::Integer(28));
    user1.add_field("premium_member", FieldValue::Boolean(true));

    writer.add_document(user1)?;
    println!("✓ Added user profile");

    println!("\n=== Using DocumentBuilder with Analyzers ===");

    // Blog post using DocumentBuilder - demonstrating unified API
    let blog_post = Document::builder()
        .add_text_with_analyzer(
            "post_id",
            "POST-789",
            keyword_analyzer.clone(), // Exact match for ID
        )
        .add_text("title", "Getting Started with Schema-less Search") // Uses default analyzer
        .add_text(
            "content",
            "Schema-less search engines provide incredible flexibility...",
        ) // Uses default analyzer
        .add_text_with_analyzer(
            "tags",
            "rust,search,tutorial",
            keyword_analyzer.clone(), // Exact match for tags
        )
        .add_text("author", "Jane Smith") // Uses default analyzer
        .build();

    writer.add_document(blog_post)?;
    println!("✓ Added blog post using DocumentBuilder");

    // Commit all documents
    writer.commit()?;
    println!("\n✓ Committed all documents to index");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schemaless_example() {
        let result = main();
        assert!(
            result.is_ok(),
            "Schema-less indexing example should run successfully"
        );
    }
}
