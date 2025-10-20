//! Example demonstrating schema-less indexing (Lucene-style)
//!
//! This example shows how to use Sage without predefined schemas,
//! allowing maximum flexibility in document structure.
//! Analyzers are configured at the writer level using PerFieldAnalyzer.

use sage::analysis::analyzer::keyword::KeywordAnalyzer;
use sage::analysis::analyzer::per_field::PerFieldAnalyzer;
use sage::analysis::analyzer::standard::StandardAnalyzer;
use sage::document::document::Document;
use sage::document::field_value::FieldValue;
use sage::lexical::index::advanced_writer::{AdvancedIndexWriter, AdvancedWriterConfig};
use sage::storage::memory::MemoryStorage;
use sage::storage::traits::StorageConfig;
use std::sync::Arc;

fn main() -> sage::error::Result<()> {
    println!("=== Schema-less Indexing Example ===\n");

    // Configure per-field analyzers using PerFieldAnalyzer (Lucene-style)
    let mut per_field_analyzer = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new()?));
    per_field_analyzer.add_analyzer("id", Arc::new(KeywordAnalyzer::new()));
    per_field_analyzer.add_analyzer("category", Arc::new(KeywordAnalyzer::new()));
    per_field_analyzer.add_analyzer("isbn", Arc::new(KeywordAnalyzer::new()));
    per_field_analyzer.add_analyzer("user_id", Arc::new(KeywordAnalyzer::new()));
    per_field_analyzer.add_analyzer("email", Arc::new(KeywordAnalyzer::new()));
    per_field_analyzer.add_analyzer("post_id", Arc::new(KeywordAnalyzer::new()));

    // Create storage and writer configuration
    let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
    let config = AdvancedWriterConfig {
        analyzer: Arc::new(per_field_analyzer),
        ..Default::default()
    };

    // Create writer in schema-less mode (no schema required!)
    let mut writer = AdvancedIndexWriter::new(storage, config)?;
    println!("✓ Created schema-less IndexWriter with PerFieldAnalyzer");

    println!("\n=== Adding E-commerce Product Documents ===");

    // Product 1: Electronics
    let mut product1 = Document::new();
    product1.add_field("id", FieldValue::Text("ELEC-001".to_string()));
    product1.add_field(
        "title",
        FieldValue::Text("Wireless Bluetooth Headphones".to_string()),
    );
    product1.add_field(
        "description",
        FieldValue::Text("High-quality wireless headphones with noise cancellation".to_string()),
    );
    product1.add_field("category", FieldValue::Text("electronics".to_string()));
    product1.add_field("price", FieldValue::Float(199.99));
    product1.add_field("in_stock", FieldValue::Boolean(true));

    writer.add_document(product1)?;
    println!("✓ Added electronics product");

    // Product 2: Book (different fields structure)
    let mut product2 = Document::new();
    product2.add_field("id", FieldValue::Text("BOOK-002".to_string()));
    product2.add_field(
        "title",
        FieldValue::Text("The Rust Programming Language".to_string()),
    );
    product2.add_field(
        "author",
        FieldValue::Text("Steve Klabnik and Carol Nichols".to_string()),
    );
    product2.add_field("isbn", FieldValue::Text("978-1718500440".to_string()));
    product2.add_field("category", FieldValue::Text("books".to_string()));
    product2.add_field("price", FieldValue::Float(39.99));
    product2.add_field("pages", FieldValue::Integer(552));

    writer.add_document(product2)?;
    println!("✓ Added book product");

    // User 1: Customer profile (completely different structure)
    let mut user1 = Document::new();
    user1.add_field("user_id", FieldValue::Text("USER-12345".to_string()));
    user1.add_field(
        "email",
        FieldValue::Text("john.doe@example.com".to_string()),
    );
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

    println!("\n=== Using DocumentBuilder ===");

    // Blog post using DocumentBuilder
    let blog_post = Document::builder()
        .add_text("post_id", "POST-789")
        .add_text("title", "Getting Started with Schema-less Search")
        .add_text(
            "content",
            "Schema-less search engines provide incredible flexibility...",
        )
        .add_text("tags", "rust,search,tutorial")
        .add_text("author", "Jane Smith")
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
