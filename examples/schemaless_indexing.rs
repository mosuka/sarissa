//! Schema-less indexing example using Platypus's PerFieldAnalyzer pipeline.
//!
//! Demonstrates how to ingest heterogeneous documents without predefined
//! schemas. Analyzer selection happens per field at write time, mirroring
//! Lucene's flexible approach.

use std::sync::Arc;

use platypus::analysis::analyzer::analyzer::Analyzer;
use platypus::analysis::analyzer::keyword::KeywordAnalyzer;
use platypus::analysis::analyzer::per_field::PerFieldAnalyzer;
use platypus::analysis::analyzer::standard::StandardAnalyzer;
use platypus::document::document::Document;
use platypus::document::field::{Field, FieldValue, TextOption};
use platypus::lexical::engine::LexicalEngine;
use platypus::lexical::index::config::{InvertedIndexConfig, LexicalIndexConfig};
use platypus::storage::file::FileStorageConfig;
use platypus::storage::{StorageConfig, StorageFactory};
use tempfile::TempDir;

fn main() -> platypus::error::Result<()> {
    println!("=== Schema-less Indexing Example ===\n");

    // Create a storage backend
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    // Create an analyzer
    let standard_analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);
    let keyword_analyzer: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
    let mut per_field_analyzer = PerFieldAnalyzer::new(Arc::clone(&standard_analyzer));
    per_field_analyzer.add_analyzer("id", Arc::clone(&keyword_analyzer));
    per_field_analyzer.add_analyzer("category", Arc::clone(&keyword_analyzer));
    per_field_analyzer.add_analyzer("isbn", Arc::clone(&keyword_analyzer));
    per_field_analyzer.add_analyzer("user_id", Arc::clone(&keyword_analyzer));
    per_field_analyzer.add_analyzer("email", Arc::clone(&keyword_analyzer));
    per_field_analyzer.add_analyzer("post_id", Arc::clone(&keyword_analyzer));

    // Create a lexical engine
    let lexical_index_config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer: Arc::new(per_field_analyzer.clone()),
        ..InvertedIndexConfig::default()
    });
    let lexical_engine = LexicalEngine::new(storage, lexical_index_config)?;

    // Configure per-field analyzers using PerFieldAnalyzer (Lucene-style)
    let mut per_field_analyzer = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new()?));
    per_field_analyzer.add_analyzer("id", Arc::new(KeywordAnalyzer::new()));
    per_field_analyzer.add_analyzer("category", Arc::new(KeywordAnalyzer::new()));
    per_field_analyzer.add_analyzer("isbn", Arc::new(KeywordAnalyzer::new()));
    per_field_analyzer.add_analyzer("user_id", Arc::new(KeywordAnalyzer::new()));
    per_field_analyzer.add_analyzer("email", Arc::new(KeywordAnalyzer::new()));
    per_field_analyzer.add_analyzer("post_id", Arc::new(KeywordAnalyzer::new()));

    println!("✓ Created schema-less LexicalEngine with PerFieldAnalyzer");

    println!("\n=== Adding E-commerce Product Documents ===");

    // Product 1: Electronics
    let mut product1 = Document::new();
    product1.add_field(
        "id",
        Field::with_default_option(FieldValue::Text("ELEC-001".to_string())),
    );
    product1.add_field(
        "title",
        Field::with_default_option(FieldValue::Text(
            "Wireless Bluetooth Headphones".to_string(),
        )),
    );
    product1.add_field(
        "description",
        Field::with_default_option(FieldValue::Text(
            "High-quality wireless headphones with noise cancellation".to_string(),
        )),
    );
    product1.add_field(
        "category",
        Field::with_default_option(FieldValue::Text("electronics".to_string())),
    );
    product1.add_field(
        "price",
        Field::with_default_option(FieldValue::Float(199.99)),
    );
    product1.add_field(
        "in_stock",
        Field::with_default_option(FieldValue::Boolean(true)),
    );

    lexical_engine.add_document(product1)?;
    println!("✓ Added electronics product");

    // Product 2: Book (different fields structure)
    let mut product2 = Document::new();
    product2.add_field(
        "id",
        Field::with_default_option(FieldValue::Text("BOOK-002".to_string())),
    );
    product2.add_field(
        "title",
        Field::with_default_option(FieldValue::Text(
            "The Rust Programming Language".to_string(),
        )),
    );
    product2.add_field(
        "author",
        Field::with_default_option(FieldValue::Text(
            "Steve Klabnik and Carol Nichols".to_string(),
        )),
    );
    product2.add_field(
        "isbn",
        Field::with_default_option(FieldValue::Text("978-1718500440".to_string())),
    );
    product2.add_field(
        "category",
        Field::with_default_option(FieldValue::Text("books".to_string())),
    );
    product2.add_field(
        "price",
        Field::with_default_option(FieldValue::Float(39.99)),
    );
    product2.add_field(
        "pages",
        Field::with_default_option(FieldValue::Integer(552)),
    );

    lexical_engine.add_document(product2)?;
    println!("✓ Added book product");

    // User 1: Customer profile (completely different structure)
    let mut user1 = Document::new();
    user1.add_field(
        "user_id",
        Field::with_default_option(FieldValue::Text("USER-12345".to_string())),
    );
    user1.add_field(
        "email",
        Field::with_default_option(FieldValue::Text("john.doe@example.com".to_string())),
    );
    user1.add_field(
        "full_name",
        Field::with_default_option(FieldValue::Text("John Doe".to_string())),
    );
    user1.add_field(
        "bio",
        Field::with_default_option(FieldValue::Text(
            "Software engineer passionate about Rust and search technologies".to_string(),
        )),
    );
    user1.add_field("age", Field::with_default_option(FieldValue::Integer(28)));
    user1.add_field(
        "premium_member",
        Field::with_default_option(FieldValue::Boolean(true)),
    );

    lexical_engine.add_document(user1)?;
    println!("✓ Added user profile");

    println!("\n=== Using DocumentBuilder ===");

    // Blog post using DocumentBuilder
    let blog_post = Document::builder()
        .add_text("post_id", "POST-789", TextOption::default())
        .add_text(
            "title",
            "Getting Started with Schema-less Search",
            TextOption::default(),
        )
        .add_text(
            "content",
            "Schema-less search engines provide incredible flexibility...",
            TextOption::default(),
        )
        .add_text("tags", "rust,search,tutorial", TextOption::default())
        .add_text("author", "Jane Smith", TextOption::default())
        .build();

    lexical_engine.add_document(blog_post)?;
    println!("✓ Added blog post using DocumentBuilder");

    // Commit all documents
    lexical_engine.commit()?;
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
