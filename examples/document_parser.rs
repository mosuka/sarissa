//! Example demonstrating DocumentParser usage.
//!
//! This example shows how to use DocumentParser to explicitly control
//! the document analysis process before indexing, similar to how QueryParser
//! is used for query analysis before searching.
//!
//! The flow is:
//! Document → DocumentParser → AnalyzedDocument → IndexWriter
//!
//! This is symmetric with:
//! Query String → QueryParser → Query → IndexReader

use std::sync::Arc;

use tempfile::TempDir;

use sage::analysis::analyzer::analyzer::Analyzer;
use sage::analysis::analyzer::keyword::KeywordAnalyzer;
use sage::analysis::analyzer::per_field::PerFieldAnalyzer;
use sage::analysis::analyzer::standard::StandardAnalyzer;
use sage::document::document::Document;
use sage::document::parser::DocumentParser;
use sage::error::Result;
use sage::lexical::index::IndexConfig;
use sage::lexical::index::advanced_writer::{AdvancedIndexWriter, AdvancedWriterConfig};
use sage::lexical::search::SearchRequest;
use sage::lexical::search::engine::LexicalEngine;

fn main() -> Result<()> {
    println!("=== Document Parser Example ===\n");

    // Step 1: Create per-field analyzer wrapper
    // Note: Reuse analyzer instances with Arc::clone to save memory
    let standard_analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);
    let keyword_analyzer: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
    let mut per_field_analyzer = PerFieldAnalyzer::new(Arc::clone(&standard_analyzer));
    per_field_analyzer.add_analyzer("id", Arc::clone(&keyword_analyzer));
    per_field_analyzer.add_analyzer("category", Arc::clone(&keyword_analyzer));

    let analyzer = Arc::new(per_field_analyzer);

    // Step 2: Create temporary directory and search engine
    let temp_dir = TempDir::new().unwrap();
    let engine = LexicalEngine::create_in_dir(temp_dir.path(), IndexConfig::default())?;

    // Get storage for creating custom writer
    let storage = engine.storage().clone();
    let config = AdvancedWriterConfig {
        analyzer: analyzer.clone(),
        ..Default::default()
    };
    let mut writer = AdvancedIndexWriter::new(storage, config)?;

    // Step 3: Create DocumentParser for explicit document parsing
    let doc_parser = DocumentParser::new(analyzer.clone());

    println!("Step 1: Creating documents and parsing them explicitly\n");

    // Create documents
    let docs = vec![
        Document::builder()
            .add_text("id", "BOOK-001")
            .add_text("title", "Rust Programming Language")
            .add_text("category", "programming")
            .build(),
        Document::builder()
            .add_text("id", "BOOK-002")
            .add_text("title", "Learning Rust")
            .add_text("category", "programming")
            .build(),
        Document::builder()
            .add_text("id", "ARTICLE-001")
            .add_text("title", "Introduction to Python")
            .add_text("category", "tutorial")
            .build(),
    ];

    // Analyze documents explicitly and add to index
    for (doc_id, doc) in docs.into_iter().enumerate() {
        println!("Analyzing document {doc_id}:");

        // Use DocumentParser to convert Document → AnalyzedDocument
        let analyzed_doc = doc_parser.parse(doc, doc_id as u64)?;

        // Show what was analyzed
        for (field, terms) in &analyzed_doc.field_terms {
            println!("  Field '{}': {} terms", field, terms.len());
            for term in terms {
                println!(
                    "    - '{}' (pos: {}, freq: {})",
                    term.term, term.position, term.frequency
                );
            }
        }

        // Add analyzed document to index
        writer.add_analyzed_document(analyzed_doc)?;
    }

    println!("\nStep 2: Committing to index\n");
    writer.commit()?;
    writer.close()?;

    // Step 4: Search using query parser
    println!("Step 3: Searching with query parser\n");

    // Search for "programming" in category field (KeywordAnalyzer)
    let query_str = "category:programming";
    println!("Query: {query_str}");

    use sage::query::parser::QueryParser;
    let parser = QueryParser::new();
    let query = parser.parse(query_str)?;

    let request = SearchRequest::new(query).load_documents(true);
    let results = engine.search(request)?;

    println!("Results: {} hits", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!("  {}. Doc {} (score: {:.4})", i + 1, hit.doc_id, hit.score);
        if let Some(doc) = &hit.document {
            if let Some(title) = doc.get_field("title")
                && let Some(title_text) = title.as_text()
            {
                println!("     Title: {title_text}");
            }
            if let Some(id) = doc.get_field("id")
                && let Some(id_text) = id.as_text()
            {
                println!("     ID: {id_text}");
            }
        }
    }

    println!("\n=== Summary ===");
    println!("DocumentParser provides symmetric API with QueryParser:");
    println!("  Index: Document → DocumentParser → AnalyzedDocument → Writer");
    println!("  Query: String → QueryParser → Query → Reader");
    println!("\nThis allows explicit control over document analysis,");
    println!("similar to how QueryParser provides control over query analysis.");

    Ok(())
}
