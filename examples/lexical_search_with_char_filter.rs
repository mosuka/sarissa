//! Lexical search example demonstrating `CharFilter` usage.
//!
//! This example shows how to use `CharFilter` to normalize text before tokenization.
//! We demonstrate:
//! 1. `MappingCharFilter`: "ph" -> "f" (phonetic normalization)
//! 2. `JapaneseIterationMarkCharFilter`: "佐々木" -> "佐佐木" (iteration mark expansion)

use std::collections::HashMap;
use std::sync::Arc;

use tempfile::TempDir;

use sarissa::analysis::analyzer::analyzer::Analyzer;
use sarissa::analysis::analyzer::pipeline::PipelineAnalyzer;
use sarissa::analysis::char_filter::japanese_iteration_mark::JapaneseIterationMarkCharFilter;
use sarissa::analysis::char_filter::mapping::MappingCharFilter;
use sarissa::analysis::token_filter::lowercase::LowercaseFilter;
use sarissa::analysis::tokenizer::unicode_word::UnicodeWordTokenizer;
use sarissa::document::document::Document;
use sarissa::document::field::TextOption;
use sarissa::error::Result;
use sarissa::lexical::engine::LexicalEngine;
use sarissa::lexical::index::config::{InvertedIndexConfig, LexicalIndexConfig};
use sarissa::lexical::index::inverted::query::Query;
use sarissa::lexical::index::inverted::query::phrase::PhraseQuery;
use sarissa::lexical::index::inverted::query::term::TermQuery;
use sarissa::lexical::search::searcher::LexicalSearchRequest;
use sarissa::storage::file::FileStorageConfig;
use sarissa::storage::{StorageConfig, StorageFactory};

fn main() -> Result<()> {
    println!("=== Lexical Search with CharFilter Example ===\n");

    // 1. Setup Analyzer with CharFilters
    println!("Step 1: Setup Analyzer with CharFilters");

    // Define mappings for MappingCharFilter (e.g., phonetic normalization)
    let mut mapping = HashMap::new();
    mapping.insert("ph".to_string(), "f".to_string());
    mapping.insert("qu".to_string(), "k".to_string());

    let mapping_filter = Arc::new(MappingCharFilter::new(mapping)?);

    // Japanese Iteration Mark Filter
    let iteration_mark_filter = Arc::new(JapaneseIterationMarkCharFilter::new(true, true));

    // Create PipelineAnalyzer
    // Pipeline: CharFilters -> Tokenizer -> TokenFilters
    let tokenizer = Arc::new(UnicodeWordTokenizer::new());
    let analyzer = PipelineAnalyzer::new(tokenizer)
        .add_char_filter(mapping_filter)
        .add_char_filter(iteration_mark_filter)
        .add_filter(Arc::new(LowercaseFilter::new()));

    let analyzer_arc: Arc<dyn Analyzer> = Arc::new(analyzer);
    println!(
        "Created PipelineAnalyzer with MappingCharFilter and JapaneseIterationMarkCharFilter.\n"
    );

    // Debug: Inspect tokens
    let debug_texts = vec!["photography", "佐々木"];
    for text in debug_texts {
        println!("Debug analysis for '{}':", text);

        let stream = analyzer_arc.analyze(text)?;
        for token in stream {
            println!(
                "  Token: [{}] ({}, {})",
                token.text, token.start_offset, token.end_offset
            );
        }
    }

    // 2. Create Index
    println!("Step 2: Create Index");
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    let config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer: analyzer_arc,
        ..Default::default()
    });
    let engine = LexicalEngine::new(storage, config)?;

    // 3. Add Documents
    println!("Step 3: Add Documents");
    let documents = vec![
        Document::builder()
            .add_text("id", "doc1", TextOption::default())
            .add_text(
                "text",
                "I like photography and philosophy.",
                TextOption::default(),
            )
            .build(),
        Document::builder()
            .add_text("id", "doc2", TextOption::default())
            .add_text("text", "The quick brown fox.", TextOption::default())
            .build(),
        Document::builder()
            .add_text("id", "doc3", TextOption::default())
            .add_text("text", "これは佐々木さんの本です。", TextOption::default())
            .build(),
    ];

    for doc in documents {
        engine.add_document(doc)?;
    }
    engine.commit()?;
    println!("Documents committed.\n");

    // 4. Search Demonstrations
    println!("Step 4: Search Demonstrations");

    // Demo 1: MappingCharFilter "ph" -> "f"
    // Original text: "photography" -> Normalized to "fotografy"
    // Search query: "fotografy" should match doc1
    println!("\n[Demo 1] mapping: 'ph' -> 'f'");
    println!("Searching for 'fotografy' (matches 'photography' in doc1)");

    let query = TermQuery::new("text", "fotografy");
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = engine.search(request)?;

    println!("Found {} hits.", results.total_hits);
    for hit in results.hits {
        if let Some(doc) = hit.document {
            if let Some(text) = doc.get_field("text").and_then(|f| f.value.as_text()) {
                println!("  Matches: {}", text);
            }
        }
    }

    // Demo 2: Japanese Iteration Mark "々" -> "佐"
    // Original text: "佐々木" -> Normalized to "佐佐木"
    // Tokenizer splits "佐佐木" into ["佐", "佐", "木"]
    println!("\n[Demo 2] iteration mark: '々' -> Previous Char");
    println!("Searching for '佐佐木' (as phrase ['佐', '佐', '木'])");

    // We use PhraseQuery because tokenizer splits the normalized text into multiple tokens.
    let query = PhraseQuery::new(
        "text",
        vec!["佐".to_string(), "佐".to_string(), "木".to_string()],
    );
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = engine.search(request)?;

    println!("Found {} hits.", results.total_hits);
    for hit in results.hits {
        if let Some(doc) = hit.document {
            if let Some(text) = doc.get_field("text").and_then(|f| f.value.as_text()) {
                println!("  Matches: {}", text);
            }
        }
    }

    // Demo 3: Verify "々" does not exist
    println!("\n[Demo 3] Verify '々' is gone");
    let query = TermQuery::new("text", "々");
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = engine.search(request)?;
    println!("Found {} hits for '々' (expected 0).", results.total_hits);

    Ok(())
}
