//! WildcardQuery example - demonstrates pattern matching with * and ? wildcards.

use std::sync::Arc;

use tempfile::TempDir;

use platypus::analysis::analyzer::analyzer::Analyzer;
use platypus::analysis::analyzer::keyword::KeywordAnalyzer;
use platypus::analysis::analyzer::per_field::PerFieldAnalyzer;
use platypus::analysis::analyzer::standard::StandardAnalyzer;
use platypus::document::document::Document;
use platypus::document::field::TextOption;
use platypus::error::Result;
use platypus::lexical::engine::LexicalEngine;
use platypus::lexical::index::config::InvertedIndexConfig;
use platypus::lexical::index::config::LexicalIndexConfig;
use platypus::lexical::index::inverted::query::Query;
use platypus::lexical::index::inverted::query::wildcard::WildcardQuery;
use platypus::lexical::search::searcher::LexicalSearchRequest;
use platypus::storage::StorageConfig;
use platypus::storage::StorageFactory;
use platypus::storage::file::FileStorageConfig;

fn main() -> Result<()> {
    println!("=== WildcardQuery Example - Pattern Matching with Wildcards ===\n");

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

    // Add documents with various patterns for wildcard matching
    let documents = vec![
        Document::builder()
            .add_text(
                "title",
                "JavaScript Tutorial for Beginners",
                TextOption::default(),
            )
            .add_text("filename", "javascript_tutorial.pdf", TextOption::default())
            .add_text(
                "description",
                "Complete JavaScript programming guide",
                TextOption::default(),
            )
            .add_text("category", "programming", TextOption::default())
            .add_text("extension", "pdf", TextOption::default())
            .add_text("id", "file001", TextOption::default())
            .build(),
        Document::builder()
            .add_text(
                "title",
                "Python Programming Reference",
                TextOption::default(),
            )
            .add_text("filename", "python_reference.html", TextOption::default())
            .add_text(
                "description",
                "Comprehensive Python programming reference",
                TextOption::default(),
            )
            .add_text("category", "programming", TextOption::default())
            .add_text("extension", "html", TextOption::default())
            .add_text("id", "file002", TextOption::default())
            .build(),
        Document::builder()
            .add_text(
                "title",
                "Machine Learning Algorithms",
                TextOption::default(),
            )
            .add_text("filename", "ml_algorithms.docx", TextOption::default())
            .add_text(
                "description",
                "Understanding machine learning techniques",
                TextOption::default(),
            )
            .add_text("category", "data-science", TextOption::default())
            .add_text("extension", "docx", TextOption::default())
            .add_text("id", "file003", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Database Design Principles", TextOption::default())
            .add_text("filename", "database_design.pptx", TextOption::default())
            .add_text(
                "description",
                "Principles of good database design",
                TextOption::default(),
            )
            .add_text("category", "database", TextOption::default())
            .add_text("extension", "pptx", TextOption::default())
            .add_text("id", "file004", TextOption::default())
            .build(),
        Document::builder()
            .add_text(
                "title",
                "Web Development Best Practices",
                TextOption::default(),
            )
            .add_text("filename", "web_dev_practices.txt", TextOption::default())
            .add_text(
                "description",
                "Best practices for web development",
                TextOption::default(),
            )
            .add_text("category", "web-development", TextOption::default())
            .add_text("extension", "txt", TextOption::default())
            .add_text("id", "file005", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "React Component Patterns", TextOption::default())
            .add_text("filename", "react_patterns.jsx", TextOption::default())
            .add_text(
                "description",
                "Common patterns in React component development",
                TextOption::default(),
            )
            .add_text("category", "frontend", TextOption::default())
            .add_text("extension", "jsx", TextOption::default())
            .add_text("id", "file006", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "API Documentation Template", TextOption::default())
            .add_text("filename", "api_docs_template.md", TextOption::default())
            .add_text(
                "description",
                "Template for creating API documentation",
                TextOption::default(),
            )
            .add_text("category", "documentation", TextOption::default())
            .add_text("extension", "md", TextOption::default())
            .add_text("id", "file007", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Configuration Settings", TextOption::default())
            .add_text("filename", "app_config.json", TextOption::default())
            .add_text(
                "description",
                "Application configuration file",
                TextOption::default(),
            )
            .add_text("category", "configuration", TextOption::default())
            .add_text("extension", "json", TextOption::default())
            .add_text("id", "file008", TextOption::default())
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    for doc in documents {
        lexical_engine.add_document(doc)?;
    }

    lexical_engine.commit()?;

    println!("\n=== WildcardQuery Examples ===\n");

    // Example 1: Wildcard at the end (prefix matching)
    println!("1. Files starting with 'java' using 'java*' pattern:");
    let query = WildcardQuery::new("filename", "java*")?;
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
            && let Some(field_value) = doc.get_field("filename")
            && let Some(filename) = field_value.value.as_text()
        {
            println!("      Filename: {filename}");
        }
    }

    // Example 2: Wildcard at the beginning (suffix matching)
    println!("\n2. Files ending with '.pdf' using '*.pdf' pattern:");
    let query = WildcardQuery::new("filename", "*.pdf")?;
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
            && let Some(field_value) = doc.get_field("filename")
            && let Some(filename) = field_value.value.as_text()
        {
            println!("      Filename: {filename}");
        }
    }

    // Example 3: Wildcard in the middle
    println!("\n3. Files with 'web' followed by anything ending in '.txt' using 'web*.txt':");
    let query = WildcardQuery::new("filename", "web*.txt")?;
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
            && let Some(field_value) = doc.get_field("filename")
            && let Some(filename) = field_value.value.as_text()
        {
            println!("      Filename: {filename}");
        }
    }

    // Example 4: Single character wildcard (?)
    println!("\n4. Extensions with pattern '?sx' (jsx, tsx, etc.):");
    let query = WildcardQuery::new("extension", "?sx")?;
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
            && let Some(field_value) = doc.get_field("extension")
            && let Some(ext) = field_value.value.as_text()
        {
            println!("      Extension: {ext}");
        }
    }

    // Example 5: Multiple wildcards
    println!("\n5. Categories starting with 'prog' and ending with 'ing' using 'prog*ing':");
    let query = WildcardQuery::new("category", "prog*ing")?;
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
            && let Some(field_value) = doc.get_field("category")
            && let Some(category) = field_value.value.as_text()
        {
            println!("      Category: {category}");
        }
    }

    // Example 6: Complex pattern with both wildcards
    println!("\n6. Filenames with pattern '*_*.????' (underscore and 4-char extension):");
    let query = WildcardQuery::new("filename", "*_*.????")?;
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
            && let Some(field_value) = doc.get_field("filename")
            && let Some(filename) = field_value.value.as_text()
        {
            println!("      Filename: {filename}");
        }
    }

    // Example 7: Title pattern matching
    println!("\n7. Titles containing 'Development' using '*Development*':");
    let query = WildcardQuery::new("title", "*Development*")?;
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

    // Example 8: Single character matching
    println!("\n8. Extensions with exactly 3 characters using '???':");
    let query = WildcardQuery::new("extension", "???")?;
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
            && let Some(field_value) = doc.get_field("extension")
            && let Some(ext) = field_value.value.as_text()
        {
            println!("      Extension: {ext}");
        }
    }

    // Example 9: Match all files with any extension
    println!("\n9. All files with any extension using '*.*':");
    let query = WildcardQuery::new("filename", "*.*")?;
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
            && let Some(field_value) = doc.get_field("filename")
            && let Some(filename) = field_value.value.as_text()
        {
            println!("      Filename: {filename}");
        }
    }

    // Example 10: No matches
    println!("\n10. Pattern with no matches using 'xyz*abc':");
    let query = WildcardQuery::new("filename", "xyz*abc")?;
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>);
    let results = lexical_engine.search(request)?;

    println!("   Found {} results", results.total_hits);

    // Example 11: Count matching documents
    println!("\n11. Counting files with 'data' in filename using '*data*':");
    let query = WildcardQuery::new("filename", "*data*")?;
    let count =
        lexical_engine.count(LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>))?;
    println!("    Count: {count} files");

    lexical_engine.close()?;
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
