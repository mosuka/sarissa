//! FuzzyQuery example - demonstrates approximate string matching with edit distance.

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
use platypus::lexical::index::config::{InvertedIndexConfig, LexicalIndexConfig};
use platypus::lexical::index::inverted::query::Query;
use platypus::lexical::index::inverted::query::fuzzy::FuzzyQuery;
use platypus::lexical::search::searcher::LexicalSearchRequest;
use platypus::storage::file::FileStorageConfig;
use platypus::storage::{StorageConfig, StorageFactory};

fn main() -> Result<()> {
    println!("=== FuzzyQuery Example - Approximate String Matching ===\n");

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
    let mut lexical_engine = LexicalEngine::new(storage, lexical_index_config)?;

    // Add documents with various spellings and terms for fuzzy matching
    let documents = vec![
        Document::builder()
            .add_text(
                "title",
                "JavaScript Programming Guide",
                TextOption::default(),
            )
            .add_text(
                "body",
                "Comprehensive guide to JavaScript development and programming techniques",
                TextOption::default(),
            )
            .add_text("author", "John Smith", TextOption::default())
            .add_text(
                "tags",
                "javascript programming tutorial",
                TextOption::default(),
            )
            .add_text("id", "doc001", TextOption::default())
            .build(),
        Document::builder()
            .add_text(
                "title",
                "Python Programming Fundamentals",
                TextOption::default(),
            )
            .add_text(
                "body",
                "Learn Python programming language from scratch with practical examples",
                TextOption::default(),
            )
            .add_text("author", "Alice Johnson", TextOption::default())
            .add_text("tags", "python programming beginner", TextOption::default())
            .add_text("id", "doc002", TextOption::default())
            .build(),
        Document::builder()
            .add_text(
                "title",
                "Machine Learning Algorithms",
                TextOption::default(),
            )
            .add_text(
                "body",
                "Understanding algorithms used in machine learning and artificial intelligence",
                TextOption::default(),
            )
            .add_text("author", "Bob Wilson", TextOption::default())
            .add_text(
                "tags",
                "machine-learning algorithms ai",
                TextOption::default(),
            )
            .add_text("id", "doc003", TextOption::default())
            .build(),
        Document::builder()
            .add_text(
                "title",
                "Database Management Systems",
                TextOption::default(),
            )
            .add_text(
                "body",
                "Introduction to database systems, SQL, and data management principles",
                TextOption::default(),
            )
            .add_text("author", "Carol Davis", TextOption::default())
            .add_text("tags", "database sql management", TextOption::default())
            .add_text("id", "doc004", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Web Development with React", TextOption::default())
            .add_text(
                "body",
                "Building modern web applications using React framework and components",
                TextOption::default(),
            )
            .add_text("author", "David Brown", TextOption::default())
            .add_text(
                "tags",
                "react web-development frontend",
                TextOption::default(),
            )
            .add_text("id", "doc005", TextOption::default())
            .build(),
        Document::builder()
            .add_text(
                "title",
                "Artificial Intelligence Overview",
                TextOption::default(),
            )
            .add_text(
                "body",
                "Introduction to artificial intelligence concepts, applications, and algorithms",
                TextOption::default(),
            )
            .add_text("author", "Eva Martinez", TextOption::default())
            .add_text(
                "tags",
                "artificial-intelligence overview concepts",
                TextOption::default(),
            )
            .add_text("id", "doc006", TextOption::default())
            .build(),
        Document::builder()
            .add_text(
                "title",
                "Software Engineering Principles",
                TextOption::default(),
            )
            .add_text(
                "body",
                "Best practices in software engineering, design patterns, and development",
                TextOption::default(),
            )
            .add_text("author", "Frank Miller", TextOption::default())
            .add_text(
                "tags",
                "software engineering principles",
                TextOption::default(),
            )
            .add_text("id", "doc007", TextOption::default())
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    for doc in documents {
        lexical_engine.add_document(doc)?;
    }

    // Commit changes to engine
    lexical_engine.commit()?;

    println!("\n=== FuzzyQuery Examples ===\n");

    // Example 1: Simple fuzzy search with small edit distance
    println!("1. Fuzzy search for 'javascritp' (typo for 'javascript') with edit distance 1:");
    let query = FuzzyQuery::new("body", "javascritp").max_edits(1);
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

    // Example 2: Fuzzy search with higher edit distance
    println!("\n2. Fuzzy search for 'programing' (missing 'm') with edit distance 2:");
    let query = FuzzyQuery::new("body", "programing").max_edits(2);
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

    // Example 3: Fuzzy search in title field
    println!("\n3. Fuzzy search for 'machne' (missing 'i') in title with edit distance 1:");
    let query = FuzzyQuery::new("title", "machne").max_edits(1);
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

    // Example 4: Fuzzy search for author names
    println!("\n4. Fuzzy search for 'Jon' (should match 'John') in author with edit distance 1:");
    let query = FuzzyQuery::new("author", "Jon").max_edits(1);
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

    // Example 5: Fuzzy search with various misspellings
    println!("\n5. Fuzzy search for 'algoritm' (missing 'h') with edit distance 2:");
    let query = FuzzyQuery::new("body", "algoritm").max_edits(2);
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

    // Example 6: Fuzzy search in tags
    println!("\n6. Fuzzy search for 'artifical' (missing 'i') in tags with edit distance 1:");
    let query = FuzzyQuery::new("tags", "artifical").max_edits(1);
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

    // Example 7: Fuzzy search with exact match (edit distance 0)
    println!("\n7. Fuzzy search for exact 'python' with edit distance 0:");
    let query = FuzzyQuery::new("body", "python").max_edits(0);
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

    // Example 8: Fuzzy search with high edit distance (more permissive)
    println!("\n8. Fuzzy search for 'databse' (missing 'a') with edit distance 3:");
    let query = FuzzyQuery::new("body", "databse").max_edits(3);
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

    // Example 9: No fuzzy matches found
    println!("\n9. Fuzzy search for 'xyz123' (no similar terms) with edit distance 2:");
    let query = FuzzyQuery::new("body", "xyz123").max_edits(2);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>);
    let results = lexical_engine.search(request)?;

    println!("   Found {} results", results.total_hits);

    // Example 10: Count fuzzy matches
    println!("\n10. Counting documents with fuzzy match for 'developement' (extra 'e'):");
    let query = FuzzyQuery::new("body", "developement").max_edits(2);
    let count =
        lexical_engine.count(LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>))?;
    println!("    Count: {count} documents");

    lexical_engine.close()?;
    println!("\nFuzzyQuery example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_query_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
