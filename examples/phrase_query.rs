//! PhraseQuery example - demonstrates phrase search for exact word sequences.

use std::sync::Arc;

use tempfile::TempDir;

use sarissa::analysis::analyzer::analyzer::Analyzer;
use sarissa::analysis::analyzer::keyword::KeywordAnalyzer;
use sarissa::analysis::analyzer::per_field::PerFieldAnalyzer;
use sarissa::analysis::analyzer::standard::StandardAnalyzer;
use sarissa::document::document::Document;
use sarissa::document::field::TextOption;
use sarissa::error::Result;
use sarissa::lexical::engine::LexicalEngine;
use sarissa::lexical::index::config::InvertedIndexConfig;
use sarissa::lexical::index::config::LexicalIndexConfig;
use sarissa::lexical::index::inverted::query::Query;
use sarissa::lexical::index::inverted::query::phrase::PhraseQuery;
use sarissa::lexical::search::searcher::LexicalSearchRequest;
use sarissa::storage::StorageConfig;
use sarissa::storage::StorageFactory;
use sarissa::storage::file::FileStorageConfig;

fn main() -> Result<()> {
    println!("=== PhraseQuery Example - Exact Phrase Matching ===\n");

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

    // Add documents with various phrases
    let documents = vec![
        Document::builder()
            .add_text("title", "Machine Learning Basics", TextOption::default())
            .add_text("body", "Machine learning is a powerful tool for data analysis and artificial intelligence applications", TextOption::default())
            .add_text("author", "Dr. Smith", TextOption::default())
            .add_text("description", "An introduction to machine learning concepts and algorithms", TextOption::default())
            .add_text("id", "001", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Deep Learning Networks", TextOption::default())
            .add_text("body", "Deep learning networks use artificial neural networks with multiple layers for complex pattern recognition", TextOption::default())
            .add_text("author", "Prof. Johnson", TextOption::default())
            .add_text("description", "Advanced techniques in deep learning and neural network architectures", TextOption::default())
            .add_text("id", "002", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Natural Language Processing", TextOption::default())
            .add_text("body", "Natural language processing combines computational linguistics with machine learning and artificial intelligence", TextOption::default())
            .add_text("author", "Dr. Wilson", TextOption::default())
            .add_text("description", "Processing and understanding human language using computational methods", TextOption::default())
            .add_text("id", "003", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Computer Vision Applications", TextOption::default())
            .add_text("body", "Computer vision applications include image recognition, object detection, and visual pattern analysis", TextOption::default())
            .add_text("author", "Prof. Davis", TextOption::default())
            .add_text("description", "Practical applications of computer vision in various industries", TextOption::default())
            .add_text("id", "004", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Data Science Fundamentals", TextOption::default())
            .add_text("body", "Data science combines statistics, programming, and domain expertise to extract insights from data", TextOption::default())
            .add_text("author", "Dr. Brown", TextOption::default())
            .add_text("description", "Essential concepts and tools for data science practitioners", TextOption::default())
            .add_text("id", "005", TextOption::default())
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    for doc in documents {
        lexical_engine.add_document(doc)?;
    }

    lexical_engine.commit()?;

    println!("\n=== PhraseQuery Examples ===\n");

    // Example 1: Simple two-word phrase
    println!("1. Searching for phrase 'machine learning' in body:");
    let query = PhraseQuery::new("body", vec!["machine".to_string(), "learning".to_string()]);
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

    // Example 2: Three-word phrase
    println!("\n2. Searching for phrase 'artificial neural networks' in body:");
    let query = PhraseQuery::new(
        "body",
        vec![
            "artificial".to_string(),
            "neural".to_string(),
            "networks".to_string(),
        ],
    );
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

    // Example 3: Phrase in title field
    println!("\n3. Searching for phrase 'deep learning' in title:");
    let query = PhraseQuery::new("title", vec!["deep".to_string(), "learning".to_string()]);
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

    // Example 4: Phrase with common words
    println!("\n4. Searching for phrase 'data science' in description:");
    let query = PhraseQuery::new(
        "description",
        vec!["data".to_string(), "science".to_string()],
    );
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

    // Example 5: Non-existent phrase
    println!("\n5. Searching for non-existent phrase 'quantum computing':");
    let query = PhraseQuery::new("body", vec!["quantum".to_string(), "computing".to_string()]);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>);
    let results = lexical_engine.search(request)?;

    println!("   Found {} results", results.total_hits);

    // Example 6: Single word phrase (equivalent to TermQuery)
    println!("\n6. Searching for single word phrase 'intelligence' in body:");
    let query = PhraseQuery::new("body", vec!["intelligence".to_string()]);
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

    // Example 7: Longer phrase search
    println!("\n7. Searching for long phrase 'extract insights from data' in body:");
    let query = PhraseQuery::new(
        "body",
        vec![
            "extract".to_string(),
            "insights".to_string(),
            "from".to_string(),
            "data".to_string(),
        ],
    );
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

    // Example 8: Count phrase matches
    println!("\n8. Counting documents with phrase 'computer vision':");
    let query = PhraseQuery::new("body", vec!["computer".to_string(), "vision".to_string()]);
    let count =
        lexical_engine.count(LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>))?;
    println!("   Count: {count} documents");

    lexical_engine.close()?;
    println!("\nPhraseQuery example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phrase_query_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
