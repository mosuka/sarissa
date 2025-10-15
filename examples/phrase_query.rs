//! PhraseQuery example - demonstrates phrase search for exact word sequences.

use sage::full_text::index::IndexConfig;
use sage::full_text_search::SearchEngine;
use sage::full_text_search::SearchRequest;
use sage::prelude::*;
use sage::query::PhraseQuery;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== PhraseQuery Example - Exact Phrase Matching ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), IndexConfig::default())?;

    // Add documents with various phrases
    let documents = vec![
        Document::builder()
            .add_text("title", "Machine Learning Basics")
            .add_text("body", "Machine learning is a powerful tool for data analysis and artificial intelligence applications")
            .add_text("author", "Dr. Smith")
            .add_text("description", "An introduction to machine learning concepts and algorithms")
            .build(),
        Document::builder()
            .add_text("title", "Deep Learning Networks")
            .add_text("body", "Deep learning networks use artificial neural networks with multiple layers for complex pattern recognition")
            .add_text("author", "Prof. Johnson")
            .add_text("description", "Advanced techniques in deep learning and neural network architectures")
            .build(),
        Document::builder()
            .add_text("title", "Natural Language Processing")
            .add_text("body", "Natural language processing combines computational linguistics with machine learning and artificial intelligence")
            .add_text("author", "Dr. Wilson")
            .add_text("description", "Processing and understanding human language using computational methods")
            .build(),
        Document::builder()
            .add_text("title", "Computer Vision Applications")
            .add_text("body", "Computer vision applications include image recognition, object detection, and visual pattern analysis")
            .add_text("author", "Prof. Davis")
            .add_text("description", "Practical applications of computer vision in various industries")
            .build(),
        Document::builder()
            .add_text("title", "Data Science Fundamentals")
            .add_text("body", "Data science combines statistics, programming, and domain expertise to extract insights from data")
            .add_text("author", "Dr. Brown")
            .add_text("description", "Essential concepts and tools for data science practitioners")
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    engine.add_documents(documents)?;

    println!("\n=== PhraseQuery Examples ===\n");

    // Example 1: Simple two-word phrase
    println!("1. Searching for phrase 'machine learning' in body:");
    let query = PhraseQuery::new("body", vec!["machine".to_string(), "learning".to_string()]);
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

    // Example 3: Phrase in title field
    println!("\n3. Searching for phrase 'deep learning' in title:");
    let query = PhraseQuery::new("title", vec!["deep".to_string(), "learning".to_string()]);
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

    // Example 4: Phrase with common words
    println!("\n4. Searching for phrase 'data science' in description:");
    let query = PhraseQuery::new(
        "description",
        vec!["data".to_string(), "science".to_string()],
    );
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

    // Example 5: Non-existent phrase
    println!("\n5. Searching for non-existent phrase 'quantum computing':");
    let query = PhraseQuery::new("body", vec!["quantum".to_string(), "computing".to_string()]);
    let request = SearchRequest::new(Box::new(query));
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);

    // Example 6: Single word phrase (equivalent to TermQuery)
    println!("\n6. Searching for single word phrase 'intelligence' in body:");
    let query = PhraseQuery::new("body", vec!["intelligence".to_string()]);
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

    // Example 8: Count phrase matches
    println!("\n8. Counting documents with phrase 'computer vision':");
    let query = PhraseQuery::new("body", vec!["computer".to_string(), "vision".to_string()]);
    let count = engine.count(Box::new(query))?;
    println!("   Count: {count} documents");

    engine.close()?;
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
