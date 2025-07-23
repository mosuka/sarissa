//! BM25 Scorer example - demonstrates BM25 scoring algorithm and parameters.

use sarissa::error::Result;
use sarissa::index::index::IndexConfig;
use sarissa::prelude::*;
use sarissa::query::{BM25Scorer, Scorer};
use sarissa::schema::{IdField, TextField};
use sarissa::search::SearchEngine;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== BM25 Scorer Example - Relevance Scoring and Ranking ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a schema
    let mut schema = Schema::new();
    schema.add_field(
        "title",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field(
        "body",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field("author", Box::new(IdField::new()))?;

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    // Add documents with varying term frequencies for demonstration
    let documents = vec![
        Document::builder()
            .add_text("title", "Rust Programming Language")
            .add_text("body", "Rust is a programming language focused on safety and performance. Rust provides memory safety without garbage collection.")
            .add_text("author", "Mozilla")
            .build(),
        Document::builder()
            .add_text("title", "Advanced Rust Programming")
            .add_text("body", "Advanced Rust programming techniques include lifetimes, traits, and macros. Rust programming requires understanding ownership.")
            .add_text("author", "Expert")
            .build(),
        Document::builder()
            .add_text("title", "Python Programming Basics")
            .add_text("body", "Python is a versatile programming language. Python programming is beginner-friendly and widely used in data science.")
            .add_text("author", "Beginner")
            .build(),
        Document::builder()
            .add_text("title", "Programming Languages Comparison")
            .add_text("body", "Comparing different programming languages: Rust, Python, Java, and C++. Each programming language has its strengths.")
            .add_text("author", "Analyst")
            .build(),
        Document::builder()
            .add_text("title", "Short Title")
            .add_text("body", "Brief content with programming mentioned once.")
            .add_text("author", "Writer")
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    engine.add_documents(documents)?;
    engine.commit()?;

    println!("\n=== BM25 Scoring Examples ===\n");

    // Example 1: Default BM25 parameters
    println!("1. Default BM25 scoring parameters:");
    let default_scorer = BM25Scorer::new(5, 20, 5, 8.0, 5, 1.0); // Simulated realistic values
    println!(
        "   k1: {} (term frequency saturation point)",
        default_scorer.k1()
    );
    println!("   b: {} (field length normalization)", default_scorer.b());
    println!("   boost: {}", default_scorer.boost());

    // Example 2: Score calculation with different term frequencies
    println!("\n2. Score calculation with different term frequencies:");
    for tf in [1.0, 2.0, 3.0, 5.0, 10.0] {
        let score = default_scorer.score(0, tf);
        println!("   Term frequency {}: Score = {:.4}", tf, score);
    }

    // Example 3: Different k1 values (term frequency saturation)
    println!("\n3. Impact of k1 parameter (term frequency saturation):");
    let k1_values = [0.5, 1.2, 2.0, 3.0];
    for k1 in k1_values {
        let scorer = BM25Scorer::new(5, 20, 5, 8.0, 5, 1.0);
        // Note: In real implementation, k1 would be configurable
        let score = scorer.score(0, 3.0); // Fixed term frequency
        println!("   k1 = {}: Score = {:.4}", k1, score);
    }

    // Example 4: Different b values (field length normalization)
    println!("\n4. Impact of b parameter (field length normalization):");
    let b_values = [0.0, 0.25, 0.75, 1.0];
    for b in b_values {
        let scorer = BM25Scorer::new(5, 20, 5, 8.0, 5, 1.0);
        // Note: In real implementation, b would be configurable
        let score = scorer.score(0, 2.0); // Fixed term frequency
        println!("   b = {}: Score = {:.4}", b, score);
    }

    // Example 5: Real search with scoring
    println!("\n5. Real search results with BM25 scoring:");
    let results = engine.search_str("programming", "body")?;
    println!("   Search for 'programming' in body field:");
    println!("   Found {} results", results.total_hits);

    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {}", title);
                }
            }
        }
    }

    // Example 6: Query boost effects
    println!("\n6. Query boost effects:");
    let boosted_scorers = [
        ("Normal boost", BM25Scorer::new(5, 20, 5, 8.0, 5, 1.0)),
        ("High boost", BM25Scorer::new(5, 20, 5, 8.0, 5, 2.0)),
        ("Very high boost", BM25Scorer::new(5, 20, 5, 8.0, 5, 5.0)),
    ];

    for (description, scorer) in boosted_scorers {
        let score = scorer.score(0, 2.0);
        println!(
            "   {}: Score = {:.4} (boost: {})",
            description,
            score,
            scorer.boost()
        );
    }

    // Example 7: Term frequency vs Document frequency impact
    println!("\n7. Term frequency vs Document frequency impact:");
    let scenarios = [
        ("Common term", 4, 15, 5),   // High doc freq, medium term freq
        ("Rare term", 1, 3, 5),      // Low doc freq, medium term freq
        ("Very rare term", 1, 1, 5), // Very low doc freq, medium term freq
    ];

    for (description, doc_freq, term_freq, doc_count) in scenarios {
        let scorer = BM25Scorer::new(doc_freq, term_freq, doc_count, 8.0, 5, 1.0);
        let score = scorer.score(0, 2.0);
        println!("   {}: Score = {:.4}", description, score);
        println!(
            "     Doc freq: {}, Term freq: {}, Total docs: {}",
            doc_freq, term_freq, doc_count
        );
    }

    engine.close()?;
    println!("\nBM25 scorer example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_scorer_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
