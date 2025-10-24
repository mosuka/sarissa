//! Hybrid Search Example - combining lexical and vector search
//!
//! This example demonstrates:
//! - Creating a hybrid search configuration
//! - Understanding hybrid search result types and scoring
//! - Configuring score normalization strategies
//! - Setting up weights for keyword vs. vector search components
//!
//! Note: This example shows the hybrid search configuration and types.
//! Full hybrid search functionality requires integration with both lexical
//! and vector indexes, which is currently under development.
//!
//! To run this example:
//! ```bash
//! cargo run --example hybrid_search
//! ```

use std::collections::HashMap;

use sage::error::Result;
use sage::hybrid::config::{HybridSearchConfig, ScoreNormalization};
use sage::hybrid::engine::HybridSearchEngine;
use sage::hybrid::types::{HybridSearchResult, HybridSearchResults};

fn main() -> Result<()> {
    println!("=== Hybrid Search Configuration Example ===\n");
    println!(
        "This example demonstrates hybrid search configuration and result types.\n\
         Full end-to-end hybrid search is under development.\n"
    );

    // Step 1: Create different hybrid search configurations
    println!("Step 1: Creating hybrid search configurations...\n");

    // Configuration 1: Balanced approach
    println!("--- Configuration 1: Balanced (Default) ---");
    let config_balanced = HybridSearchConfig::default();
    println!("  Keyword weight: {}", config_balanced.keyword_weight);
    println!("  Vector weight: {}", config_balanced.vector_weight);
    println!("  Min keyword score: {}", config_balanced.min_keyword_score);
    println!(
        "  Min vector similarity: {}",
        config_balanced.min_vector_similarity
    );
    println!("  Max results: {}", config_balanced.max_results);
    println!("  Require both matches: {}", config_balanced.require_both);
    println!("  Normalization: {:?}\n", config_balanced.normalization);

    // Configuration 2: Keyword-focused
    println!("--- Configuration 2: Keyword-Focused ---");
    let config_keyword_focused = HybridSearchConfig {
        keyword_weight: 0.8,
        vector_weight: 0.2,
        min_keyword_score: 0.5,
        min_vector_similarity: 0.3,
        max_results: 20,
        require_both: false,
        normalization: ScoreNormalization::MinMax,
        ..Default::default()
    };
    println!(
        "  Keyword weight: {}",
        config_keyword_focused.keyword_weight
    );
    println!("  Vector weight: {}", config_keyword_focused.vector_weight);
    println!(
        "  Min keyword score: {}",
        config_keyword_focused.min_keyword_score
    );
    println!(
        "  Description: Emphasizes exact keyword matching (80%) over semantic similarity (20%)\n"
    );

    // Configuration 3: Semantic-focused
    println!("--- Configuration 3: Semantic-Focused ---");
    let config_semantic_focused = HybridSearchConfig {
        keyword_weight: 0.3,
        vector_weight: 0.7,
        min_keyword_score: 0.0,
        min_vector_similarity: 0.5,
        max_results: 20,
        require_both: false,
        normalization: ScoreNormalization::MinMax,
        ..Default::default()
    };
    println!(
        "  Keyword weight: {}",
        config_semantic_focused.keyword_weight
    );
    println!("  Vector weight: {}", config_semantic_focused.vector_weight);
    println!(
        "  Min vector similarity: {}",
        config_semantic_focused.min_vector_similarity
    );
    println!("  Description: Emphasizes semantic similarity (70%) over exact keywords (30%)\n");

    // Configuration 4: Strict matching
    println!("--- Configuration 4: Strict Matching ---");
    let config_strict = HybridSearchConfig {
        keyword_weight: 0.5,
        vector_weight: 0.5,
        min_keyword_score: 0.7,
        min_vector_similarity: 0.7,
        max_results: 10,
        require_both: true, // Both keyword and vector must match
        normalization: ScoreNormalization::MinMax,
        ..Default::default()
    };
    println!("  Keyword weight: {}", config_strict.keyword_weight);
    println!("  Vector weight: {}", config_strict.vector_weight);
    println!("  Min keyword score: {}", config_strict.min_keyword_score);
    println!(
        "  Min vector similarity: {}",
        config_strict.min_vector_similarity
    );
    println!("  Require both matches: {}", config_strict.require_both);
    println!("  Description: Requires both keyword and vector matches with high thresholds\n");

    // Step 2: Demonstrate score normalization strategies
    println!("{}", "=".repeat(80));
    println!("\nStep 2: Score Normalization Strategies\n");

    println!("--- Normalization Strategy: None ---");
    println!("  Uses raw scores from both search types");
    println!("  Pros: Preserves original score magnitudes");
    println!("  Cons: May favor one search type if score ranges differ significantly\n");

    println!("--- Normalization Strategy: MinMax ---");
    println!("  Normalizes scores to [0, 1] range");
    println!("  Formula: (score - min) / (max - min)");
    println!("  Pros: Puts both score types on equal footing");
    println!("  Cons: Sensitive to outliers\n");

    println!("--- Normalization Strategy: ZScore ---");
    println!("  Normalizes using mean and standard deviation");
    println!("  Formula: (score - mean) / std_dev");
    println!("  Pros: Accounts for score distribution");
    println!("  Cons: Requires sufficient data for meaningful statistics\n");

    println!("--- Normalization Strategy: Rank ---");
    println!("  Uses rank positions instead of scores");
    println!("  Formula: 1.0 / (rank + k)  (Reciprocal Rank Fusion)");
    println!("  Pros: Robust to score scale differences");
    println!("  Cons: Loses information about score magnitudes\n");

    // Step 3: Create sample hybrid search results
    println!("{}", "=".repeat(80));
    println!("\nStep 3: Hybrid Search Result Types\n");

    // Create sample results
    let mut sample_results = Vec::new();

    // Result 1: Strong in both keyword and vector
    let mut doc1_fields = HashMap::new();
    doc1_fields.insert("title".to_string(), "Rust Programming Guide".to_string());
    doc1_fields.insert("category".to_string(), "programming".to_string());

    let result1 = HybridSearchResult::new(1, 0.85)
        .with_keyword_score(0.8)
        .with_vector_similarity(0.9)
        .with_document(doc1_fields);
    sample_results.push(result1);

    // Result 2: Strong in keyword, weak in vector
    let mut doc2_fields = HashMap::new();
    doc2_fields.insert("title".to_string(), "Web Development with Rust".to_string());
    doc2_fields.insert("category".to_string(), "web-development".to_string());

    let result2 = HybridSearchResult::new(2, 0.72)
        .with_keyword_score(0.9)
        .with_vector_similarity(0.6)
        .with_document(doc2_fields);
    sample_results.push(result2);

    // Result 3: Weak in keyword, strong in vector
    let mut doc3_fields = HashMap::new();
    doc3_fields.insert(
        "title".to_string(),
        "Systems Programming Concepts".to_string(),
    );
    doc3_fields.insert("category".to_string(), "programming".to_string());

    let result3 = HybridSearchResult::new(3, 0.68)
        .with_keyword_score(0.5)
        .with_vector_similarity(0.85)
        .with_document(doc3_fields);
    sample_results.push(result3);

    // Create hybrid search results collection
    let mut results = HybridSearchResults::new(
        sample_results,
        100,                            // total_searched
        25,                             // keyword_matches
        30,                             // vector_matches
        150,                            // query_time_ms
        "rust programming".to_string(), // query_text
    );

    println!("--- Sample Hybrid Search Results ---");
    println!("Query: \"{}\"", results.query_text);
    println!("Total documents searched: {}", results.total_searched);
    println!("Keyword matches: {}", results.keyword_matches);
    println!("Vector matches: {}", results.vector_matches);
    println!("Query time: {} ms", results.query_time_ms);
    println!("Results returned: {}\n", results.len());

    println!("Individual Results:");
    for (i, result) in results.results.iter().enumerate() {
        println!("\n  Result {}:", i + 1);
        println!("    Doc ID: {}", result.doc_id);
        println!("    Hybrid Score: {:.4}", result.hybrid_score);
        if let Some(kw_score) = result.keyword_score {
            println!("    Keyword Score: {:.4}", kw_score);
        }
        if let Some(vec_sim) = result.vector_similarity {
            println!("    Vector Similarity: {:.4}", vec_sim);
        }
        if let Some(doc) = &result.document {
            if let Some(title) = doc.get("title") {
                println!("    Title: {}", title);
            }
            if let Some(category) = doc.get("category") {
                println!("    Category: {}", category);
            }
        }
    }

    // Step 4: Demonstrate result operations
    println!("\n{}", "=".repeat(80));
    println!("\nStep 4: Result Operations\n");

    println!("--- Best Result ---");
    if let Some(best) = results.best_result() {
        println!("Doc ID: {}", best.doc_id);
        println!("Hybrid Score: {:.4}", best.hybrid_score);
    }

    println!("\n--- Filtering by Score (>= 0.7) ---");
    let original_len = results.len();
    results.filter_by_score(0.7);
    println!(
        "Filtered from {} to {} results",
        original_len,
        results.len()
    );

    println!("\n--- Limiting Results (top 2) ---");
    results.limit(2);
    println!("Limited to {} results", results.len());

    // Step 5: Create hybrid search engine
    println!("\n{}", "=".repeat(80));
    println!("\nStep 5: Creating Hybrid Search Engine\n");

    let engine = HybridSearchEngine::new(config_balanced)?;
    println!("Hybrid search engine created successfully!");
    println!("Configuration:");
    println!("  Keyword weight: {}", engine.config().keyword_weight);
    println!("  Vector weight: {}", engine.config().vector_weight);
    println!("  Normalization: {:?}", engine.config().normalization);

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("\n=== Summary ===");
    println!("\nHybrid Search combines two powerful search approaches:");
    println!("  1. Lexical/Keyword Search:");
    println!("     - Exact term matching with BM25 scoring");
    println!("     - Good for precise queries with known terms");
    println!("     - Example: \"rust programming language\"");
    println!("\n  2. Vector/Semantic Search:");
    println!("     - Similarity based on embeddings");
    println!("     - Good for conceptual queries");
    println!("     - Example: \"concurrent systems programming\"");
    println!("\nConfiguration Options:");
    println!("  - Adjustable weights for keyword vs. vector components");
    println!("  - Multiple score normalization strategies");
    println!("  - Minimum score thresholds for filtering");
    println!("  - Option to require matches from both search types");
    println!("\nResult Types:");
    println!("  - HybridSearchResult: Individual result with combined scoring");
    println!("  - HybridSearchResults: Collection with metadata and operations");
    println!("\nExample completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_search_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
