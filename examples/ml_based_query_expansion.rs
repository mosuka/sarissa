//! ML-based Query Expansion Example
//!
//! This example demonstrates how to use the ML-based query expansion system
//! with intent classification for multilingual contexts (English and Japanese).

use anyhow::Result;
use sarissa::analysis::analyzer::language::{EnglishAnalyzer, JapaneseAnalyzer};
use sarissa::ml::MLContext;
use sarissa::ml::query_expansion::{QueryExpansion, QueryExpansionConfig};
use sarissa::query::Query;
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== ML-based Query Expansion Example ===\n");

    // Create English query expander
    println!("Creating English query expander...");
    let en_config = QueryExpansionConfig {
        use_ml_classifier: true,
        ml_training_data_path: Some("resource/ml/intent_samples_en.json".to_string()),
        ml_training_language: Some("en".to_string()),
        enable_synonyms: true,
        enable_semantic: false,
        ..Default::default()
    };
    let en_analyzer = Arc::new(EnglishAnalyzer::new()?);
    let en_query_expander = QueryExpansion::new(en_config, en_analyzer)?;
    println!("English query expander created!");

    // Create Japanese query expander
    println!("\nCreating Japanese query expander...");
    let ja_config = QueryExpansionConfig {
        use_ml_classifier: true,
        ml_training_data_path: Some("resource/ml/intent_samples_ja.json".to_string()),
        ml_training_language: Some("ja".to_string()),
        enable_synonyms: true,
        enable_semantic: false,
        ..Default::default()
    };
    let ja_analyzer = Arc::new(JapaneseAnalyzer::new()?);
    let ja_query_expander = QueryExpansion::new(ja_config, ja_analyzer)?;
    println!("Japanese query expander created!");

    let ml_context = MLContext {
        user_session: None,
        search_history: Vec::new(),
        user_preferences: HashMap::new(),
        timestamp: chrono::Utc::now(),
    };

    // Test English queries (using terms that have synonyms in the default dictionary)
    println!("\n=== English Query Expansion ===");
    let en_test_queries = vec![
        "what is ml",                   // "ml" has synonyms: "machine learning", etc.
        "learn ai algorithms",          // "ai" and "algorithm" have synonyms
        "python programming tutorial",  // "programming" has synonyms: "coding", "development"
        "data science introduction",    // "data" has synonyms: "dataset", "data science"
        "download vscode",
    ];

    for query in en_test_queries {
        let expanded = en_query_expander.expand_query(query, "content", &ml_context)?;
        println!("\nQuery: \"{}\"", query);
        println!("  Intent: {:?}", expanded.intent);
        println!("  Confidence: {:.2}", expanded.confidence);
        println!("  Original query: {}", expanded.original_query.description());
        if !expanded.expanded_queries.is_empty() {
            println!("  Expanded queries:");
            for exp in &expanded.expanded_queries {
                println!(
                    "    - {} (type: {:?}, confidence: {:.2}, boost: {:.2})",
                    exp.query.description(),
                    exp.expansion_type,
                    exp.confidence,
                    exp.query.boost()
                );
            }
        }
        println!("  Combined boolean query: {}", expanded.to_boolean_query().description());
    }

    // Test Japanese queries (using terms that have synonyms)
    println!("\n=== Japanese Query Expansion ===");
    let ja_test_queries = vec![
        "機械学習とは",             // "機械" and "学習" have synonyms
        "人工知能の基礎",           // "人工" and "知能" have synonyms
        "プログラミングの学習方法", // "プログラミング" and "学習" have synonyms
        "ノートパソコンを購入",     // "購入" has synonyms: "買う", "注文"
        "Dockerをインストール",
    ];

    for query in ja_test_queries {
        let expanded = ja_query_expander.expand_query(query, "content", &ml_context)?;
        println!("\nQuery: \"{}\"", query);
        println!("  Intent: {:?}", expanded.intent);
        println!("  Confidence: {:.2}", expanded.confidence);
        println!("  Original query: {}", expanded.original_query.description());
        if !expanded.expanded_queries.is_empty() {
            println!("  Expanded queries:");
            for exp in &expanded.expanded_queries {
                println!(
                    "    - {} (type: {:?}, confidence: {:.2}, boost: {:.2})",
                    exp.query.description(),
                    exp.expansion_type,
                    exp.confidence,
                    exp.query.boost()
                );
            }
        }
        println!("  Combined boolean query: {}", expanded.to_boolean_query().description());
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
