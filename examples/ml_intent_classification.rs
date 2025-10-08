//! ML-based Intent Classification Example
//!
//! This example demonstrates how to use the ML-based intent classifier
//! for query understanding in multilingual contexts (English and Japanese).

use anyhow::Result;
use sarissa::ml::MLContext;
use sarissa::ml::intent_classifier::IntentClassifier;
use sarissa::ml::query_expansion::{QueryExpansion, QueryExpansionConfig};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== ML-based Intent Classification Example ===\n");

    // Load training data from JSON file
    println!("Loading training data from resource/ml/intent_samples.json...");
    let training_data_path = "resource/ml/intent_samples.json";
    let samples = IntentClassifier::load_training_data(training_data_path)?;
    println!("Loaded {} training samples", samples.len());

    // Train the ML classifier
    println!("\nTraining ML intent classifier...");
    let classifier = IntentClassifier::new_ml(samples)?;
    println!("Training completed!");

    // Test queries in English
    println!("\n=== English Query Intent Classification ===");
    let en_queries = vec![
        "what is artificial intelligence",
        "how to create a website",
        "github official website",
        "download python installer",
        "buy macbook pro",
    ];

    for query in en_queries {
        let intent = classifier.predict(query)?;
        println!("Query: \"{}\" => Intent: {:?}", query, intent);
    }

    // Test queries in Japanese
    println!("\n=== Japanese Query Intent Classification ===");
    let ja_queries = vec![
        "人工知能とは何ですか",
        "Webサイトの作り方",
        "GitHub公式サイト",
        "Pythonをダウンロード",
        "MacBook Proを購入",
    ];

    for query in ja_queries {
        let intent = classifier.predict(query)?;
        println!("Query: \"{}\" => Intent: {:?}", query, intent);
    }

    // Demonstrate integration with QueryExpansion
    println!("\n=== Integration with Query Expansion ===");

    let config = QueryExpansionConfig {
        use_ml_classifier: true,
        ml_training_data_path: Some(training_data_path.to_string()),
        enable_synonyms: true,
        enable_semantic: false,
        ..Default::default()
    };

    let query_expander = QueryExpansion::new(config)?;

    let ml_context = MLContext {
        user_session: None,
        search_history: Vec::new(),
        user_preferences: HashMap::new(),
        timestamp: chrono::Utc::now(),
    };

    let test_queries = vec!["what is kubernetes", "GitHubのログイン", "download vscode"];

    for query in test_queries {
        let expanded = query_expander.expand_query(query, &ml_context)?;
        println!("\nQuery: \"{}\"", query);
        println!("  Intent: {:?}", expanded.intent);
        println!("  Confidence: {:.2}", expanded.confidence);
        println!("  Original terms: {:?}", expanded.original_terms);
        if !expanded.expanded_terms.is_empty() {
            println!("  Expanded terms: {:?}", expanded.expanded_terms);
        }
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
