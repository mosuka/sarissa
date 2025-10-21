//! ML-based Intent Classification Example
//!
//! This example demonstrates how to use the ML-based intent classifier
//! for query understanding in multilingual contexts (English and Japanese).

use std::sync::Arc;

use anyhow::Result;

use sage::analysis::analyzer::language::english::EnglishAnalyzer;
use sage::analysis::analyzer::language::japanese::JapaneseAnalyzer;
use sage::ml::intent_classifier::core;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== ML-based Intent Classification Example ===\n");

    // Create English classifier
    println!("Creating English intent classifier...");
    let en_training_data_path = "resources/ml/intent_samples_en.json";
    let en_analyzer = Arc::new(EnglishAnalyzer::new()?);
    let en_samples = core::load_training_data(en_training_data_path)?;
    println!("Loaded {} English training samples", en_samples.len());
    let en_classifier = core::new_ml_based(en_samples, en_analyzer.clone())?;
    println!("English classifier created!");

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
        let intent = en_classifier.predict(query)?;
        println!("Query: \"{}\" => Intent: {:?}", query, intent);
    }

    // Create Japanese classifier
    println!("\nCreating Japanese intent classifier...");
    let ja_training_data_path = "resources/ml/intent_samples_ja.json";
    let ja_analyzer = Arc::new(JapaneseAnalyzer::new()?);
    let ja_samples = core::load_training_data(ja_training_data_path)?;
    println!("Loaded {} Japanese training samples", ja_samples.len());
    let ja_classifier = core::new_ml_based(ja_samples, ja_analyzer.clone())?;
    println!("Japanese classifier created!");

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
        let intent = ja_classifier.predict(query)?;
        println!("Query: \"{}\" => Intent: {:?}", query, intent);
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
