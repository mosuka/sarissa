//! Keyword-based Intent Classification Example
//!
//! This example demonstrates how to use the keyword-based intent classifier
//! for query understanding in multilingual contexts (English and Japanese).

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;

use sage::analysis::analyzer::language::{EnglishAnalyzer, JapaneseAnalyzer};
use sage::ml::intent_classifier;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Keyword-based Intent Classification Example ===\n");

    // Define English keyword sets
    println!("Creating keyword-based classifier with English keywords...");
    let mut informational_en = HashSet::new();
    informational_en.extend(vec![
        "what".to_string(),
        "how".to_string(),
        "why".to_string(),
        "when".to_string(),
        "where".to_string(),
        "who".to_string(),
        "definition".to_string(),
        "explain".to_string(),
    ]);

    let mut navigational_en = HashSet::new();
    navigational_en.extend(vec![
        "homepage".to_string(),
        "website".to_string(),
        "site".to_string(),
        "login".to_string(),
        "official".to_string(),
    ]);

    let mut transactional_en = HashSet::new();
    transactional_en.extend(vec![
        "buy".to_string(),
        "purchase".to_string(),
        "order".to_string(),
        "download".to_string(),
        "install".to_string(),
        "get".to_string(),
        "free".to_string(),
        "price".to_string(),
    ]);

    let analyzer_en = Arc::new(EnglishAnalyzer::new()?);
    let classifier_en = intent_classifier::new_keyword_based(
        informational_en,
        navigational_en,
        transactional_en,
        analyzer_en,
    );
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
        let intent = classifier_en.predict(query)?;
        println!("Query: \"{}\" => Intent: {:?}", query, intent);
    }

    // Define Japanese keyword sets
    println!("\n\nCreating keyword-based classifier with Japanese keywords...");
    let mut informational_ja = HashSet::new();
    informational_ja.extend(vec![
        "何".to_string(),
        "なに".to_string(),
        "どう".to_string(),
        "なぜ".to_string(),
        "いつ".to_string(),
        "どこ".to_string(),
        "誰".to_string(),
        "とは".to_string(),
        "説明".to_string(),
    ]);

    let mut navigational_ja = HashSet::new();
    navigational_ja.extend(vec![
        "ホームページ".to_string(),
        "サイト".to_string(),
        "ログイン".to_string(),
        "公式".to_string(),
        "ウェブサイト".to_string(),
    ]);

    let mut transactional_ja = HashSet::new();
    transactional_ja.extend(vec![
        "購入".to_string(),
        "買う".to_string(),
        "注文".to_string(),
        "ダウンロード".to_string(),
        "インストール".to_string(),
        "取得".to_string(),
        "無料".to_string(),
        "価格".to_string(),
    ]);

    let analyzer_ja = Arc::new(JapaneseAnalyzer::new()?);
    let classifier_ja = intent_classifier::new_keyword_based(
        informational_ja,
        navigational_ja,
        transactional_ja,
        analyzer_ja,
    );
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
        let intent = classifier_ja.predict(query)?;
        println!("Query: \"{}\" => Intent: {:?}", query, intent);
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
