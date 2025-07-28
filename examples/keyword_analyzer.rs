//! Example demonstrating the KeywordAnalyzer
//!
//! This example shows how to use the keyword analyzer which treats
//! the entire input as a single token. This is useful for exact matching
//! of IDs, codes, or other fields that should not be tokenized.

use sarissa::analysis::{Analyzer, KeywordAnalyzer};

fn main() -> sarissa::error::Result<()> {
    println!("=== KeywordAnalyzer Example ===\n");

    // Create a keyword analyzer
    let analyzer = KeywordAnalyzer::new();

    // Example 1: Basic usage
    println!("1. Basic usage - entire input as single token:");
    let texts = vec![
        "product-id-12345",
        "USER EMAIL: john.doe@example.com",
        "Multiple Words Are Kept Together",
        "   Spaces   Are   Preserved   ",
        "Special!@#$%^&*()Characters_Included",
    ];

    for text in &texts {
        let tokens: Vec<_> = analyzer.analyze(text)?.collect();
        println!("   Input: \"{}\"", text);
        println!("   Token count: {}", tokens.len());
        if let Some(token) = tokens.first() {
            println!("   Token text: \"{}\"", token.text);
            println!(
                "   Position: {}, Offset: {}-{}",
                token.position, token.start_offset, token.end_offset
            );
        }
        println!();
    }

    // Example 2: Use case - product SKUs
    println!("2. Use case - Product SKUs:");
    let skus = vec!["ABC-123-XYZ", "DEF 456 UVW", "ghi_789_rst", "JKL/012/MNO"];

    println!("   Processing product SKUs that must remain intact:");
    for sku in &skus {
        let tokens: Vec<_> = analyzer.analyze(sku)?.collect();
        if let Some(token) = tokens.first() {
            println!("     SKU: \"{}\" -> Token: \"{}\"", sku, token.text);
        }
    }
    println!();

    // Example 3: Use case - email addresses
    println!("3. Use case - Email addresses:");
    let emails = vec![
        "simple@example.com",
        "user.name+tag@example.co.uk",
        "test_email_123@sub.domain.com",
    ];

    println!("   Processing email addresses:");
    for email in &emails {
        let tokens: Vec<_> = analyzer.analyze(email)?.collect();
        if let Some(token) = tokens.first() {
            println!("     Email: \"{}\" -> Token: \"{}\"", email, token.text);
        }
    }
    println!();

    // Example 4: Comparing with other analyzers
    println!("4. Comparison with tokenizing analyzers:");
    let comparison_text = "This-Would-Be-Split-By-Other-Analyzers";

    let tokens: Vec<_> = analyzer.analyze(comparison_text)?.collect();
    println!("   Input: \"{}\"", comparison_text);
    println!("   KeywordAnalyzer result: {} token(s)", tokens.len());
    if let Some(token) = tokens.first() {
        println!("     Token: \"{}\"", token.text);
    }
    println!("   Note: Other analyzers would split this into multiple tokens");
    println!();

    // Example 5: Empty and whitespace handling
    println!("5. Edge cases:");
    let edge_cases = vec![
        ("Empty string", ""),
        ("Just spaces", "   "),
        ("Newlines", "\n\n"),
        ("Mixed whitespace", "\t \n"),
    ];

    for (desc, text) in &edge_cases {
        let tokens: Vec<_> = analyzer.analyze(text)?.collect();
        println!("   {}: \"{}\"", desc, text.escape_debug());
        println!("     Token count: {}", tokens.len());
        if let Some(token) = tokens.first() {
            println!("     Token: \"{}\"", token.text.escape_debug());
        }
        println!();
    }

    Ok(())
}
