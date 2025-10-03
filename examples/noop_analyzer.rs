//! Example demonstrating the NoOpAnalyzer
//!
//! This example shows how to use the no-op analyzer which doesn't perform
//! any analysis and returns an empty token stream. This is useful for
//! stored-only fields or testing scenarios.

use sarissa::analysis::{Analyzer, NoOpAnalyzer};

fn main() -> sarissa::error::Result<()> {
    println!("=== NoOpAnalyzer Example ===\n");

    // Create a no-op analyzer
    let analyzer = NoOpAnalyzer::new();

    println!("The NoOpAnalyzer is a special analyzer that performs no analysis.");
    println!("It always returns an empty token stream regardless of input.\n");

    // Example 1: Various inputs all produce empty results
    println!("1. Testing various inputs:");
    let test_inputs = vec![
        "Simple text",
        "Complex text with numbers 123 and symbols!@#",
        "UPPERCASE lowercase MixedCase",
        "Multiple\nLines\nOf\nText",
        "   Spaces   everywhere   ",
        "", // Empty string
    ];

    for (i, input) in test_inputs.iter().enumerate() {
        let tokens: Vec<_> = analyzer.analyze(input)?.collect();
        println!("   Test {}: \"{}\"", i + 1, input.escape_debug());
        println!("   Token count: {}", tokens.len());
        println!("   Has tokens: {}", !tokens.is_empty());
        println!();
    }

    // Example 2: Use cases for NoOpAnalyzer
    println!("2. Common use cases for NoOpAnalyzer:");
    println!("   a) Stored-only fields that don't need to be searchable");
    println!("   b) Fields used only for sorting or faceting");
    println!("   c) Binary data fields");
    println!("   d) Testing and benchmarking scenarios");
    println!("   e) Placeholder analyzer during development\n");

    // Example 3: Comparing with other analyzers
    println!("3. Comparison with other analyzers:");
    let comparison_text = "The quick brown fox";

    let noop_tokens: Vec<_> = analyzer.analyze(comparison_text)?.collect();

    println!("   Input: \"{comparison_text}\"");
    println!("   NoOpAnalyzer tokens: {}", noop_tokens.len());
    println!("   Other analyzers would produce multiple tokens");
    println!("   NoOpAnalyzer always produces: 0 tokens\n");

    // Example 4: Performance characteristics
    println!("4. Performance characteristics:");
    println!("   - Zero memory allocation for tokens");
    println!("   - Constant O(1) time complexity");
    println!("   - No CPU cycles spent on text processing");
    println!("   - Ideal for fields that don't require searching\n");

    // Verify it truly produces no tokens
    let large_text = "Lorem ipsum ".repeat(1000);
    let tokens: Vec<_> = analyzer.analyze(&large_text)?.collect();
    println!("5. Verification with large input:");
    println!("   Input size: {} characters", large_text.len());
    println!("   Tokens produced: {}", tokens.len());
    println!("   Memory used for tokens: 0 bytes");

    Ok(())
}
