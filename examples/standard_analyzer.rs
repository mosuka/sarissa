//! StandardAnalyzer example - demonstrates text analysis and tokenization.

use sarissa::analysis::{Analyzer, StandardAnalyzer};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== StandardAnalyzer Example - Text Analysis and Tokenization ===\n");

    // Create a StandardAnalyzer
    let analyzer = StandardAnalyzer::new()?;
    println!("Created StandardAnalyzer with default configuration");

    println!("\n=== Text Analysis Examples ===\n");

    // Example 1: Basic text analysis
    println!("1. Basic text analysis:");
    let text = "The Great Gatsby is a masterpiece!";
    let tokens: Vec<_> = analyzer.analyze(text)?.collect();
    println!("   Original: \"{}\"", text);
    println!(
        "   Tokens: {:?}",
        tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );
    println!("   Token count: {}", tokens.len());

    // Example 2: Complex text with punctuation
    println!("\n2. Complex text with punctuation:");
    let complex_text = "Hello, World! This is a test... with various punctuation marks: (parentheses), [brackets], and 'quotes'.";
    let tokens: Vec<_> = analyzer.analyze(complex_text)?.collect();
    println!("   Original: \"{}\"", complex_text);
    println!(
        "   Tokens: {:?}",
        tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );
    println!("   Token count: {}", tokens.len());

    // Example 3: Numbers and mixed content
    println!("\n3. Numbers and mixed content:");
    let mixed_text = "The year 2024 marks the 100th anniversary of this event in New York City.";
    let tokens: Vec<_> = analyzer.analyze(mixed_text)?.collect();
    println!("   Original: \"{}\"", mixed_text);
    println!(
        "   Tokens: {:?}",
        tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );
    println!("   Token count: {}", tokens.len());

    // Example 4: Case normalization
    println!("\n4. Case normalization:");
    let case_text = "UPPERCASE lowercase MiXeD CaSe";
    let tokens: Vec<_> = analyzer.analyze(case_text)?.collect();
    println!("   Original: \"{}\"", case_text);
    println!(
        "   Tokens: {:?}",
        tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );
    println!("   Note: All tokens are normalized to lowercase");

    // Example 5: Empty and whitespace handling
    println!("\n5. Empty and whitespace handling:");
    let whitespace_text = "   \t\n  spaced   out    text   \t\n  ";
    let tokens: Vec<_> = analyzer.analyze(whitespace_text)?.collect();
    println!("   Original: {:?}", whitespace_text);
    println!(
        "   Tokens: {:?}",
        tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );
    println!("   Note: Extra whitespace is normalized");

    // Example 6: Token positions and offsets
    println!("\n6. Token positions and offsets:");
    let position_text = "search engine optimization";
    let tokens: Vec<_> = analyzer.analyze(position_text)?.collect();
    println!("   Original: \"{}\"", position_text);
    println!("   Detailed token information:");
    for (i, token) in tokens.iter().enumerate() {
        println!(
            "     Token {}: '{}' (position: {}, start: {}, end: {})",
            i + 1,
            token.text,
            token.position,
            token.start_offset,
            token.end_offset
        );
    }

    // Example 7: Special characters and Unicode
    println!("\n7. Special characters and Unicode:");
    let unicode_text = "café résumé naïve Москва 東京 🚀";
    let tokens: Vec<_> = analyzer.analyze(unicode_text)?.collect();
    println!("   Original: \"{}\"", unicode_text);
    println!(
        "   Tokens: {:?}",
        tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );
    println!("   Note: Unicode characters are preserved");

    println!("\nStandardAnalyzer example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_analyzer_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
