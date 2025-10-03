//! Example demonstrating the usage of WhitespaceTokenizer.

use sarissa::analysis::tokenizer::{Tokenizer, WhitespaceTokenizer};
use sarissa::error::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== WhitespaceTokenizer Examples ===\n");

    let tokenizer = WhitespaceTokenizer::new();

    // Example 1: Basic sentence
    println!("1. Basic sentence:");
    demonstrate_tokenizer(&tokenizer, "Hello world this is a test")?;

    // Example 2: Multiple spaces
    println!("\n2. Multiple spaces:");
    demonstrate_tokenizer(&tokenizer, "Hello    world   with     multiple    spaces")?;

    // Example 3: Mixed whitespace (spaces, tabs, newlines)
    println!("\n3. Mixed whitespace:");
    demonstrate_tokenizer(&tokenizer, "Hello\tworld\nthis\r\nis\ta\n\ntest")?;

    // Example 4: Leading and trailing whitespace
    println!("\n4. Leading and trailing whitespace:");
    demonstrate_tokenizer(&tokenizer, "   Hello world   ")?;

    // Example 5: Punctuation preserved
    println!("\n5. Punctuation preserved:");
    demonstrate_tokenizer(&tokenizer, "Hello, world! How are you? Fine, thanks.")?;

    // Example 6: Numbers and special characters
    println!("\n6. Numbers and special characters:");
    demonstrate_tokenizer(
        &tokenizer,
        "Price: $123.45 Quantity: 10 Email: user@example.com",
    )?;

    // Example 7: Code-like text
    println!("\n7. Code-like text:");
    demonstrate_tokenizer(&tokenizer, "let x = 10; if x > 5 { println!(\"Hello\"); }")?;

    // Example 8: Empty and whitespace-only input
    println!("\n8. Empty input:");
    demonstrate_tokenizer(&tokenizer, "")?;

    println!("\n9. Whitespace-only input:");
    demonstrate_tokenizer(&tokenizer, "   \t\n\r   ")?;

    // Example 9: Unicode text
    println!("\n10. Unicode text:");
    demonstrate_tokenizer(&tokenizer, "café naïve résumé 日本語 русский")?;

    // Example 10: Long text with mixed content
    println!("\n11. Long mixed content:");
    demonstrate_tokenizer(
        &tokenizer,
        "The quick brown fox jumps over the lazy dog. 123 Main St. user@domain.com +1-555-0123",
    )?;

    println!("\n=== Performance Demonstration ===\n");

    // Performance comparison for ASCII vs Unicode text
    performance_demo(&tokenizer)?;

    Ok(())
}

fn demonstrate_tokenizer(tokenizer: &dyn Tokenizer, text: &str) -> Result<()> {
    println!("Input: {text:?}");

    let tokens: Vec<_> = tokenizer.tokenize(text)?.collect();

    if tokens.is_empty() {
        println!("No tokens found.");
    } else {
        println!("Tokens ({}):", tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            println!(
                "  {}: {:?} (pos: {}, offset: {}..{})",
                i, token.text, token.position, token.start_offset, token.end_offset
            );
        }
    }

    Ok(())
}

fn performance_demo(tokenizer: &WhitespaceTokenizer) -> Result<()> {
    println!("12. Performance comparison:");

    // Short ASCII text (fallback path)
    let short_ascii = "Hello world test";
    let start = Instant::now();
    let tokens1: Vec<_> = tokenizer.tokenize(short_ascii)?.collect();
    let duration1 = start.elapsed();
    println!(
        "Short ASCII text ({} chars): {} tokens in {:?}",
        short_ascii.len(),
        tokens1.len(),
        duration1
    );

    // Long ASCII text (SIMD path)
    let long_ascii = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    let start = Instant::now();
    let tokens2: Vec<_> = tokenizer.tokenize(&long_ascii)?.collect();
    let duration2 = start.elapsed();
    println!(
        "Long ASCII text ({} chars): {} tokens in {:?} (SIMD optimized)",
        long_ascii.len(),
        tokens2.len(),
        duration2
    );

    // Unicode text (fallback path)
    let unicode_text = "café naïve résumé 日本語 русский العربية ".repeat(5);
    let start = Instant::now();
    let tokens3: Vec<_> = tokenizer.tokenize(&unicode_text)?.collect();
    let duration3 = start.elapsed();
    println!(
        "Unicode text ({} chars): {} tokens in {:?} (fallback)",
        unicode_text.len(),
        tokens3.len(),
        duration3
    );

    Ok(())
}
