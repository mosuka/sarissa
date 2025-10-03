//! Example demonstrating the usage of WholeTokenizer.

use sarissa::analysis::tokenizer::{Tokenizer, WholeTokenizer};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== WholeTokenizer Examples ===\n");

    let tokenizer = WholeTokenizer::new();

    // Example 1: Basic usage
    println!("1. Basic text:");
    demonstrate_tokenizer(&tokenizer, "Hello, world!")?;

    // Example 2: Multi-word text
    println!("\n2. Multi-word text:");
    demonstrate_tokenizer(
        &tokenizer,
        "This is a complete sentence with multiple words.",
    )?;

    // Example 3: ID field
    println!("\n3. ID field:");
    demonstrate_tokenizer(&tokenizer, "USER_ID_12345_ABC")?;

    // Example 4: URL
    println!("\n4. URL:");
    demonstrate_tokenizer(
        &tokenizer,
        "https://example.com/path/to/resource?param=value",
    )?;

    // Example 5: File path
    println!("\n5. File path:");
    demonstrate_tokenizer(&tokenizer, "/home/user/documents/important_file.txt")?;

    // Example 6: Code snippet
    println!("\n6. Code snippet:");
    demonstrate_tokenizer(&tokenizer, "fn main() { println!(\"Hello, world!\"); }")?;

    // Example 7: JSON-like data
    println!("\n7. JSON-like data:");
    demonstrate_tokenizer(
        &tokenizer,
        r#"{"name": "John", "age": 30, "city": "New York"}"#,
    )?;

    // Example 8: Empty string
    println!("\n8. Empty string:");
    demonstrate_tokenizer(&tokenizer, "")?;

    // Example 9: Whitespace only
    println!("\n9. Whitespace only:");
    demonstrate_tokenizer(&tokenizer, "   \t\n  ")?;

    // Example 10: Special characters and Unicode
    println!("\n10. Special characters and Unicode:");
    demonstrate_tokenizer(&tokenizer, "café, naïve, résumé, 日本語, emoji: 🦀🔥")?;

    Ok(())
}

fn demonstrate_tokenizer(tokenizer: &dyn Tokenizer, text: &str) -> Result<()> {
    println!("Input: \"{text}\"");

    let tokens: Vec<_> = tokenizer.tokenize(text)?.collect();

    if tokens.is_empty() {
        println!("No tokens found.");
    } else {
        println!("Tokens ({}):", tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            println!(
                "  {}: \"{}\" (pos: {}, offset: {}..{}, len: {})",
                i,
                token.text,
                token.position,
                token.start_offset,
                token.end_offset,
                token.text.len()
            );
        }
    }

    Ok(())
}
