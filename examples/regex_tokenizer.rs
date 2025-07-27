//! Example demonstrating the usage of RegexTokenizer.

use sarissa::analysis::tokenizer::{RegexTokenizer, Tokenizer};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== RegexTokenizer Examples ===\n");

    // Example 1: Default regex tokenizer (matches word characters)
    println!("1. Default RegexTokenizer (\\w+):");
    let tokenizer = RegexTokenizer::new()?;
    demonstrate_tokenizer(&tokenizer, "Hello, world! This is a test-case.")?;

    // Example 2: Custom pattern - email addresses
    println!("\n2. Email pattern tokenizer:");
    let email_tokenizer =
        RegexTokenizer::with_pattern(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")?;
    demonstrate_tokenizer(
        &email_tokenizer,
        "Contact us at info@example.com or support@test.org for help.",
    )?;

    // Example 3: Numbers (including decimals)
    println!("\n3. Numbers tokenizer (including decimals):");
    let number_tokenizer = RegexTokenizer::with_pattern(r"\d+(?:\.\d+)?")?;
    demonstrate_tokenizer(
        &number_tokenizer,
        "The price is $123.45 and quantity is 10 items.",
    )?;

    // Example 4: Using gaps - extract non-word characters
    println!("\n4. Gaps tokenizer (extract punctuation and spaces):");
    let gaps_tokenizer = RegexTokenizer::with_gaps(r"\w+")?;
    demonstrate_tokenizer(&gaps_tokenizer, "Hello, world! How are you?")?;

    // Example 4b: Integer-only tokenizer for comparison
    println!("\n4b. Integer-only tokenizer (for comparison):");
    let int_tokenizer = RegexTokenizer::with_pattern(r"\d+")?;
    demonstrate_tokenizer(
        &int_tokenizer,
        "The price is $123.45 and quantity is 10 items.",
    )?;

    // Example 5: URLs
    println!("\n5. URL pattern tokenizer:");
    let url_tokenizer = RegexTokenizer::with_pattern(r"https?://[^\s]+")?;
    demonstrate_tokenizer(
        &url_tokenizer,
        "Visit https://example.com or http://test.org for more info.",
    )?;

    // Example 6: Code identifiers (camelCase, snake_case)
    println!("\n6. Code identifier tokenizer:");
    let identifier_tokenizer = RegexTokenizer::with_pattern(r"[a-zA-Z_][a-zA-Z0-9_]*")?;
    demonstrate_tokenizer(
        &identifier_tokenizer,
        "let user_name = getUserName(); const API_KEY = 'secret';",
    )?;

    // Example 7: Hashtags and mentions
    println!("\n7. Social media tokenizer (hashtags and mentions):");
    let social_tokenizer = RegexTokenizer::with_pattern(r"[#@]\w+")?;
    demonstrate_tokenizer(
        &social_tokenizer,
        "Check out #rustlang and follow @rustlang for updates!",
    )?;

    demonstrate_tokenizer(&tokenizer, "Events: 2023-12-25, 2024-01-01, 2024-02-14")?;

    Ok(())
}

fn demonstrate_tokenizer(tokenizer: &dyn Tokenizer, text: &str) -> Result<()> {
    println!("Input: \"{}\"", text);

    let tokens: Vec<_> = tokenizer.tokenize(text)?.collect();

    if tokens.is_empty() {
        println!("No tokens found.");
    } else {
        println!("Tokens ({}):", tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            println!(
                "  {}: \"{}\" (pos: {}, offset: {}..{})",
                i, token.text, token.position, token.start_offset, token.end_offset
            );
        }
    }

    Ok(())
}
