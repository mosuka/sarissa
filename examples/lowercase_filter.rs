//! Example demonstrating the LowercaseFilter.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{Filter, LowercaseFilter};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== LowercaseFilter Examples ===\n");

    let filter = LowercaseFilter::new();

    // Example 1: Basic mixed case text
    println!("1. Basic mixed case conversion:");
    let tokens = create_test_tokens(&["Hello", "WORLD", "CamelCase"]);
    demonstrate_filter(&filter, tokens, "Mixed case words")?;

    // Example 2: All uppercase text
    println!("\n2. All uppercase text:");
    let tokens = create_test_tokens(&["PROGRAMMING", "LANGUAGE", "RUST"]);
    demonstrate_filter(&filter, tokens, "Uppercase technical terms")?;

    // Example 3: Mixed with symbols and numbers
    println!("\n3. Text with symbols and numbers:");
    let tokens = create_test_tokens(&["API_KEY", "Version2.0", "HTTP_200"]);
    demonstrate_filter(&filter, tokens, "Technical identifiers")?;

    // Example 4: Unicode characters
    println!("\n4. Unicode characters:");
    let tokens = create_test_tokens(&["CAFÉ", "NAÏVE", "RÉSUMÉ"]);
    demonstrate_filter(&filter, tokens, "Unicode accented characters")?;

    // Example 5: Already lowercase text
    println!("\n5. Already lowercase text:");
    let tokens = create_test_tokens(&["already", "lowercase", "text"]);
    demonstrate_filter(&filter, tokens, "No changes expected")?;

    // Example 6: Real-world sentence
    println!("\n6. Real-world sentence:");
    let tokens = create_test_tokens(&[
        "The", "Quick", "Brown", "FOX", "jumps", "OVER", "the", "LAZY", "dog",
    ]);
    demonstrate_filter(&filter, tokens, "Mixed case sentence")?;

    // Example 7: Programming code
    println!("\n7. Programming identifiers:");
    let tokens = create_test_tokens(&["getUserName", "API_ENDPOINT", "MAX_SIZE", "isValid"]);
    demonstrate_filter(&filter, tokens, "Code identifiers")?;

    // Example 8: Empty and special cases
    println!("\n8. Empty and special cases:");
    let tokens = create_test_tokens(&["", "A", "123", "!@#"]);
    demonstrate_filter(&filter, tokens, "Edge cases")?;

    println!("\n=== Filter Properties ===\n");
    println!("Filter name: {}", filter.name());

    println!("\n=== Use Cases ===\n");
    println!("LowercaseFilter is ideal for:");
    println!("  • Case-insensitive search and matching");
    println!("  • Text normalization before analysis");
    println!("  • Reducing vocabulary size in NLP tasks");
    println!("  • Consistent data preprocessing");
    println!("  • Search engine indexing");
    println!("  • User input standardization");

    println!("\n=== Performance Notes ===\n");
    println!("• Uses SIMD-optimized ASCII conversion when possible");
    println!("• Handles Unicode characters correctly");
    println!("• Skips processing for stopped tokens");
    println!("• Minimal memory allocation");

    Ok(())
}

fn create_test_tokens(texts: &[&str]) -> TokenStream {
    let tokens: Vec<Token> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| Token::new(*text, i))
        .collect();
    Box::new(tokens.into_iter())
}

fn demonstrate_filter(filter: &dyn Filter, tokens: TokenStream, description: &str) -> Result<()> {
    println!("Description: {}", description);

    let input_tokens: Vec<Token> = tokens.collect();
    let input_count = input_tokens.len();
    println!(
        "Input:  {:?}",
        input_tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );

    let input_clone = input_tokens.clone();
    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();

    println!(
        "Output: {:?}",
        filtered_tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );

    // Show transformations
    println!("Changes:");
    let mut changes = 0;
    for (i, (input, output)) in input_clone.iter().zip(filtered_tokens.iter()).enumerate() {
        if input.text != output.text {
            println!("  {}: '{}' → '{}'", i, input.text, output.text);
            changes += 1;
        }
    }
    if changes == 0 {
        println!("  (no changes)");
    }

    println!("Count: {} → {}", input_count, filtered_tokens.len());

    Ok(())
}
