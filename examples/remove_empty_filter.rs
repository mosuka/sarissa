//! Example demonstrating the RemoveEmptyFilter.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{Filter, RemoveEmptyFilter};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== RemoveEmptyFilter Examples ===\n");

    let filter = RemoveEmptyFilter::new();

    // Example 1: Basic empty token removal
    println!("1. Basic empty token removal:");
    let tokens = create_test_tokens(&["hello", "", "world", "", "test"]);
    demonstrate_filter(&filter, tokens, "Mixed text with empty tokens")?;

    // Example 2: Whitespace-only tokens
    println!("\n2. Whitespace-only tokens:");
    let tokens = create_test_tokens(&["text", "   ", "\t", "\n", "more", "  \t\n  "]);
    demonstrate_filter(
        &filter,
        tokens,
        "Whitespace is NOT removed (use StripFilter first)",
    )?;

    // Example 3: All empty tokens
    println!("\n3. All empty tokens:");
    let tokens = create_test_tokens(&["", "", "", ""]);
    demonstrate_filter(&filter, tokens, "Only empty strings")?;

    // Example 4: No empty tokens
    println!("\n4. No empty tokens:");
    let tokens = create_test_tokens(&["clean", "text", "without", "empty", "tokens"]);
    demonstrate_filter(&filter, tokens, "No empty tokens to remove")?;

    // Example 5: Single character tokens
    println!("\n5. Single character tokens:");
    let tokens = create_test_tokens(&["a", "", "b", "", "c", "d", ""]);
    demonstrate_filter(&filter, tokens, "Preserves single characters")?;

    // Example 6: After tokenization cleanup
    println!("\n6. Post-tokenization cleanup:");
    let tokens = create_test_tokens(&["word1", "", "word2", "", "", "word3", ""]);
    demonstrate_filter(&filter, tokens, "Typical post-tokenization scenario")?;

    // Example 7: Stopped tokens handling
    println!("\n7. Stopped tokens handling:");
    let tokens = vec![
        Token::new("normal", 0),
        Token::new("", 1),
        Token::new("text", 2).stop(), // Stopped but not empty
        Token::new("", 3).stop(),     // Stopped AND empty
        Token::new("end", 4),
    ];
    let token_stream = Box::new(tokens.into_iter());
    demonstrate_filter(&filter, token_stream, "Interaction with stopped tokens")?;

    // Example 8: Unicode empty handling
    println!("\n8. Unicode considerations:");
    let tokens = create_test_tokens(&["text", "", "unicode", "🦀", "", "emoji"]);
    demonstrate_filter(&filter, tokens, "Unicode content with empty strings")?;

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
        "Input:  {:?} (count: {})",
        input_tokens.iter().map(|t| &t.text).collect::<Vec<_>>(),
        input_count
    );

    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();
    let output_count = filtered_tokens.len();

    println!(
        "Output: {:?} (count: {})",
        filtered_tokens.iter().map(|t| &t.text).collect::<Vec<_>>(),
        output_count
    );

    let removed_count = input_count - output_count;
    if removed_count > 0 {
        println!(
            "Removed: {} empty token{}",
            removed_count,
            if removed_count == 1 { "" } else { "s" }
        );
    } else {
        println!("Removed: none");
    }

    Ok(())
}
