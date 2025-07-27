//! Example demonstrating the RemoveEmptyFilter.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{Filter, RemoveEmptyFilter};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== RemoveEmptyFilter Examples ===\n");

    let filter = RemoveEmptyFilter::new();

    // Example 1: Basic empty token removal
    println!("1. Basic empty token removal:");
    let tokens = create_test_tokens(&[
        "hello", "", "world", "", "test"
    ]);
    demonstrate_filter(&filter, tokens, "Mixed text with empty tokens")?;

    // Example 2: Whitespace-only tokens
    println!("\n2. Whitespace-only tokens:");
    let tokens = create_test_tokens(&[
        "text", "   ", "\t", "\n", "more", "  \t\n  "
    ]);
    demonstrate_filter(&filter, tokens, "Whitespace is NOT removed (use StripFilter first)")?;

    // Example 3: All empty tokens
    println!("\n3. All empty tokens:");
    let tokens = create_test_tokens(&[
        "", "", "", ""
    ]);
    demonstrate_filter(&filter, tokens, "Only empty strings")?;

    // Example 4: No empty tokens
    println!("\n4. No empty tokens:");
    let tokens = create_test_tokens(&[
        "clean", "text", "without", "empty", "tokens"
    ]);
    demonstrate_filter(&filter, tokens, "No empty tokens to remove")?;

    // Example 5: Single character tokens
    println!("\n5. Single character tokens:");
    let tokens = create_test_tokens(&[
        "a", "", "b", "", "c", "d", ""
    ]);
    demonstrate_filter(&filter, tokens, "Preserves single characters")?;

    // Example 6: After tokenization cleanup
    println!("\n6. Post-tokenization cleanup:");
    let tokens = create_test_tokens(&[
        "word1", "", "word2", "", "", "word3", ""
    ]);
    demonstrate_filter(&filter, tokens, "Typical post-tokenization scenario")?;

    // Example 7: Stopped tokens handling
    println!("\n7. Stopped tokens handling:");
    let tokens = vec![
        Token::new("normal", 0),
        Token::new("", 1),
        Token::new("text", 2).stop(),  // Stopped but not empty
        Token::new("", 3).stop(),      // Stopped AND empty
        Token::new("end", 4),
    ];
    let token_stream = Box::new(tokens.into_iter());
    demonstrate_filter(&filter, token_stream, "Interaction with stopped tokens")?;

    // Example 8: Unicode empty handling
    println!("\n8. Unicode considerations:");
    let tokens = create_test_tokens(&[
        "text", "", "unicode", "🦀", "", "emoji"
    ]);
    demonstrate_filter(&filter, tokens, "Unicode content with empty strings")?;

    println!("\n=== Filter Properties ===\n");
    println!("Filter name: {}", filter.name());
    
    println!("\n=== Behavior Details ===\n");
    println!("Token handling:");
    println!("  • Removes tokens with empty text (\"\")");
    println!("  • Preserves stopped tokens (not removed)");
    println!("  • Does NOT remove whitespace-only tokens");
    println!("  • Maintains relative positions of remaining tokens");
    println!("  • Works with any character encoding");

    println!("\n=== What is NOT removed ===\n");
    println!("The filter does NOT remove:");
    println!("  • Whitespace-only tokens (\"   \", \"\\t\", \"\\n\")");
    println!("  • Single character tokens (\"a\", \"1\", \"!\")");
    println!("  • Stopped tokens (even if empty)");
    println!("  • Unicode whitespace");
    println!("  • Zero-width characters");

    println!("\n=== Use Cases ===\n");
    println!("Tokenization cleanup:");
    println!("  • Remove artifacts from regex tokenization");
    println!("  • Clean up split() operations");
    println!("  • Handle missing data in CSV parsing");
    
    println!("\nData preprocessing:");
    println!("  • Database field cleaning");
    println!("  • API response normalization");
    println!("  • File parsing error handling");
    
    println!("\nText analysis:");
    println!("  • Reduce token count for efficiency");
    println!("  • Prevent empty tokens in downstream filters");
    println!("  • Clean input for machine learning");

    println!("\n=== Pipeline Considerations ===\n");
    println!("Common pipeline order:");
    println!("  1. Tokenization");
    println!("  2. → StripFilter (remove whitespace)");
    println!("  3. → RemoveEmptyFilter (remove empty results)");
    println!("  4. → Other filters...");
    
    println!("\nAlternative approach:");
    println!("  • Use StripFilter alone (marks empty as stopped)");
    println!("  • Use RemoveEmptyFilter for hard removal");

    println!("\n=== Performance Notes ===\n");
    println!("• O(n) linear filtering");
    println!("• Minimal memory allocation");
    println!("• Early termination for stopped tokens");
    println!("• Efficient string length checking");

    Ok(())
}

fn create_test_tokens(texts: &[&str]) -> TokenStream {
    let tokens: Vec<Token> = texts.iter()
        .enumerate()
        .map(|(i, text)| Token::new(*text, i))
        .collect();
    Box::new(tokens.into_iter())
}

fn demonstrate_filter(
    filter: &dyn Filter,
    tokens: TokenStream,
    description: &str,
) -> Result<()> {
    println!("Description: {}", description);
    
    let input_tokens: Vec<Token> = tokens.collect();
    let input_count = input_tokens.len();
    println!("Input:  {:?} (count: {})", 
        input_tokens.iter().map(|t| &t.text).collect::<Vec<_>>(), input_count);
    
    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();
    let output_count = filtered_tokens.len();
    
    println!("Output: {:?} (count: {})", 
        filtered_tokens.iter().map(|t| &t.text).collect::<Vec<_>>(), output_count);
    
    let removed_count = input_count - output_count;
    if removed_count > 0 {
        println!("Removed: {} empty token{}", removed_count, 
            if removed_count == 1 { "" } else { "s" });
    } else {
        println!("Removed: none");
    }
    
    Ok(())
}