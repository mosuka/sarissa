//! Example demonstrating the LimitFilter.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{Filter, LimitFilter};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== LimitFilter Examples ===\n");

    // Example 1: Basic token limiting
    println!("1. Basic token limiting:");
    let limit_filter = LimitFilter::new(3);
    let tokens = create_test_tokens(&[
        "first", "second", "third", "fourth", "fifth", "sixth"
    ]);
    demonstrate_filter(&limit_filter, tokens, "Limit to first 3 tokens")?;

    // Example 2: Single token limit
    println!("\n2. Single token limit:");
    let single_filter = LimitFilter::new(1);
    let tokens = create_test_tokens(&[
        "only", "first", "token", "matters"
    ]);
    demonstrate_filter(&single_filter, tokens, "Take only the first token")?;

    // Example 3: Limit larger than input
    println!("\n3. Limit larger than input:");
    let large_filter = LimitFilter::new(10);
    let tokens = create_test_tokens(&[
        "short", "input", "list"
    ]);
    demonstrate_filter(&large_filter, tokens, "Limit exceeds input size")?;

    // Example 4: Zero limit
    println!("\n4. Zero limit:");
    let zero_filter = LimitFilter::new(0);
    let tokens = create_test_tokens(&[
        "all", "tokens", "will", "be", "filtered", "out"
    ]);
    demonstrate_filter(&zero_filter, tokens, "Remove all tokens with zero limit")?;

    // Example 5: Large document truncation
    println!("\n5. Large document truncation:");
    let doc_filter = LimitFilter::new(5);
    let tokens = create_test_tokens(&[
        "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "and", "runs", "away"
    ]);
    demonstrate_filter(&doc_filter, tokens, "Document summary with first 5 tokens")?;

    // Example 6: Search result limiting
    println!("\n6. Search result limiting:");
    let search_filter = LimitFilter::new(4);
    let tokens = create_test_tokens(&[
        "rust", "programming", "language", "systems", "memory", "safe", "concurrent", "fast"
    ]);
    demonstrate_filter(&search_filter, tokens, "Top 4 search terms")?;

    // Example 7: Preview generation
    println!("\n7. Text preview generation:");
    let preview_filter = LimitFilter::new(6);
    let tokens = create_test_tokens(&[
        "In", "this", "comprehensive", "tutorial", "we", "will", "explore", "advanced", 
        "techniques", "for", "building", "robust", "applications"
    ]);
    demonstrate_filter(&preview_filter, tokens, "Article preview with 6 words")?;

    // Example 8: Performance testing with small set
    println!("\n8. Performance optimization:");
    let perf_filter = LimitFilter::new(2);
    let tokens = create_test_tokens(&[
        "performance", "optimization", "benchmark", "profiling", "analysis", "metrics"
    ]);
    demonstrate_filter(&perf_filter, tokens, "Quick analysis with 2 tokens")?;

    // Example 9: Stopped tokens handling
    println!("\n9. Stopped tokens interaction:");
    let limit_filter = LimitFilter::new(3);
    let stopped_tokens = vec![
        Token::new("normal", 0),
        Token::new("stopped", 1).stop(),
        Token::new("active", 2),
        Token::new("another", 3).stop(),
        Token::new("final", 4),
    ];
    let token_stream = Box::new(stopped_tokens.into_iter());
    demonstrate_filter_with_stopped(&limit_filter, token_stream, "Limit includes stopped tokens")?;

    // Example 10: Empty input handling
    println!("\n10. Empty input handling:");
    let empty_filter = LimitFilter::new(5);
    let tokens = create_test_tokens(&[]);
    demonstrate_filter(&empty_filter, tokens, "Empty token stream")?;

    println!("\n=== Filter Properties ===\n");
    let filter = LimitFilter::new(42);
    println!("Filter name: {}", filter.name());
    println!("Limit: {}", filter.limit());

    println!("\n=== Behavior Details ===\n");
    println!("Token processing:");
    println!("  • Takes first N tokens from stream");
    println!("  • Preserves token order");
    println!("  • Includes stopped tokens in count");
    println!("  • Maintains original token properties");
    println!("  • Stops iteration after limit reached");

    println!("\n=== Use Cases ===\n");
    println!("Text processing:");
    println!("  • Document preview generation");
    println!("  • Summary creation");
    println!("  • Content truncation");
    
    println!("\nSearch optimization:");
    println!("  • Result set limiting");
    println!("  • Performance optimization");
    println!("  • Memory usage control");
    
    println!("\nData analysis:");
    println!("  • Sample data extraction");
    println!("  • Quick testing with small sets");
    println!("  • Prototype development");

    println!("\n=== Pipeline Position ===\n");
    println!("Typical usage:");
    println!("  1. Tokenization");
    println!("  2. → Other filters (optional)");
    println!("  3. → LimitFilter (final stage)");
    println!("  • Often used as last filter");
    println!("  • Reduces processing in downstream components");

    println!("\n=== Performance Notes ===\n");
    println!("• Early termination after limit reached");
    println!("• O(limit) time complexity");
    println!("• Minimal memory overhead");
    println!("• Efficient for large token streams");

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
    
    let filtered_count = input_count - output_count;
    if filtered_count > 0 {
        println!("Filtered: {} token{}", filtered_count, 
            if filtered_count == 1 { "" } else { "s" });
    } else {
        println!("Filtered: none");
    }
    
    Ok(())
}

fn demonstrate_filter_with_stopped(
    filter: &dyn Filter,
    tokens: TokenStream,
    description: &str,
) -> Result<()> {
    println!("Description: {}", description);
    
    let input_tokens: Vec<Token> = tokens.collect();
    println!("Input tokens:");
    for (i, token) in input_tokens.iter().enumerate() {
        let status = if token.is_stopped() { " (stopped)" } else { "" };
        println!("  {}: '{}'{}", i, token.text, status);
    }
    
    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();
    
    println!("Output tokens:");
    for (i, token) in filtered_tokens.iter().enumerate() {
        let status = if token.is_stopped() { " (stopped)" } else { "" };
        println!("  {}: '{}'{}", i, token.text, status);
    }
    
    Ok(())
}