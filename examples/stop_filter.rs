//! Example demonstrating the StopFilter.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{Filter, StopFilter};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== StopFilter Examples ===\n");

    // Example 1: Default English stop words
    println!("1. Default English stop words:");
    let default_filter = StopFilter::new();
    let tokens = create_test_tokens(&[
        "the", "quick", "brown", "fox", "and", "the", "lazy", "dog"
    ]);
    demonstrate_filter(&default_filter, tokens, "Classic sentence with stop words")?;

    // Example 2: Custom stop words
    println!("\n2. Custom stop words:");
    let custom_filter = StopFilter::from_words(vec![
        "quick".to_string(), "lazy".to_string(), "over".to_string()
    ]);
    let tokens = create_test_tokens(&[
        "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"
    ]);
    demonstrate_filter(&custom_filter, tokens, "Custom stop word list")?;

    // Example 3: Programming-specific stop words
    println!("\n3. Programming-specific stop words:");
    let prog_filter = StopFilter::from_words(vec![
        "let".to_string(), "const".to_string(), "var".to_string(), "function".to_string()
    ]);
    let tokens = create_test_tokens(&[
        "let", "user", "const", "API_KEY", "function", "getName", "var", "result"
    ]);
    demonstrate_filter(&prog_filter, tokens, "Code keywords as stop words")?;

    // Example 4: Preserve stopped tokens (don't remove)
    println!("\n4. Preserve stopped tokens:");
    let preserve_filter = StopFilter::from_words(vec![
        "the".to_string(), "and".to_string()
    ]).remove_stopped(false);
    let tokens = create_test_tokens(&[
        "the", "cat", "and", "the", "dog", "are", "friends"
    ]);
    demonstrate_filter_with_stopped(&preserve_filter, tokens, "Mark as stopped but keep tokens")?;

    // Example 5: Empty stop word list
    println!("\n5. Empty stop word list:");
    let empty_filter = StopFilter::from_words(Vec::<String>::new());
    let tokens = create_test_tokens(&["the", "quick", "brown", "fox"]);
    demonstrate_filter(&empty_filter, tokens, "No stop words defined")?;

    // Example 6: Social media stop words
    println!("\n6. Social media stop words:");
    let social_filter = StopFilter::from_words(vec![
        "lol".to_string(), "omg".to_string(), "btw".to_string(), "imho".to_string()
    ]);
    let tokens = create_test_tokens(&[
        "lol", "this", "is", "omg", "amazing", "btw", "great", "work", "imho"
    ]);
    demonstrate_filter(&social_filter, tokens, "Social media abbreviations")?;

    // Example 7: Multilingual stop words
    println!("\n7. Multilingual stop words:");
    let multi_filter = StopFilter::from_words(vec![
        "the".to_string(), "le".to_string(), "der".to_string(), "el".to_string()
    ]);
    let tokens = create_test_tokens(&[
        "the", "cat", "le", "chat", "der", "Hund", "el", "gato"
    ]);
    demonstrate_filter(&multi_filter, tokens, "Articles in different languages")?;

    // Example 8: Case sensitivity
    println!("\n8. Case sensitivity:");
    let case_filter = StopFilter::from_words(vec![
        "the".to_string(), "AND".to_string()
    ]);
    let tokens = create_test_tokens(&[
        "the", "The", "THE", "and", "AND", "brown", "fox"
    ]);
    demonstrate_filter(&case_filter, tokens, "Stop words are case-sensitive")?;

    println!("\n=== Filter Properties ===\n");
    let default_filter = StopFilter::new();
    println!("Filter name: {}", default_filter.name());
    println!("Default stop words count: {}", default_filter.len());
    println!("Is empty: {}", default_filter.is_empty());
    
    println!("\nSample stop words check:");
    let sample_words = ["the", "and", "or", "but", "rust", "programming"];
    for word in sample_words {
        println!("  '{}': {}", word, 
            if default_filter.is_stop_word(word) { "✓ stop word" } else { "✗ not stop word" });
    }

    println!("\n=== Stop Word Strategies ===\n");
    println!("1. Remove stopped tokens (default):");
    println!("   • Reduces token count");
    println!("   • Good for search and analysis");
    println!("   • Saves memory and processing");
    
    println!("\n2. Preserve stopped tokens:");
    println!("   • Maintains original token count");
    println!("   • Good for syntax analysis");
    println!("   • Allows downstream filters to decide");

    println!("\n=== Use Cases ===\n");
    println!("Default English stop words:");
    println!("  • General text processing");
    println!("  • Search engines");
    println!("  • Document similarity");
    println!("  • Topic modeling");
    
    println!("\nCustom stop words:");
    println!("  • Domain-specific filtering");
    println!("  • Code analysis");
    println!("  • Multilingual processing");
    println!("  • Social media analysis");

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
    println!("Input:  {:?}", 
        input_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();
    
    println!("Output: {:?}", 
        filtered_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    println!("Count: {} → {}", input_count, filtered_tokens.len());
    
    Ok(())
}

fn demonstrate_filter_with_stopped(
    filter: &dyn Filter,
    tokens: TokenStream,
    description: &str,
) -> Result<()> {
    println!("Description: {}", description);
    
    let input_tokens: Vec<Token> = tokens.collect();
    println!("Input:  {:?}", 
        input_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();
    
    println!("Output tokens:");
    for (i, token) in filtered_tokens.iter().enumerate() {
        let status = if token.is_stopped() { " (stopped)" } else { "" };
        println!("  {}: '{}'{}",  i, token.text, status);
    }
    
    Ok(())
}