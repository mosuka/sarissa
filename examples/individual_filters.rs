//! Example demonstrating individual token filters.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{
    BoostFilter, Filter, LimitFilter, LowercaseFilter, RemoveEmptyFilter, StopFilter, StripFilter,
};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== Individual Token Filter Examples ===\n");

    // Example 1: LowercaseFilter
    println!("1. LowercaseFilter:");
    let lowercase_filter = LowercaseFilter::new();
    let tokens = create_test_tokens(&["Hello", "WORLD", "CamelCase", "UPPERCASE", "lowercase"]);
    demonstrate_filter(&lowercase_filter, tokens, "Mixed case text")?;

    // Example 2: StripFilter
    println!("\n2. StripFilter:");
    let strip_filter = StripFilter::new();
    let tokens = create_test_tokens(&["  hello  ", "world", "   trimmed   ", "", "  ", "test"]);
    demonstrate_filter(&strip_filter, tokens, "Text with whitespace")?;

    // Example 3: RemoveEmptyFilter
    println!("\n3. RemoveEmptyFilter:");
    let remove_empty_filter = RemoveEmptyFilter::new();
    let tokens = create_test_tokens(&["hello", "", "world", "   ", "test", ""]);
    demonstrate_filter(&remove_empty_filter, tokens, "Text with empty tokens")?;

    // Example 4: BoostFilter
    println!("\n4. BoostFilter:");
    let boost_filter = BoostFilter::new(2.5);
    let tokens = create_test_tokens(&["important", "normal", "significant"]);
    // Set different initial boost values
    let token_vec = tokens_to_vec(tokens)?;
    let boosted_tokens: Vec<Token> = token_vec
        .into_iter()
        .enumerate()
        .map(|(i, mut token)| {
            token.boost = 1.0 + i as f32 * 0.5; // 1.0, 1.5, 2.0
            token
        })
        .collect();
    let final_tokens = Box::new(boosted_tokens.into_iter());
    demonstrate_filter_with_boost(
        &boost_filter,
        final_tokens,
        "Text with different boost values",
    )?;

    // Example 5: LimitFilter
    println!("\n5. LimitFilter:");
    let limit_filter = LimitFilter::new(3);
    let tokens = create_test_tokens(&["first", "second", "third", "fourth", "fifth", "sixth"]);
    demonstrate_filter(&limit_filter, tokens, "Long text (limited to 3 tokens)")?;

    // Example 6: StopFilter with default English stop words
    println!("\n6. StopFilter (default English):");
    let stop_filter = StopFilter::new();
    let tokens = create_test_tokens(&["the", "quick", "brown", "fox", "and", "the", "lazy", "dog"]);
    demonstrate_filter(&stop_filter, tokens, "Text with stop words")?;

    // Example 7: StopFilter with custom stop words
    println!("\n7. StopFilter (custom stop words):");
    let custom_stop_filter = StopFilter::from_words(vec![
        "quick".to_string(),
        "lazy".to_string(),
        "over".to_string(),
    ]);
    let tokens = create_test_tokens(&[
        "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
    ]);
    demonstrate_filter(&custom_stop_filter, tokens, "Text with custom stop words")?;

    // Example 8: StopFilter preserving stopped tokens
    println!("\n8. StopFilter (preserve stopped tokens):");
    let preserve_stop_filter =
        StopFilter::from_words(vec!["the".to_string(), "and".to_string()]).remove_stopped(false);
    let tokens = create_test_tokens(&["the", "cat", "and", "the", "dog", "are", "friends"]);
    demonstrate_filter_with_stopped(&preserve_stop_filter, tokens, "Text preserving stop words")?;

    println!("\n=== Filter Properties ===\n");

    // Show filter properties
    println!("Filter names:");
    println!("  LowercaseFilter: {}", LowercaseFilter::new().name());
    println!("  StripFilter: {}", StripFilter::new().name());
    println!("  RemoveEmptyFilter: {}", RemoveEmptyFilter::new().name());
    println!("  BoostFilter: {}", BoostFilter::new(1.0).name());
    println!("  LimitFilter: {}", LimitFilter::new(10).name());
    println!("  StopFilter: {}", StopFilter::new().name());

    println!("\nStopFilter statistics:");
    let stop_filter = StopFilter::new();
    println!("  Stop words count: {}", stop_filter.len());
    println!("  Is empty: {}", stop_filter.is_empty());
    println!("  Contains 'the': {}", stop_filter.is_stop_word("the"));
    println!("  Contains 'rust': {}", stop_filter.is_stop_word("rust"));

    println!("\nBoostFilter properties:");
    let boost_filter = BoostFilter::new(3.14);
    println!("  Boost factor: {}", boost_filter.boost());

    println!("\nLimitFilter properties:");
    let limit_filter = LimitFilter::new(42);
    println!("  Limit: {}", limit_filter.limit());

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

fn tokens_to_vec(tokens: TokenStream) -> Result<Vec<Token>> {
    Ok(tokens.collect())
}

fn demonstrate_filter(filter: &dyn Filter, tokens: TokenStream, description: &str) -> Result<()> {
    println!("Description: {description}");

    let input_tokens: Vec<Token> = tokens.collect();
    let input_count = input_tokens.len();
    println!(
        "Input tokens: {:?}",
        input_tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );

    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();

    println!(
        "Output tokens: {:?}",
        filtered_tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );
    println!("Count: {} → {}", input_count, filtered_tokens.len());

    Ok(())
}

fn demonstrate_filter_with_boost(
    filter: &dyn Filter,
    tokens: TokenStream,
    description: &str,
) -> Result<()> {
    println!("Description: {description}");

    let input_tokens: Vec<Token> = tokens.collect();
    println!("Input tokens with boost:");
    for token in &input_tokens {
        println!("  '{}' (boost: {:.2})", token.text, token.boost);
    }

    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();

    println!("Output tokens with boost:");
    for token in &filtered_tokens {
        println!("  '{}' (boost: {:.2})", token.text, token.boost);
    }

    Ok(())
}

fn demonstrate_filter_with_stopped(
    filter: &dyn Filter,
    tokens: TokenStream,
    description: &str,
) -> Result<()> {
    println!("Description: {description}");

    let input_tokens: Vec<Token> = tokens.collect();
    println!(
        "Input tokens: {:?}",
        input_tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );

    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();

    println!("Output tokens:");
    for (i, token) in filtered_tokens.iter().enumerate() {
        let status = if token.is_stopped() { " (stopped)" } else { "" };
        println!("  {}: '{}'{}", i, token.text, status);
    }

    Ok(())
}
