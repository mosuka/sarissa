//! Example demonstrating the StripFilter.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{Filter, StripFilter};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== StripFilter Examples ===\n");

    let filter = StripFilter::new();

    // Example 1: Basic whitespace trimming
    println!("1. Basic whitespace trimming:");
    let tokens = create_test_tokens(&[
        "  hello  ", "world", "   test   "
    ]);
    demonstrate_filter(&filter, tokens, "Leading and trailing spaces")?;

    // Example 2: Mixed whitespace types
    println!("\n2. Mixed whitespace types:");
    let tokens = create_test_tokens(&[
        "\t\ttab", "space ", "\n\rnewline\n", "   mixed \t\n "
    ]);
    demonstrate_filter(&filter, tokens, "Tabs, spaces, and newlines")?;

    // Example 3: Empty after trimming
    println!("\n3. Empty after trimming:");
    let tokens = create_test_tokens(&[
        "normal", "   ", "\t\t", " \n \r ", "text"
    ]);
    demonstrate_filter(&filter, tokens, "Whitespace-only tokens become stopped")?;

    // Example 4: No trimming needed
    println!("\n4. No trimming needed:");
    let tokens = create_test_tokens(&[
        "clean", "text", "without", "extra", "spaces"
    ]);
    demonstrate_filter(&filter, tokens, "Already clean text")?;

    // Example 5: Extreme whitespace
    println!("\n5. Extreme whitespace:");
    let tokens = create_test_tokens(&[
        "        lots        ", "  of  ", "    whitespace    "
    ]);
    demonstrate_filter(&filter, tokens, "Excessive whitespace")?;

    // Example 6: User input cleaning
    println!("\n6. User input cleaning:");
    let tokens = create_test_tokens(&[
        " John ", "  Doe  ", "   ", " user@example.com ", ""
    ]);
    demonstrate_filter(&filter, tokens, "Typical user form input")?;

    // Example 7: Code snippet cleaning
    println!("\n7. Code snippet tokens:");
    let tokens = create_test_tokens(&[
        "  function  ", " getName() ", "   {   ", " return ", " name; ", "  }  "
    ]);
    demonstrate_filter(&filter, tokens, "Code with formatting spaces")?;

    // Example 8: Unicode whitespace
    println!("\n8. Unicode whitespace:");
    let tokens = create_test_tokens(&[
        "\u{00A0}unicode\u{00A0}", "\u{2000}em\u{2000}", "normal"
    ]);
    demonstrate_filter(&filter, tokens, "Unicode whitespace characters")?;

    println!("\n=== Filter Properties ===\n");
    println!("Filter name: {}", filter.name());
    
    println!("\n=== Whitespace Types Handled ===\n");
    println!("Standard whitespace characters:");
    println!("  • Space (U+0020): ' '");
    println!("  • Tab (U+0009): '\\t'");
    println!("  • Newline (U+000A): '\\n'");
    println!("  • Carriage return (U+000D): '\\r'");
    println!("  • Form feed (U+000C): '\\f'");
    println!("  • Vertical tab (U+000B): '\\v'");
    
    println!("\nUnicode whitespace (if present):");
    println!("  • Non-breaking space (U+00A0)");
    println!("  • En quad (U+2000)");
    println!("  • Em quad (U+2001)");
    println!("  • And other Unicode whitespace");

    println!("\n=== Behavior Details ===\n");
    println!("Token processing:");
    println!("  • Skips already stopped tokens");
    println!("  • Trims leading and trailing whitespace");
    println!("  • Stops tokens that become empty after trimming");
    println!("  • Preserves internal whitespace");
    println!("  • Maintains original token positions");

    println!("\n=== Use Cases ===\n");
    println!("Data cleaning:");
    println!("  • User input normalization");
    println!("  • CSV/TSV data preprocessing");
    println!("  • Form data validation");
    
    println!("\nText processing:");
    println!("  • Document parsing cleanup");
    println!("  • Code tokenization");
    println!("  • Log file processing");
    
    println!("\nSearch preprocessing:");
    println!("  • Query normalization");
    println!("  • Index preparation");
    println!("  • Keyword extraction");

    println!("\n=== Pipeline Position ===\n");
    println!("Typical usage order:");
    println!("  1. Tokenization");
    println!("  2. → StripFilter (early cleanup)");
    println!("  3. → LowercaseFilter");
    println!("  4. → StopFilter");
    println!("  5. → Other filters...");

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
    println!("Input:  {:?}", 
        input_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    let input_clone = input_tokens.clone();
    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();
    
    println!("Output: [{}]", 
        filtered_tokens.iter()
            .map(|t| if t.is_stopped() { 
                format!("\"{}\" (stopped)", t.text) 
            } else { 
                format!("\"{}\"", t.text) 
            })
            .collect::<Vec<_>>()
            .join(", "));
    
    // Show detailed transformations
    println!("Transformations:");
    for (i, (input, output)) in input_clone.iter()
        .zip(filtered_tokens.iter())
        .enumerate() {
        if input.text != output.text || output.is_stopped() {
            let status = if output.is_stopped() { " (→ stopped)" } else { "" };
            println!("  {}: {:?} → {:?}{}", i, input.text, output.text, status);
        }
    }
    
    Ok(())
}