//! Example demonstrating the BoostFilter.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{BoostFilter, Filter};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== BoostFilter Examples ===\n");

    // Example 1: Basic boost multiplication
    println!("1. Basic boost multiplication:");
    let boost_filter = BoostFilter::new(2.0);
    let tokens = create_boosted_tokens(&[
        ("important", 1.0),
        ("normal", 1.0),
        ("critical", 1.5),
    ]);
    demonstrate_boost_filter(&boost_filter, tokens, "2x boost multiplier")?;

    // Example 2: Fractional boost (dampening)
    println!("\n2. Fractional boost (dampening):");
    let dampen_filter = BoostFilter::new(0.5);
    let tokens = create_boosted_tokens(&[
        ("loud", 3.0),
        ("medium", 2.0),
        ("quiet", 1.0),
    ]);
    demonstrate_boost_filter(&dampen_filter, tokens, "0.5x dampening factor")?;

    // Example 3: High precision boost
    println!("\n3. High precision boost:");
    let precise_filter = BoostFilter::new(1.25);
    let tokens = create_boosted_tokens(&[
        ("keyword", 1.0),
        ("title", 2.0),
        ("meta", 0.8),
    ]);
    demonstrate_boost_filter(&precise_filter, tokens, "1.25x precision boost")?;

    // Example 4: Zero boost handling
    println!("\n4. Zero boost handling:");
    let boost_filter = BoostFilter::new(3.0);
    let tokens = create_boosted_tokens(&[
        ("active", 1.0),
        ("inactive", 0.0),
        ("disabled", 0.0),
        ("enabled", 2.0),
    ]);
    demonstrate_boost_filter(&boost_filter, tokens, "Handling zero boost values")?;

    // Example 5: Large boost values
    println!("\n5. Large boost values:");
    let large_filter = BoostFilter::new(10.0);
    let tokens = create_boosted_tokens(&[
        ("regular", 1.0),
        ("important", 1.5),
        ("critical", 2.0),
    ]);
    demonstrate_boost_filter(&large_filter, tokens, "10x large boost")?;

    // Example 6: Search relevance simulation
    println!("\n6. Search relevance boost:");
    let relevance_filter = BoostFilter::new(1.5);
    let tokens = create_boosted_tokens(&[
        ("rust", 3.0),        // Exact match
        ("programming", 2.0), // Related term
        ("language", 1.5),    // Context term
        ("the", 0.1),         // Stop word
    ]);
    demonstrate_boost_filter(&relevance_filter, tokens, "Search term relevance")?;

    // Example 7: Document section boosting
    println!("\n7. Document section boosting:");
    let section_filter = BoostFilter::new(2.5);
    let tokens = create_boosted_tokens(&[
        ("title_word", 4.0),    // Title has highest weight
        ("header_word", 2.0),   // Headers are important
        ("body_word", 1.0),     // Body text baseline
        ("footer_word", 0.5),   // Footer less important
    ]);
    demonstrate_boost_filter(&section_filter, tokens, "Document structure weighting")?;

    // Example 8: Category-based boosting
    println!("\n8. Category-based boosting:");
    let category_filter = BoostFilter::new(1.8);
    let tokens = create_boosted_tokens(&[
        ("technology", 2.5),  // Tech category
        ("business", 2.0),    // Business category
        ("general", 1.0),     // General content
        ("archive", 0.3),     // Archived content
    ]);
    demonstrate_boost_filter(&category_filter, tokens, "Content category weighting")?;

    println!("\n=== Filter Properties ===\n");
    let filter = BoostFilter::new(3.14);
    println!("Filter name: {}", filter.name());
    println!("Boost factor: {:.2}", filter.boost());

    println!("\n=== Boost Strategies ===\n");
    println!("1. Multiplicative boost (default):");
    println!("   new_boost = original_boost × boost_factor");
    println!("   • Preserves relative importance");
    println!("   • Scales all values proportionally");
    
    println!("\n2. Common boost factors:");
    println!("   • 1.0: No change (identity)");
    println!("   • > 1.0: Increase importance");
    println!("   • 0.0 - 1.0: Decrease importance");
    println!("   • 0.0: Effectively disable token");

    println!("\n=== Use Cases ===\n");
    println!("Search engines:");
    println!("  • Query term matching boost");
    println!("  • Document freshness weighting");
    println!("  • Authority domain boost");
    
    println!("\nContent management:");
    println!("  • Featured content highlighting");
    println!("  • Section importance weighting");
    println!("  • User preference adjustments");
    
    println!("\nMachine learning:");
    println!("  • Feature importance scaling");
    println!("  • Training data weighting");
    println!("  • Model confidence adjustment");

    println!("\n=== Performance Notes ===\n");
    println!("• O(1) per token processing");
    println!("• Minimal memory overhead");
    println!("• Skips stopped tokens automatically");
    println!("• Safe floating-point arithmetic");

    Ok(())
}

fn create_boosted_tokens(data: &[(&str, f32)]) -> TokenStream {
    let tokens: Vec<Token> = data.iter()
        .enumerate()
        .map(|(i, (text, boost))| {
            let mut token = Token::new(*text, i);
            token.boost = *boost;
            token
        })
        .collect();
    Box::new(tokens.into_iter())
}

fn demonstrate_boost_filter(
    filter: &dyn Filter,
    tokens: TokenStream,
    description: &str,
) -> Result<()> {
    println!("Description: {}", description);
    
    let input_tokens: Vec<Token> = tokens.collect();
    println!("Input tokens with boost:");
    for (i, token) in input_tokens.iter().enumerate() {
        println!("  {}: '{}' (boost: {:.2})", i, token.text, token.boost);
    }
    
    let input_clone = input_tokens.clone();
    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();
    
    println!("Output tokens with boost:");
    for (i, token) in filtered_tokens.iter().enumerate() {
        println!("  {}: '{}' (boost: {:.2})", i, token.text, token.boost);
    }
    
    // Show boost changes
    println!("Boost changes:");
    for (i, (input, output)) in input_clone.iter()
        .zip(filtered_tokens.iter())
        .enumerate() {
        let change = output.boost - input.boost;
        let multiplier = if input.boost != 0.0 { output.boost / input.boost } else { 0.0 };
        println!("  {}: {:.2} → {:.2} (×{:.2}, +{:.2})", 
            i, input.boost, output.boost, multiplier, change);
    }
    
    Ok(())
}