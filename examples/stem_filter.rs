//! Example demonstrating the StemFilter with different stemming algorithms.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{
    Filter, IdentityStemmer, PorterStemmer, SimpleStemmer, StemFilter, Stemmer,
};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== StemFilter Examples ===\n");

    // Example 1: Porter Stemmer (default)
    println!("1. Porter Stemmer (default):");
    let porter_filter = StemFilter::new();
    let tokens = create_test_tokens(&[
        "running",
        "runs",
        "ran",
        "runner",
        "programming",
        "programs",
        "programmer",
    ]);
    demonstrate_filter(&porter_filter, tokens, "High-precision linguistic stemming")?;

    // Example 2: Simple Stemmer
    println!("\n2. Simple Stemmer:");
    let simple_filter = StemFilter::simple();
    let tokens = create_test_tokens(&[
        "running",
        "runs",
        "ran",
        "runner",
        "programming",
        "programs",
        "programmer",
    ]);
    demonstrate_filter(&simple_filter, tokens, "Fast suffix-based stemming")?;

    // Example 3: Identity Stemmer (no changes)
    println!("\n3. Identity Stemmer:");
    let identity_filter = StemFilter::with_stemmer(Box::new(IdentityStemmer::new()));
    let tokens = create_test_tokens(&["running", "runs", "ran", "runner"]);
    demonstrate_filter(&identity_filter, tokens, "No stemming applied")?;

    // Example 4: Custom Simple Stemmer
    println!("\n4. Custom Simple Stemmer:");
    let custom_stemmer =
        SimpleStemmer::with_suffixes(vec!["ing".to_string(), "ed".to_string(), "ly".to_string()]);
    let custom_filter = StemFilter::with_stemmer(Box::new(custom_stemmer));
    let tokens = create_test_tokens(&[
        "walking", "walked", "quickly", "testing", "tested", "slowly",
    ]);
    demonstrate_filter(&custom_filter, tokens, "Custom suffix removal")?;

    // Example 5: English word variations
    println!("\n5. English word variations:");
    let porter_filter = StemFilter::new();
    let tokens = create_test_tokens(&["beautiful", "beautifully", "beauty", "beautification"]);
    demonstrate_filter(&porter_filter, tokens, "Word family normalization")?;

    // Example 6: Technical terms
    println!("\n6. Technical terms:");
    let porter_filter = StemFilter::new();
    let tokens = create_test_tokens(&[
        "optimization",
        "optimize",
        "optimized",
        "optimizing",
        "optimizer",
    ]);
    demonstrate_filter(&porter_filter, tokens, "Technical vocabulary")?;

    // Example 7: Past tense handling
    println!("\n7. Past tense forms:");
    let porter_filter = StemFilter::new();
    let tokens = create_test_tokens(&[
        "connected",
        "disconnected",
        "reconnected",
        "connection",
        "connecting",
    ]);
    demonstrate_filter(&porter_filter, tokens, "Past tense and related forms")?;

    // Example 8: Comparative forms
    println!("\n8. Comparative and superlative:");
    let simple_filter = StemFilter::simple();
    let tokens = create_test_tokens(&[
        "faster", "fastest", "slower", "slowest", "bigger", "biggest",
    ]);
    demonstrate_filter(&simple_filter, tokens, "Comparison forms")?;

    // Example 9: Algorithm comparison
    println!("\n9. Algorithm Comparison:");
    compare_stemming_algorithms(&[
        "running",
        "beautiful",
        "optimization",
        "traditional",
        "itemization",
    ])?;

    // Example 10: Real-world text
    println!("\n10. Real-world text processing:");
    let porter_filter = StemFilter::new();
    let tokens = create_test_tokens(&[
        "The",
        "developers",
        "are",
        "optimizing",
        "applications",
        "using",
        "advanced",
        "algorithms",
    ]);
    demonstrate_filter(&porter_filter, tokens, "Sentence processing")?;

    println!("\n=== Stemmer Properties ===\n");
    let porter = PorterStemmer::new();
    let simple = SimpleStemmer::new();
    let identity = IdentityStemmer::new();

    println!("Available stemmers:");
    println!("  Porter: {} (linguistic rules)", porter.name());
    println!("  Simple: {} (suffix removal)", simple.name());
    println!("  Identity: {} (no changes)", identity.name());

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
    println!("Description: {description}");

    let input_tokens: Vec<Token> = tokens.collect();
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

    // Show stemming transformations
    println!("Transformations:");
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

    Ok(())
}

fn compare_stemming_algorithms(words: &[&str]) -> Result<()> {
    let porter = PorterStemmer::new();
    let simple = SimpleStemmer::new();
    let identity = IdentityStemmer::new();

    println!("Comparing stemming algorithms:");
    println!("Word          | Porter     | Simple     | Identity");
    println!("--------------|------------|------------|------------");

    for word in words {
        let porter_result = porter.stem(word);
        let simple_result = simple.stem(word);
        let identity_result = identity.stem(word);

        println!("{word:13} | {porter_result:10} | {simple_result:10} | {identity_result:10}");
    }

    Ok(())
}
