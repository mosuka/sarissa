//! Example demonstrating stemming filters and different stemmer algorithms.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{
    Filter, IdentityStemmer, PorterStemmer, SimpleStemmer, StemFilter, Stemmer,
};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== Stemming Filter Examples ===\n");

    // Test words for stemming demonstration
    let test_words = vec![
        "running",
        "runs",
        "ran",
        "runner",
        "flies",
        "flying",
        "fly",
        "flew",
        "beautiful",
        "beautifully",
        "beauty",
        "programming",
        "programs",
        "programmer",
        "programmed",
        "happiness",
        "happy",
        "happily",
        "happier",
        "connection",
        "connected",
        "connecting",
        "connect",
        "organization",
        "organize",
        "organized",
        "organizing",
        "traditional",
        "tradition",
        "traditionally",
        "itemization",
        "itemize",
        "items",
        "sensational",
        "sensation",
        "sensations",
    ];

    // Example 1: Porter Stemmer
    println!("1. Porter Stemmer:");
    let porter_stemmer = PorterStemmer::new();
    demonstrate_stemmer(&porter_stemmer, &test_words)?;

    // Example 2: Simple Stemmer
    println!("\n2. Simple Stemmer:");
    let simple_stemmer = SimpleStemmer::new();
    demonstrate_stemmer(&simple_stemmer, &test_words)?;

    // Example 3: Identity Stemmer
    println!("\n3. Identity Stemmer:");
    let identity_stemmer = IdentityStemmer::new();
    demonstrate_stemmer(&identity_stemmer, &test_words[0..5])?; // Show fewer for brevity

    // Example 4: StemFilter with Porter Stemmer
    println!("\n4. StemFilter with Porter Stemmer:");
    let porter_filter = StemFilter::new(); // Default uses Porter
    let tokens = create_test_tokens(&test_words[0..8]);
    demonstrate_filter(&porter_filter, tokens, "Text with various word forms")?;

    // Example 5: StemFilter with Simple Stemmer
    println!("\n5. StemFilter with Simple Stemmer:");
    let simple_filter = StemFilter::simple();
    let tokens = create_test_tokens(&test_words[0..8]);
    demonstrate_filter(&simple_filter, tokens, "Text with various word forms")?;

    // Example 6: StemFilter with custom stemmer
    println!("\n6. StemFilter with Identity Stemmer:");
    let identity_filter = StemFilter::with_stemmer(Box::new(IdentityStemmer::new()));
    let tokens = create_test_tokens(&test_words[0..5]);
    demonstrate_filter(&identity_filter, tokens, "Text without stemming")?;

    // Example 7: Custom Simple Stemmer with specific suffixes
    println!("\n7. Custom Simple Stemmer:");
    let custom_simple = SimpleStemmer::with_suffixes(vec![
        "ing".to_string(),
        "ed".to_string(),
        "er".to_string(),
        "ly".to_string(),
    ]);
    let custom_words = vec!["running", "walked", "bigger", "quickly", "testing"];
    demonstrate_stemmer(&custom_simple, &custom_words)?;

    // Example 8: Stemming comparison
    println!("\n8. Stemming Algorithm Comparison:");
    let comparison_words = vec![
        "beautiful",
        "running",
        "flies",
        "programming",
        "traditional",
        "itemization",
    ];
    compare_stemmers(&comparison_words)?;

    // Example 9: Handling edge cases
    println!("\n9. Edge Cases:");
    let edge_cases = vec!["a", "I", "am", "go", "be", "do", "it"];
    println!("Short words and common verbs:");
    compare_stemmers(&edge_cases)?;

    // Example 10: Real-world text example
    println!("\n10. Real-world Text Example:");
    let real_text_words = vec![
        "The",
        "developers",
        "are",
        "programming",
        "applications",
        "using",
        "advanced",
        "algorithms",
        "and",
        "optimizations",
    ];
    let tokens = create_test_tokens(&real_text_words);
    let porter_filter = StemFilter::new();
    demonstrate_filter(&porter_filter, tokens, "Real-world sentence")?;

    println!("\n=== Stemmer Properties ===\n");

    // Show stemmer properties
    println!("Stemmer names:");
    println!("  PorterStemmer: {}", PorterStemmer::new().name());
    println!("  SimpleStemmer: {}", SimpleStemmer::new().name());
    println!("  IdentityStemmer: {}", IdentityStemmer::new().name());

    println!("\nFilter names:");
    println!("  StemFilter (Porter): {}", StemFilter::new().name());
    println!("  StemFilter (Simple): {}", StemFilter::simple().name());
    println!(
        "  StemFilter (Identity): {}",
        StemFilter::with_stemmer(Box::new(IdentityStemmer::new())).name()
    );

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

fn demonstrate_stemmer(stemmer: &dyn Stemmer, words: &[&str]) -> Result<()> {
    println!(
        "Stemmer: {} ({})",
        stemmer.name(),
        match stemmer.name() {
            "porter" => "Linguistic algorithm",
            "simple" => "Suffix removal",
            "identity" => "No changes",
            _ => "Unknown",
        }
    );

    for word in words {
        let stemmed = stemmer.stem(word);
        if stemmed != *word {
            println!("  '{}' → '{}'", word, stemmed);
        } else {
            println!("  '{}' (unchanged)", word);
        }
    }

    Ok(())
}

fn demonstrate_filter(filter: &dyn Filter, tokens: TokenStream, description: &str) -> Result<()> {
    println!("Description: {}", description);

    let input_tokens: Vec<Token> = tokens.collect();
    println!(
        "Input tokens: {:?}",
        input_tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );

    let input_clone = input_tokens.clone();
    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();

    println!(
        "Output tokens: {:?}",
        filtered_tokens.iter().map(|t| &t.text).collect::<Vec<_>>()
    );

    // Show the stemming transformations
    println!("Transformations:");
    for (i, (input, output)) in input_clone.iter().zip(filtered_tokens.iter()).enumerate() {
        if input.text != output.text {
            println!("  {}: '{}' → '{}'", i, input.text, output.text);
        }
    }

    Ok(())
}

fn compare_stemmers(words: &[&str]) -> Result<()> {
    let porter = PorterStemmer::new();
    let simple = SimpleStemmer::new();
    let identity = IdentityStemmer::new();

    println!("Word        | Porter     | Simple     | Identity");
    println!("------------|------------|------------|------------");

    for word in words {
        let porter_result = porter.stem(word);
        let simple_result = simple.stem(word);
        let identity_result = identity.stem(word);

        println!(
            "{:11} | {:10} | {:10} | {:10}",
            word, porter_result, simple_result, identity_result
        );
    }

    Ok(())
}
