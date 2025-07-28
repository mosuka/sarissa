//! Example demonstrating the IdentityStemmer (no-op stemmer).

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{Filter, IdentityStemmer, StemFilter, Stemmer};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== IdentityStemmer Examples ===\n");

    let identity_stemmer = IdentityStemmer::new();

    // Example 1: Basic demonstration (no changes)
    println!("1. Basic demonstration:");
    let basic_words = vec![
        "running", "runs", "ran", "runner", "flies", "flying", "fly", "flew",
    ];
    demonstrate_stemmer(&identity_stemmer, &basic_words)?;

    // Example 2: Complex words remain unchanged
    println!("\n2. Complex words remain unchanged:");
    let complex_words = vec![
        "beautiful",
        "beautifully",
        "beauty",
        "beautification",
        "programming",
        "programs",
        "programmer",
        "programmed",
    ];
    demonstrate_stemmer(&identity_stemmer, &complex_words)?;

    // Example 3: Technical terms preservation
    println!("\n3. Technical terms preservation:");
    let technical_words = vec![
        "optimization",
        "optimize",
        "optimized",
        "optimizing",
        "optimizer",
        "classification",
        "classify",
        "classified",
        "classifying",
        "classifier",
    ];
    demonstrate_stemmer(&identity_stemmer, &technical_words)?;

    // Example 4: StemFilter with IdentityStemmer
    println!("\n4. StemFilter with IdentityStemmer:");
    let identity_filter = StemFilter::with_stemmer(Box::new(IdentityStemmer::new()));
    let tokens = create_test_tokens(&["running", "flies", "beautiful", "programming", "happiness"]);
    demonstrate_filter(&identity_filter, tokens, "Token stream with no stemming")?;

    // Example 5: Exact matching scenario
    println!("\n5. Exact matching scenario:");
    let exact_words = vec![
        "JavaScript",
        "TypeScript",
        "WebAssembly",
        "MongoDB",
        "PostgreSQL",
    ];
    demonstrate_stemmer(&identity_stemmer, &exact_words)?;

    // Example 6: Proper nouns and names
    println!("\n6. Proper nouns and names:");
    let proper_nouns = vec!["London", "Paris", "Tokyo", "Microsoft", "Google", "Amazon"];
    demonstrate_stemmer(&identity_stemmer, &proper_nouns)?;

    // Example 7: Code identifiers
    println!("\n7. Code identifiers:");
    let code_identifiers = vec![
        "getUserName",
        "API_ENDPOINT",
        "MAX_SIZE",
        "isValid",
        "toString",
        "valueOf",
    ];
    demonstrate_stemmer(&identity_stemmer, &code_identifiers)?;

    // Example 8: Debugging and development
    println!("\n8. Debugging and development:");
    let debug_tokens = create_test_tokens(&[
        "originalWord",
        "stemmedWord",
        "processedToken",
        "debugMode",
        "testCase",
    ]);
    demonstrate_filter(
        &identity_filter,
        debug_tokens,
        "Development and debugging tokens",
    )?;

    // Example 9: Multilingual terms
    println!("\n9. Multilingual terms:");
    let multilingual_words = vec!["café", "naïve", "résumé", "piñata", "jalapeño", "Москва"];
    demonstrate_stemmer(&identity_stemmer, &multilingual_words)?;

    // Example 10: Domain-specific vocabulary
    println!("\n10. Domain-specific vocabulary:");
    let domain_words = vec![
        "OAuth2",
        "RESTful",
        "GraphQL",
        "Kubernetes",
        "DevOps",
        "CI/CD",
    ];
    demonstrate_stemmer(&identity_stemmer, &domain_words)?;

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
    println!("Stemmer: {} (No changes)", stemmer.name());

    for word in words {
        let stemmed = stemmer.stem(word);
        if stemmed != *word {
            println!("  '{}' → '{}' (UNEXPECTED CHANGE!)", word, stemmed);
        } else {
            println!("  '{}' (preserved)", word);
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

    // Show the stemming transformations (should be none for Identity)
    println!("Transformations:");
    let mut changes = 0;
    for (i, (input, output)) in input_clone.iter().zip(filtered_tokens.iter()).enumerate() {
        if input.text != output.text {
            println!(
                "  {}: '{}' → '{}' (UNEXPECTED!)",
                i, input.text, output.text
            );
            changes += 1;
        }
    }
    if changes == 0 {
        println!("  (no changes - as expected)");
    } else {
        println!("  WARNING: {} unexpected changes detected!", changes);
    }

    Ok(())
}
