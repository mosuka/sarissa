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
        "running", "runs", "ran", "runner",
        "flies", "flying", "fly", "flew"
    ];
    demonstrate_stemmer(&identity_stemmer, &basic_words)?;

    // Example 2: Complex words remain unchanged
    println!("\n2. Complex words remain unchanged:");
    let complex_words = vec![
        "beautiful", "beautifully", "beauty", "beautification",
        "programming", "programs", "programmer", "programmed"
    ];
    demonstrate_stemmer(&identity_stemmer, &complex_words)?;

    // Example 3: Technical terms preservation
    println!("\n3. Technical terms preservation:");
    let technical_words = vec![
        "optimization", "optimize", "optimized", "optimizing", "optimizer",
        "classification", "classify", "classified", "classifying", "classifier"
    ];
    demonstrate_stemmer(&identity_stemmer, &technical_words)?;

    // Example 4: StemFilter with IdentityStemmer
    println!("\n4. StemFilter with IdentityStemmer:");
    let identity_filter = StemFilter::with_stemmer(Box::new(IdentityStemmer::new()));
    let tokens = create_test_tokens(&[
        "running", "flies", "beautiful", "programming", "happiness"
    ]);
    demonstrate_filter(&identity_filter, tokens, "Token stream with no stemming")?;

    // Example 5: Exact matching scenario
    println!("\n5. Exact matching scenario:");
    let exact_words = vec![
        "JavaScript", "TypeScript", "WebAssembly", "MongoDB", "PostgreSQL"
    ];
    demonstrate_stemmer(&identity_stemmer, &exact_words)?;

    // Example 6: Proper nouns and names
    println!("\n6. Proper nouns and names:");
    let proper_nouns = vec![
        "London", "Paris", "Tokyo", "Microsoft", "Google", "Amazon"
    ];
    demonstrate_stemmer(&identity_stemmer, &proper_nouns)?;

    // Example 7: Code identifiers
    println!("\n7. Code identifiers:");
    let code_identifiers = vec![
        "getUserName", "API_ENDPOINT", "MAX_SIZE", "isValid", "toString", "valueOf"
    ];
    demonstrate_stemmer(&identity_stemmer, &code_identifiers)?;

    // Example 8: Debugging and development
    println!("\n8. Debugging and development:");
    let debug_tokens = create_test_tokens(&[
        "originalWord", "stemmedWord", "processedToken", "debugMode", "testCase"
    ]);
    demonstrate_filter(&identity_filter, debug_tokens, "Development and debugging tokens")?;

    // Example 9: Multilingual terms
    println!("\n9. Multilingual terms:");
    let multilingual_words = vec![
        "café", "naïve", "résumé", "piñata", "jalapeño", "Москва"
    ];
    demonstrate_stemmer(&identity_stemmer, &multilingual_words)?;

    // Example 10: Domain-specific vocabulary
    println!("\n10. Domain-specific vocabulary:");
    let domain_words = vec![
        "OAuth2", "RESTful", "GraphQL", "Kubernetes", "DevOps", "CI/CD"
    ];
    demonstrate_stemmer(&identity_stemmer, &domain_words)?;

    println!("\n=== Identity Algorithm Details ===\n");
    println!("Stemmer name: {}", identity_stemmer.name());
    println!("Algorithm type: No-operation (identity function)");
    
    println!("\n=== Algorithm Behavior ===\n");
    println!("Processing steps:");
    println!("  1. Receive input word");
    println!("  2. Return input word unchanged");
    println!("  3. No transformation applied");
    println!("  4. Perfect word preservation");
    
    println!("\nCharacteristics:");
    println!("  • Zero processing overhead");
    println!("  • Perfect accuracy (no changes)");
    println!("  • No risk of over-stemming");
    println!("  • Preserves all word forms");
    println!("  • Language-agnostic");

    println!("\n=== Performance Characteristics ===\n");
    println!("Speed: Fastest possible");
    println!("  • O(1) constant time complexity");
    println!("  • No string operations");
    println!("  • Minimal memory usage");
    println!("  • No CPU overhead");
    
    println!("\nAccuracy: Perfect preservation");
    println!("  • 100% word form preservation");
    println!("  • No false transformations");
    println!("  • No over-stemming issues");
    println!("  • Complete semantic preservation");

    println!("\n=== Use Cases ===\n");
    println!("Exact matching requirements:");
    println!("  • Legal document processing");
    println!("  • Contract and agreement analysis");
    println!("  • Compliance and audit systems");
    println!("  • Regulatory text processing");
    
    println!("\nTechnical documentation:");
    println!("  • API documentation indexing");
    println!("  • Code comment analysis");
    println!("  • Technical manual processing");
    println!("  • Software specification parsing");
    
    println!("\nDevelopment and testing:");
    println!("  • Debugging stemming pipelines");
    println!("  • A/B testing baseline");
    println!("  • Unit test control cases");
    println!("  • Algorithm comparison studies");
    
    println!("\nMultilingual applications:");
    println!("  • Languages without stemming rules");
    println!("  • Mixed-language documents");
    println!("  • International content processing");
    println!("  • Unicode text preservation");

    println!("\n=== When to Use Identity Stemmer ===\n");
    println!("Recommended scenarios:");
    println!("  • Exact term matching is critical");
    println!("  • Proper nouns must be preserved");
    println!("  • Technical identifiers need exact forms");
    println!("  • Testing and debugging stemming effects");
    println!("  • Baseline comparison for other algorithms");
    println!("  • Languages without morphological complexity");
    
    println!("\nNot recommended when:");
    println!("  • Text normalization is needed");
    println!("  • Vocabulary reduction is desired");
    println!("  • Fuzzy matching is required");
    println!("  • Search recall needs improvement");

    println!("\n=== Integration Patterns ===\n");
    println!("Pipeline testing:");
    println!("  1. Start with IdentityStemmer");
    println!("  2. Test functionality without stemming");
    println!("  3. Replace with SimpleStemmer for basic stemming");
    println!("  4. Upgrade to PorterStemmer for precision");
    
    println!("\nConditional stemming:");
    println!("  • Use Identity for proper nouns");
    println!("  • Use Porter for common words");
    println!("  • Use Simple for performance-critical paths");
    
    println!("\nA/B testing:");
    println!("  • Control group: IdentityStemmer");
    println!("  • Test group: Other stemming algorithms");
    println!("  • Compare search effectiveness");

    println!("\n=== Null Object Pattern ===\n");
    println!("Design pattern implementation:");
    println!("  • Provides default 'do nothing' behavior");
    println!("  • Eliminates need for null checks");
    println!("  • Maintains consistent interface");
    println!("  • Simplifies client code");
    
    println!("\nBenefits:");
    println!("  • No special case handling required");
    println!("  • Polymorphic behavior guaranteed");
    println!("  • Easy to substitute different stemmers");
    println!("  • Clean separation of concerns");

    println!("\n=== Performance Comparison ===\n");
    println!("Speed ranking (fastest to slowest):");
    println!("  1. IdentityStemmer: O(1) - No processing");
    println!("  2. SimpleStemmer: O(n) - Basic suffix removal");
    println!("  3. PorterStemmer: O(n) - Complex linguistic rules");
    
    println!("\nMemory usage:");
    println!("  • IdentityStemmer: Minimal (no data structures)");
    println!("  • SimpleStemmer: Low (suffix list storage)");
    println!("  • PorterStemmer: Low (algorithm state only)");

    Ok(())
}

fn create_test_tokens(texts: &[&str]) -> TokenStream {
    let tokens: Vec<Token> = texts.iter()
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

fn demonstrate_filter(
    filter: &dyn Filter,
    tokens: TokenStream,
    description: &str,
) -> Result<()> {
    println!("Description: {}", description);
    
    let input_tokens: Vec<Token> = tokens.collect();
    println!("Input tokens: {:?}", 
        input_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    let input_clone = input_tokens.clone();
    let input_stream = Box::new(input_tokens.into_iter());
    let filtered_tokens: Vec<Token> = filter.filter(input_stream)?.collect();
    
    println!("Output tokens: {:?}", 
        filtered_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    // Show the stemming transformations (should be none for Identity)
    println!("Transformations:");
    let mut changes = 0;
    for (i, (input, output)) in input_clone.iter()
        .zip(filtered_tokens.iter())
        .enumerate() {
        if input.text != output.text {
            println!("  {}: '{}' → '{}' (UNEXPECTED!)", i, input.text, output.text);
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