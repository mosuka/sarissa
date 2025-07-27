//! Example demonstrating the PorterStemmer algorithm.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{Filter, PorterStemmer, StemFilter, Stemmer};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== PorterStemmer Examples ===\n");

    let porter_stemmer = PorterStemmer::new();

    // Example 1: Basic word stemming
    println!("1. Basic word stemming:");
    let basic_words = vec![
        "running", "runs", "ran", "runner",
        "flies", "flying", "fly", "flew"
    ];
    demonstrate_stemmer(&porter_stemmer, &basic_words)?;

    // Example 2: Complex morphological analysis
    println!("\n2. Complex morphological analysis:");
    let complex_words = vec![
        "beautiful", "beautifully", "beauty", "beautification",
        "programming", "programs", "programmer", "programmed"
    ];
    demonstrate_stemmer(&porter_stemmer, &complex_words)?;

    // Example 3: Word family normalization
    println!("\n3. Word family normalization:");
    let family_words = vec![
        "happiness", "happy", "happily", "happier",
        "connection", "connected", "connecting", "connect"
    ];
    demonstrate_stemmer(&porter_stemmer, &family_words)?;

    // Example 4: Long words with multiple suffixes
    println!("\n4. Long words with multiple suffixes:");
    let long_words = vec![
        "organization", "organize", "organized", "organizing",
        "traditional", "tradition", "traditionally",
        "itemization", "itemize", "items",
        "sensational", "sensation", "sensations"
    ];
    demonstrate_stemmer(&porter_stemmer, &long_words)?;

    // Example 5: Technical and scientific terms
    println!("\n5. Technical and scientific terms:");
    let technical_words = vec![
        "optimization", "optimize", "optimized", "optimizing", "optimizer",
        "classification", "classify", "classified", "classifying", "classifier",
        "normalization", "normalize", "normalized", "normalizing"
    ];
    demonstrate_stemmer(&porter_stemmer, &technical_words)?;

    // Example 6: Past tense and participles
    println!("\n6. Past tense and participles:");
    let tense_words = vec![
        "created", "creating", "creation", "creative", "creator",
        "developed", "developing", "development", "developer",
        "implemented", "implementing", "implementation"
    ];
    demonstrate_stemmer(&porter_stemmer, &tense_words)?;

    // Example 7: Comparative and superlative forms
    println!("\n7. Comparative and superlative forms:");
    let comparative_words = vec![
        "faster", "fastest", "slow", "slower", "slowest",
        "better", "best", "good", "worse", "worst"
    ];
    demonstrate_stemmer(&porter_stemmer, &comparative_words)?;

    // Example 8: StemFilter with PorterStemmer
    println!("\n8. StemFilter with PorterStemmer:");
    let porter_filter = StemFilter::new(); // Default uses Porter
    let tokens = create_test_tokens(&[
        "running", "flies", "beautiful", "programming", "happiness", "connection"
    ]);
    demonstrate_filter(&porter_filter, tokens, "Token stream with Porter stemming")?;

    // Example 9: Real-world document processing
    println!("\n9. Real-world document processing:");
    let document_tokens = create_test_tokens(&[
        "The", "developers", "are", "programming", "applications",
        "using", "advanced", "algorithms", "and", "optimizations"
    ]);
    demonstrate_filter(&porter_filter, document_tokens, "Document with technical terms")?;

    // Example 10: Research paper keywords
    println!("\n10. Research paper keywords:");
    let research_words = vec![
        "methodological", "systematic", "empirical", "theoretical",
        "experimental", "analytical", "statistical", "computational"
    ];
    demonstrate_stemmer(&porter_stemmer, &research_words)?;

    println!("\n=== Porter Algorithm Details ===\n");
    println!("Stemmer name: {}", porter_stemmer.name());
    println!("Algorithm type: Linguistic morphological analysis");
    
    println!("\n=== Porter Stemming Rules ===\n");
    println!("The Porter algorithm applies rules in multiple steps:");
    println!("Step 1a: Deal with plurals and past participles");
    println!("  • 'sses' → 'ss' (e.g., 'stresses' → 'stress')");
    println!("  • 'ies' → 'i' (e.g., 'ponies' → 'poni')");
    println!("  • 'ss' → 'ss' (e.g., 'stress' → 'stress')");
    println!("  • 's' → '' (e.g., 'cats' → 'cat')");
    
    println!("\nStep 1b: Deal with '-ed' and '-ing'");
    println!("  • 'eed' → 'ee' (e.g., 'agreed' → 'agree')");
    println!("  • 'ing' → '' (e.g., 'running' → 'runn')");
    println!("  • 'ed' → '' (e.g., 'plastered' → 'plaster')");
    
    println!("\nStep 2-5: Handle various suffixes");
    println!("  • 'ational' → 'ate' (e.g., 'relational' → 'relate')");
    println!("  • 'iveness' → 'ive' (e.g., 'effectiveness' → 'effective')");
    println!("  • 'fulness' → 'ful' (e.g., 'usefulness' → 'useful')");

    println!("\n=== Accuracy and Performance ===\n");
    println!("Accuracy characteristics:");
    println!("  • High precision for English words");
    println!("  • Handles complex morphology correctly");
    println!("  • Reduces over-stemming compared to simple algorithms");
    println!("  • Maintains semantic relationships");
    
    println!("\nPerformance characteristics:");
    println!("  • Slower than simple suffix removal");
    println!("  • O(n) complexity where n is word length");
    println!("  • Suitable for offline processing");
    println!("  • Memory efficient");

    println!("\n=== Use Cases ===\n");
    println!("Academic and research:");
    println!("  • Literature analysis and text mining");
    println!("  • Information retrieval systems");
    println!("  • Natural language processing research");
    println!("  • Linguistic corpus analysis");
    
    println!("\nCommercial applications:");
    println!("  • Search engines (high-quality results)");
    println!("  • Document management systems");
    println!("  • Content recommendation engines");
    println!("  • Text classification systems");
    
    println!("\nSpecialized domains:");
    println!("  • Legal document processing");
    println!("  • Medical text analysis");
    println!("  • Technical documentation indexing");
    println!("  • Academic paper analysis");

    println!("\n=== When NOT to Use Porter ===\n");
    println!("Consider alternatives when:");
    println!("  • Real-time processing is critical");
    println!("  • Non-English languages are primary");
    println!("  • Exact word matching is required");
    println!("  • Very large-scale processing (billions of words)");
    println!("  • Domain-specific terminology needs preservation");

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
    println!("Stemmer: {} (Linguistic algorithm)", stemmer.name());
    
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
    
    // Show the stemming transformations
    println!("Transformations:");
    let mut changes = 0;
    for (i, (input, output)) in input_clone.iter()
        .zip(filtered_tokens.iter())
        .enumerate() {
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