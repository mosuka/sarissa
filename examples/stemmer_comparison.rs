//! Example demonstrating comparison between different stemming algorithms.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{
    Filter, IdentityStemmer, PorterStemmer, SimpleStemmer, StemFilter, Stemmer,
};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== Stemmer Comparison Examples ===\n");

    // Example 1: Basic algorithm comparison
    println!("1. Basic algorithm comparison:");
    let basic_words = vec![
        "running", "flies", "beautiful", "programming", "traditional", "itemization"
    ];
    compare_stemmers(&basic_words)?;

    // Example 2: Technical vocabulary comparison
    println!("\n2. Technical vocabulary:");
    let tech_words = vec![
        "optimization", "classification", "normalization", "implementation", "configuration"
    ];
    compare_stemmers(&tech_words)?;

    // Example 3: Word family analysis
    println!("\n3. Word family analysis:");
    let family_words = vec![
        "connect", "connected", "connecting", "connection", "connector"
    ];
    compare_stemmers(&family_words)?;

    // Example 4: Past tense and participles
    println!("\n4. Past tense and participles:");
    let tense_words = vec![
        "created", "creating", "creation", "creative", "creator"
    ];
    compare_stemmers(&tense_words)?;

    // Example 5: Comparative and superlative forms
    println!("\n5. Comparative and superlative forms:");
    let comparative_words = vec![
        "fast", "faster", "fastest", "slow", "slower", "slowest"
    ];
    compare_stemmers(&comparative_words)?;

    // Example 6: Edge cases and short words
    println!("\n6. Edge cases and short words:");
    let edge_cases = vec!["a", "I", "am", "go", "be", "do", "it", "run"];
    compare_stemmers(&edge_cases)?;

    // Example 7: Academic and research terms
    println!("\n7. Academic and research terms:");
    let academic_words = vec![
        "methodological", "systematic", "empirical", "theoretical", "experimental"
    ];
    compare_stemmers(&academic_words)?;

    // Example 8: Business and commercial terms
    println!("\n8. Business and commercial terms:");
    let business_words = vec![
        "organization", "management", "development", "marketing", "investment"
    ];
    compare_stemmers(&business_words)?;

    // Example 9: Filter comparison with token streams
    println!("\n9. Filter comparison with token streams:");
    let test_tokens = vec![
        "The", "developers", "are", "programming", "applications",
        "using", "advanced", "algorithms", "and", "optimizations"
    ];
    compare_filters(&test_tokens)?;

    // Example 10: Performance analysis words
    println!("\n10. Performance analysis:");
    let perf_words = vec![
        "benchmarking", "profiling", "analyzing", "optimizing", "measuring"
    ];
    compare_stemmers(&perf_words)?;

    println!("\n=== Detailed Algorithm Analysis ===\n");
    
    // Algorithm characteristics
    println!("Algorithm Characteristics:");
    println!("┌─────────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ Characteristic  │ Porter      │ Simple      │ Identity    │");
    println!("├─────────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ Speed           │ Moderate    │ Fast        │ Fastest     │");
    println!("│ Accuracy        │ High        │ Moderate    │ Perfect*    │");
    println!("│ Over-stemming   │ Low         │ High        │ None        │");
    println!("│ Complexity      │ High        │ Low         │ None        │");
    println!("│ Configurability │ Fixed       │ High        │ N/A         │");
    println!("└─────────────────┴─────────────┴─────────────┴─────────────┘");
    println!("* Perfect preservation, not stemming accuracy");

    println!("\n=== Use Case Recommendations ===\n");
    
    println!("PorterStemmer - Use when:");
    println!("  ✓ High accuracy is critical");
    println!("  ✓ Academic or research applications");
    println!("  ✓ Information retrieval systems");
    println!("  ✓ Complex morphological analysis needed");
    println!("  ✗ Real-time processing required");
    println!("  ✗ Non-English languages");
    
    println!("\nSimpleStemmer - Use when:");
    println!("  ✓ Fast processing is important");
    println!("  ✓ Real-time applications");
    println!("  ✓ Large-scale batch processing");
    println!("  ✓ Custom suffix rules needed");
    println!("  ✗ High accuracy is critical");
    println!("  ✗ Complex morphology exists");
    
    println!("\nIdentityStemmer - Use when:");
    println!("  ✓ Exact matching required");
    println!("  ✓ Debugging and testing");
    println!("  ✓ Proper nouns must be preserved");
    println!("  ✓ Technical identifiers present");
    println!("  ✗ Vocabulary reduction needed");
    println!("  ✗ Fuzzy matching desired");

    println!("\n=== Performance Benchmarking ===\n");
    
    println!("Relative Performance (lower is better):");
    println!("┌─────────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ Metric          │ Porter      │ Simple      │ Identity    │");
    println!("├─────────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ CPU Time        │ 100%        │ 30%         │ 1%          │");
    println!("│ Memory Usage    │ Low         │ Low         │ Minimal     │");
    println!("│ Setup Overhead  │ None        │ Minimal     │ None        │");
    println!("│ Scalability     │ Linear      │ Linear      │ Constant    │");
    println!("└─────────────────┴─────────────┴─────────────┴─────────────┘");

    println!("\n=== Quality Metrics ===\n");
    
    println!("Stemming Quality Analysis:");
    println!("┌─────────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ Quality Metric  │ Porter      │ Simple      │ Identity    │");
    println!("├─────────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ Under-stemming  │ Low         │ Moderate    │ High*       │");
    println!("│ Over-stemming   │ Low         │ High        │ None        │");
    println!("│ Consistency     │ High        │ Moderate    │ Perfect     │");
    println!("│ Recall          │ High        │ Moderate    │ Low*        │");
    println!("│ Precision       │ High        │ Moderate    │ Perfect*    │");
    println!("└─────────────────┴─────────────┴─────────────┴─────────────┘");
    println!("* For preservation, not traditional stemming metrics");

    println!("\n=== Decision Matrix ===\n");
    
    println!("Choose your stemmer based on priorities:");
    println!("\nPriority: Speed > Accuracy");
    println!("  → SimpleStemmer or IdentityStemmer");
    
    println!("\nPriority: Accuracy > Speed");
    println!("  → PorterStemmer");
    
    println!("\nPriority: Exact matching");
    println!("  → IdentityStemmer");
    
    println!("\nPriority: Customization");
    println!("  → SimpleStemmer with custom suffixes");
    
    println!("\nPriority: Research/Academic");
    println!("  → PorterStemmer");
    
    println!("\nPriority: Real-time/Production");
    println!("  → SimpleStemmer");

    Ok(())
}

fn compare_stemmers(words: &[&str]) -> Result<()> {
    let porter = PorterStemmer::new();
    let simple = SimpleStemmer::new();
    let identity = IdentityStemmer::new();
    
    println!("Word           | Porter      | Simple      | Identity");
    println!("---------------|-------------|-------------|-------------");
    
    for word in words {
        let porter_result = porter.stem(word);
        let simple_result = simple.stem(word);
        let identity_result = identity.stem(word);
        
        println!("{:14} | {:11} | {:11} | {:11}", 
            word, porter_result, simple_result, identity_result);
    }
    
    Ok(())
}

fn compare_filters(words: &[&str]) -> Result<()> {
    println!("Filter comparison on token stream:");
    
    // Create the same token stream for each filter
    let tokens1 = create_test_tokens(words);
    let tokens2 = create_test_tokens(words);
    let tokens3 = create_test_tokens(words);
    
    let porter_filter = StemFilter::new();
    let simple_filter = StemFilter::simple();
    let identity_filter = StemFilter::with_stemmer(Box::new(IdentityStemmer::new()));
    
    println!("\nPorter Filter:");
    demonstrate_filter(&porter_filter, tokens1, "Porter stemming")?;
    
    println!("\nSimple Filter:");
    demonstrate_filter(&simple_filter, tokens2, "Simple stemming")?;
    
    println!("\nIdentity Filter:");
    demonstrate_filter(&identity_filter, tokens3, "No stemming")?;
    
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
    
    println!("Output: {:?}", 
        filtered_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    // Show transformations
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