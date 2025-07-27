//! Example demonstrating the SimpleStemmer algorithm.

use sarissa::analysis::token::{Token, TokenStream};
use sarissa::analysis::token_filter::{Filter, SimpleStemmer, StemFilter, Stemmer};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== SimpleStemmer Examples ===\n");

    let simple_stemmer = SimpleStemmer::new();

    // Example 1: Basic suffix removal
    println!("1. Basic suffix removal:");
    let basic_words = vec![
        "running", "runs", "ran", "runner",
        "flies", "flying", "fly", "flew"
    ];
    demonstrate_stemmer(&simple_stemmer, &basic_words)?;

    // Example 2: Common English suffixes
    println!("\n2. Common English suffixes:");
    let common_words = vec![
        "walking", "walked", "walker", "walks",
        "talking", "talked", "talker", "talks",
        "quickly", "slowly", "carefully", "easily"
    ];
    demonstrate_stemmer(&simple_stemmer, &common_words)?;

    // Example 3: Programming and technical terms
    println!("\n3. Programming and technical terms:");
    let tech_words = vec![
        "programming", "programs", "programmer", "programmed",
        "testing", "tests", "tester", "tested",
        "building", "builds", "builder", "built"
    ];
    demonstrate_stemmer(&simple_stemmer, &tech_words)?;

    // Example 4: Custom SimpleStemmer with specific suffixes
    println!("\n4. Custom SimpleStemmer:");
    let custom_stemmer = SimpleStemmer::with_suffixes(vec![
        "ing".to_string(), "ed".to_string(), "er".to_string(), "ly".to_string()
    ]);
    let custom_words = vec!["running", "walked", "bigger", "quickly", "testing"];
    demonstrate_custom_stemmer(&custom_stemmer, &custom_words)?;

    // Example 5: Minimal custom stemmer (only -ing and -ed)
    println!("\n5. Minimal custom stemmer:");
    let minimal_stemmer = SimpleStemmer::with_suffixes(vec![
        "ing".to_string(), "ed".to_string()
    ]);
    let minimal_words = vec![
        "reading", "writing", "coded", "tested", "runner", "faster"
    ];
    demonstrate_custom_stemmer(&minimal_stemmer, &minimal_words)?;

    // Example 6: Extended suffix list
    println!("\n6. Extended suffix list:");
    let extended_stemmer = SimpleStemmer::with_suffixes(vec![
        "ing".to_string(), "ed".to_string(), "er".to_string(), "est".to_string(),
        "ly".to_string(), "ness".to_string(), "ment".to_string(), "tion".to_string()
    ]);
    let extended_words = vec![
        "fastest", "kindness", "development", "creation", "effectively"
    ];
    demonstrate_custom_stemmer(&extended_stemmer, &extended_words)?;

    // Example 7: StemFilter with SimpleStemmer
    println!("\n7. StemFilter with SimpleStemmer:");
    let simple_filter = StemFilter::simple();
    let tokens = create_test_tokens(&[
        "running", "flies", "beautiful", "programming", "happiness", "connection"
    ]);
    demonstrate_filter(&simple_filter, tokens, "Token stream with simple stemming")?;

    // Example 8: Performance comparison data
    println!("\n8. Performance comparison:");
    let perf_words = vec![
        "optimization", "optimizing", "optimized", "optimizer",
        "classification", "classifying", "classified", "classifier",
        "normalization", "normalizing", "normalized", "normalizer"
    ];
    demonstrate_stemmer(&simple_stemmer, &perf_words)?;

    // Example 9: Real-world text processing
    println!("\n9. Real-world text processing:");
    let real_tokens = create_test_tokens(&[
        "The", "developers", "are", "programming", "applications",
        "using", "advanced", "algorithms", "and", "optimizations"
    ]);
    demonstrate_filter(&simple_filter, real_tokens, "Document with technical terms")?;

    // Example 10: Edge cases and limitations
    println!("\n10. Edge cases and limitations:");
    let edge_words = vec![
        "a", "I", "am", "go", "be", "do", "it", "running", "sing", "ring"
    ];
    demonstrate_stemmer(&simple_stemmer, &edge_words)?;

    println!("\n=== Simple Algorithm Details ===\n");
    println!("Stemmer name: {}", simple_stemmer.name());
    println!("Algorithm type: Suffix removal");
    
    println!("\n=== Default Suffix List ===\n");
    println!("The SimpleStemmer removes these suffixes by default:");
    println!("  • 'ing' (e.g., 'running' → 'runn')");
    println!("  • 'ly' (e.g., 'quickly' → 'quick')");
    println!("  • 'ed' (e.g., 'walked' → 'walk')");
    println!("  • 'ies' (e.g., 'flies' → 'fl')");
    println!("  • 'ied' (e.g., 'tried' → 'tr')");
    println!("  • 'ying' (e.g., 'flying' → 'fly')");
    println!("  • 'ion' (e.g., 'creation' → 'creat')");
    println!("  • 'er' (e.g., 'runner' → 'runn')");
    println!("  • 'est' (e.g., 'fastest' → 'fast')");

    println!("\n=== Algorithm Behavior ===\n");
    println!("Processing steps:");
    println!("  1. Check if word ends with any configured suffix");
    println!("  2. Remove the longest matching suffix");
    println!("  3. Return the remaining stem");
    println!("  4. No linguistic analysis or validation");
    
    println!("\nCharacteristics:");
    println!("  • Fast O(n) processing (n = word length)");
    println!("  • No dictionary lookup required");
    println!("  • Simple pattern matching");
    println!("  • May produce non-words (over-stemming)");
    println!("  • Configurable suffix list");

    println!("\n=== Performance Characteristics ===\n");
    println!("Speed: Very fast");
    println!("  • Simple string operations only");
    println!("  • No complex linguistic rules");
    println!("  • Suitable for real-time processing");
    println!("  • Minimal CPU and memory usage");
    
    println!("\nAccuracy: Moderate");
    println!("  • Good for common word patterns");
    println!("  • May over-stem irregular words");
    println!("  • No semantic understanding");
    println!("  • Context-independent");

    println!("\n=== Use Cases ===\n");
    println!("High-performance applications:");
    println!("  • Real-time search suggestions");
    println!("  • Chat and messaging systems");
    println!("  • Live content filtering");
    println!("  • Large-scale batch processing");
    
    println!("\nBasic text processing:");
    println!("  • Simple search functionality");
    println!("  • Keyword matching");
    println!("  • Content categorization");
    println!("  • Tag normalization");
    
    println!("\nPrototyping and development:");
    println!("  • Quick proof-of-concept implementations");
    println!("  • Testing stemming effects");
    println!("  • Development and debugging");
    println!("  • Educational demonstrations");

    println!("\n=== Customization Examples ===\n");
    println!("Domain-specific stemming:");
    println!("  • Code analysis: remove 'ing', 'er', 'ed'");
    println!("  • Social media: remove 'ing', 'ly', 'er'");
    println!("  • Technical docs: include 'tion', 'ment', 'ness'");
    
    println!("\nLanguage adaptation:");
    println!("  • Minimal English: 'ing', 'ed', 's'");
    println!("  • Extended English: include 'ness', 'ment', 'able'");
    println!("  • Custom languages: language-specific suffixes");

    println!("\n=== Comparison with Porter ===\n");
    println!("SimpleStemmer advantages:");
    println!("  • Much faster processing");
    println!("  • Easy to understand and modify");
    println!("  • Configurable suffix rules");
    println!("  • Minimal resource usage");
    
    println!("\nPorterStemmer advantages:");
    println!("  • Higher accuracy for English");
    println!("  • Linguistic rule-based approach");
    println!("  • Less over-stemming");
    println!("  • Better semantic preservation");

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
    println!("Stemmer: {} (Suffix removal)", stemmer.name());
    
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

fn demonstrate_custom_stemmer(stemmer: &SimpleStemmer, words: &[&str]) -> Result<()> {
    println!("Custom SimpleStemmer with specific suffixes:");
    
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