//! Simple example demonstrating synonym expansion with SynonymGraphFilter.
//!
//! This example shows how to:
//! - Create a synonym dictionary
//! - Apply synonym expansion to text
//! - Use boost to adjust synonym weights
//! - Observe how tokens are expanded with synonyms

use sage::analysis::synonym::dictionary::SynonymDictionary;
use sage::analysis::token_filter::Filter;
use sage::analysis::token_filter::synonym_graph::SynonymGraphFilter;
use sage::analysis::tokenizer::Tokenizer;
use sage::analysis::tokenizer::whitespace::WhitespaceTokenizer;
use sage::error::Result;

fn main() -> Result<()> {
    println!("=== SynonymGraphFilter Usage Example ===\n");

    // 1. Create a synonym dictionary
    println!("Step 1: Creating synonym dictionary");
    let mut dict = SynonymDictionary::new(None)?;

    // Add synonym groups (bidirectional)
    dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);
    dict.add_synonym_group(vec![
        "ai".to_string(),
        "artificial intelligence".to_string(),
    ]);

    println!("  ✓ Added synonyms:");
    println!("    - 'ml' ↔ 'machine learning'");
    println!("    - 'ai' ↔ 'artificial intelligence'\n");

    // 2. Example without boost
    println!("Step 2: Applying filter WITHOUT boost");
    let tokenizer = WhitespaceTokenizer;
    let synonym_filter = SynonymGraphFilter::new(dict.clone(), true); // keep_original=true

    let input_text = "ml tutorial";
    println!("  Input: \"{}\"\n", input_text);

    let tokens = tokenizer.tokenize(input_text)?;
    let result_tokens = synonym_filter.filter(tokens)?;

    println!("  Output tokens:");
    for (i, token) in result_tokens.enumerate() {
        println!(
            "    [{}] '{}' (pos={}, pos_inc={}, pos_len={}, boost={:.2})",
            i,
            token.text,
            token.position,
            token.position_increment,
            token.position_length,
            token.boost
        );
    }

    println!("\n  Explanation:");
    println!("    - All tokens have boost=1.0 (default)");
    println!("    - Synonyms have equal weight to original tokens\n");

    // 3. Example with boost
    println!("Step 3: Applying filter WITH boost=0.8");
    let synonym_filter_with_boost = SynonymGraphFilter::new(dict, true).with_boost(0.8); // Synonyms get 80% weight

    println!("  Input: \"{}\"\n", input_text);

    let tokens = tokenizer.tokenize(input_text)?;
    let result_tokens = synonym_filter_with_boost.filter(tokens)?;

    println!("  Output tokens:");
    for (i, token) in result_tokens.enumerate() {
        println!(
            "    [{}] '{}' (pos={}, pos_inc={}, pos_len={}, boost={:.2})",
            i,
            token.text,
            token.position,
            token.position_increment,
            token.position_length,
            token.boost
        );
    }

    println!("\n  Explanation:");
    println!("    - Original token 'ml' has boost=1.0");
    println!("    - 'machine' has boost=0.72 (0.9 × 0.8) - first token of multi-word synonym");
    println!("    - 'learning' has boost=0.64 (0.8 × 0.8) - second token");
    println!("    - Lower boost means synonyms contribute less to the final score");
    println!("    - This helps prioritize exact matches over synonym matches\n");

    println!("Use cases for boost:");
    println!("  - boost=0.8: Synonyms have 80% weight (common default)");
    println!("  - boost=0.5: Synonyms have 50% weight (conservative)");
    println!("  - boost=1.0: Synonyms equal to originals (no adjustment)\n");

    Ok(())
}
