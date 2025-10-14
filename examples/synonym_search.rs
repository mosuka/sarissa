//! Simple example demonstrating synonym expansion with SynonymGraphFilter.
//!
//! This example shows how to:
//! - Create a synonym dictionary
//! - Apply synonym expansion to text
//! - Observe how tokens are expanded with synonyms

use sarissa::analysis::token_filter::{Filter, SynonymDictionary, SynonymGraphFilter};
use sarissa::analysis::tokenizer::{Tokenizer, WhitespaceTokenizer};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== Synonym Search Example ===\n");

    // 1. Create a synonym dictionary
    println!("1. Creating synonym dictionary...");
    let mut dict = SynonymDictionary::new(None)?;

    // Add synonym groups
    dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);

    dict.add_synonym_group(vec![
        "ai".to_string(),
        "artificial intelligence".to_string(),
    ]);

    println!("   ✓ Added synonyms:");
    println!("     - 'ml' ↔ 'machine learning'");
    println!("     - 'ai' ↔ 'artificial intelligence'\n");

    // 2. Case 1: Single word → Multi-word synonym expansion
    println!("2. Case 1: Single word → Multi-word synonym expansion");
    println!("   Processing text: \"ml and ai tutorial\"\n");

    let tokenizer = WhitespaceTokenizer;
    let tokens = tokenizer.tokenize("ml and ai tutorial")?;

    let synonym_filter = SynonymGraphFilter::new(dict.clone(), true);
    let result_tokens = synonym_filter.filter(tokens)?;

    println!("   Resulting tokens:");
    for (i, token) in result_tokens.enumerate() {
        println!(
            "     [{}] text='{}' position={} pos_inc={} pos_len={}",
            i, token.text, token.position, token.position_increment, token.position_length
        );
    }

    // 3. Case 2: Multi-word → Single word synonym expansion
    println!("\n3. Case 2: Multi-word → Single word synonym expansion");
    println!("   Processing text: \"machine learning tutorial\"\n");

    let tokens2 = tokenizer.tokenize("machine learning tutorial")?;
    let synonym_filter2 = SynonymGraphFilter::new(dict, true);
    let result_tokens2 = synonym_filter2.filter(tokens2)?;

    println!("   Resulting tokens:");
    for (i, token) in result_tokens2.enumerate() {
        println!(
            "     [{}] text='{}' position={} pos_inc={} pos_len={}",
            i, token.text, token.position, token.position_increment, token.position_length
        );
    }

    Ok(())
}
